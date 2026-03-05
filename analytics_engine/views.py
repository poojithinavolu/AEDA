from io import BytesIO

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
import plotly.graph_objects as go

from projects.permissions import can_edit_project, get_accessible_datasets
from projects.models import Dataset

from .forms import ModelSelectionForm, ModelTrainingForm
from .models import AnalysisRun, DatasetCopilotMessage, MLModelRun, Visualization
from .services.analysis_service import analyze_dataset_and_create_run
from .services.copilot_service import ask_dataset_copilot
from .services.ml_service import predict_with_trained_model, train_model
from .services.visualization_service import build_chart_html_list
from .utils.data_io import load_dataset_frame


def _build_confusion_matrix_html(confusion_matrix_json):
    if not confusion_matrix_json:
        return ""
    matrix = confusion_matrix_json
    size = len(matrix)
    labels = [f"Class {i}" for i in range(size)]
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=matrix,
            texttemplate="%{text}",
            showscale=True,
        )
    )
    fig.update_layout(
        title="Confusion Matrix Heatmap",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _build_roc_curve_html(roc_curve_json):
    if not roc_curve_json:
        return ""
    fpr = roc_curve_json.get("fpr")
    tpr = roc_curve_json.get("tpr")
    if not fpr or not tpr:
        return ""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve"))
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Baseline",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _get_dataset_or_404_for_user(user, dataset_id: int):
    return get_object_or_404(get_accessible_datasets(user).select_related("project"), id=dataset_id)


@login_required
def auto_insights_view(request, dataset_id: int):
    dataset = _get_dataset_or_404_for_user(request.user, dataset_id)
    run = AnalysisRun.objects.filter(dataset=dataset).order_by("-created_at").first()

    if request.method == "POST":
        if not can_edit_project(dataset.project, request.user):
            messages.error(request, "You do not have edit permission for this project.")
            return redirect("auto_insights", dataset_id=dataset.id)
        run = analyze_dataset_and_create_run(dataset=dataset, user=request.user)
        return redirect("auto_insights", dataset_id=dataset.id)

    feature_plan_items = list((run.llm_feature_plan or {}).items()) if run else []
    applied_items = list((run.applied_transformations or {}).items()) if run else []
    return render(
        request,
        "analytics_engine/auto_insights.html",
        {
            "dataset": dataset,
            "run": run,
            "feature_plan_items": feature_plan_items,
            "applied_items": applied_items,
        },
    )


@login_required
def visualization_list_view(request, dataset_id: int):
    dataset = _get_dataset_or_404_for_user(request.user, dataset_id)
    visualizations = Visualization.objects.filter(dataset=dataset)
    return render(
        request,
        "analytics_engine/visualizations.html",
        {"dataset": dataset, "visualizations": visualizations},
    )


@login_required
def dataset_copilot_view(request, dataset_id: int):
    dataset = _get_dataset_or_404_for_user(request.user, dataset_id)
    chat_messages_qs = DatasetCopilotMessage.objects.filter(dataset=dataset, owner=request.user).order_by("created_at")
    chat_messages = list(chat_messages_qs)

    if request.method == "POST":
        question = (request.POST.get("question") or "").strip()
        if question:
            DatasetCopilotMessage.objects.create(
                owner=request.user,
                dataset=dataset,
                role=DatasetCopilotMessage.ROLE_USER,
                content=question,
                metadata_json={},
            )
            history = [{"role": m.role, "content": m.content} for m in chat_messages[-6:]]
            result = ask_dataset_copilot(dataset.summary_json or {}, question, history)
            DatasetCopilotMessage.objects.create(
                owner=request.user,
                dataset=dataset,
                role=DatasetCopilotMessage.ROLE_ASSISTANT,
                content=result["answer"],
                metadata_json={"charts": result.get("charts", [])},
            )
        return redirect("dataset_copilot", dataset_id=dataset.id)

    frame = load_dataset_frame(dataset.file.path, dataset.file_type)
    rendered_messages = []
    for message in chat_messages[-16:]:
        charts = message.metadata_json.get("charts", []) if message.metadata_json else []
        chart_html_items = build_chart_html_list(frame, charts) if charts else []
        rendered_messages.append({"obj": message, "chart_html_items": chart_html_items})

    return render(
        request,
        "analytics_engine/dataset_copilot.html",
        {"dataset": dataset, "rendered_messages": rendered_messages},
    )


@login_required
def model_training_view(request):
    form = ModelTrainingForm(request.POST or None, user=request.user)
    model_run = None

    if request.method == "POST" and form.is_valid():
        dataset = form.cleaned_data["dataset"]
        if not can_edit_project(dataset.project, request.user):
            messages.error(request, "You do not have edit permission for this project.")
            return redirect("model_training")

        training_config = {
            "test_size": form.cleaned_data["test_size"],
            "random_state": form.cleaned_data["random_state"],
            "cv_folds": form.cleaned_data["cv_folds"],
            "max_iter": form.cleaned_data["max_iter"],
            "n_estimators": form.cleaned_data["n_estimators"],
            "svm_c": form.cleaned_data["svm_c"],
            "svm_kernel": form.cleaned_data["svm_kernel"],
            "auto_tune": form.cleaned_data["auto_tune"],
        }
        try:
            model_run = train_model(
                dataset=dataset,
                owner=request.user,
                model_type=form.cleaned_data["model_type"],
                target_column=form.cleaned_data["target_column"],
                training_config=training_config,
            )
            return redirect("model_results", run_id=model_run.id)
        except ValueError as exc:
            messages.error(request, str(exc))
        except Exception:
            messages.error(request, "Model training failed due to an unexpected error. Check inputs and try again.")

    return render(request, "analytics_engine/model_training.html", {"form": form, "model_run": model_run})


@login_required
def model_results_view(request, run_id: int):
    model_run = get_object_or_404(MLModelRun.objects.select_related("dataset", "dataset__project"), id=run_id)
    if not get_accessible_datasets(request.user).filter(id=model_run.dataset_id).exists():
        return redirect("dashboard")

    history = MLModelRun.objects.filter(dataset__in=get_accessible_datasets(request.user)).select_related("dataset")[:10]
    metrics = model_run.metrics_json or {}

    training_config = metrics.get("training_config", {})
    training_config_items = list(training_config.items())
    headline_metric_keys = [
        "r2",
        "mae",
        "mse",
        "rmse",
        "accuracy",
        "f1_score",
        "precision",
        "recall",
        "auc",
        "train_score",
        "test_score",
        "cv_mean",
        "cv_std",
    ]
    headline_metrics = [(k, metrics.get(k)) for k in headline_metric_keys if k in metrics]
    notes = metrics.get("notes", [])
    prediction_preview = metrics.get("prediction_preview", [])
    task_type = metrics.get("task_type", "")
    confusion_matrix_html = _build_confusion_matrix_html(model_run.confusion_matrix_json)
    roc_curve_html = _build_roc_curve_html(model_run.roc_curve_json)
    explainability = metrics.get("explainability", {})

    return render(
        request,
        "analytics_engine/model_results.html",
        {
            "model_run": model_run,
            "history": history,
            "metrics": metrics,
            "task_type": task_type,
            "training_config_items": training_config_items,
            "headline_metrics": headline_metrics,
            "notes": notes,
            "prediction_preview": prediction_preview,
            "confusion_matrix_html": confusion_matrix_html,
            "roc_curve_html": roc_curve_html,
            "explainability": explainability,
        },
    )


def _coerce_input_value(raw_value: str, dtype_text: str):
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if value == "":
        return None

    dtype = (dtype_text or "").lower()
    try:
        if "int" in dtype:
            return int(float(value))
        if "float" in dtype:
            return float(value)
    except ValueError:
        return value
    return value


@login_required
def model_predict_view(request):
    selected_dataset_id = request.GET.get("dataset") or request.POST.get("dataset")
    dataset_id = int(selected_dataset_id) if str(selected_dataset_id).isdigit() else None
    initial = {}
    run_id_qs = request.GET.get("run")
    if dataset_id:
        initial["dataset"] = dataset_id
    if run_id_qs and str(run_id_qs).isdigit():
        initial["model_run"] = int(run_id_qs)
    selection_form = ModelSelectionForm(request.POST or None, user=request.user, dataset_id=dataset_id, initial=initial)

    selected_run = None
    feature_fields = []
    prediction_result = None
    input_payload = {}

    if request.method == "POST" and selection_form.is_valid():
        selected_run = selection_form.cleaned_data["model_run"]
        selected_dataset = selection_form.cleaned_data["dataset"]
        if selected_run.dataset_id != selected_dataset.id:
            messages.error(request, "Invalid dataset/model run selection.")
            return redirect("model_predict")

        schema_dtypes = selected_dataset.schema_json.get("dtypes", {}) if selected_dataset.schema_json else {}
        for col in selected_run.feature_columns or []:
            dtype = schema_dtypes.get(col, "object")
            field_name = f"feature__{col}"
            raw_val = request.POST.get(field_name, "")
            cast_val = _coerce_input_value(raw_val, dtype)
            input_payload[col] = cast_val
            feature_fields.append({"name": col, "dtype": dtype, "value": raw_val})

        try:
            prediction_result = predict_with_trained_model(selected_run, input_payload)
        except ValueError as exc:
            messages.error(request, str(exc))
        except Exception:
            messages.error(request, "Prediction failed unexpectedly for this model run.")
    elif dataset_id:
        dataset = get_accessible_datasets(request.user).filter(id=dataset_id).first()
        run_id = request.GET.get("run")
        if dataset and run_id and str(run_id).isdigit():
            selected_run = MLModelRun.objects.filter(dataset=dataset, id=int(run_id)).first()
            if selected_run:
                schema_dtypes = dataset.schema_json.get("dtypes", {}) if dataset.schema_json else {}
                for col in selected_run.feature_columns or []:
                    feature_fields.append({"name": col, "dtype": schema_dtypes.get(col, "object"), "value": ""})

    recent_runs = MLModelRun.objects.filter(dataset__in=get_accessible_datasets(request.user)).select_related("dataset")[:25]
    return render(
        request,
        "analytics_engine/model_predict.html",
        {
            "selection_form": selection_form,
            "selected_run": selected_run,
            "feature_fields": feature_fields,
            "prediction_result": prediction_result,
            "input_payload": input_payload,
            "recent_runs": recent_runs,
        },
    )


@login_required
def model_leaderboard_view(request):
    metric = request.GET.get("metric", "test_score")
    allowed_metrics = ["test_score", "cv_mean", "accuracy", "f1_score", "auc", "r2", "rmse"]
    if metric not in allowed_metrics:
        metric = "test_score"

    runs = MLModelRun.objects.filter(dataset__in=get_accessible_datasets(request.user)).select_related("dataset", "dataset__project")
    rows = []
    for run in runs:
        metrics = run.metrics_json or {}
        rows.append(
            {
                "run": run,
                "dataset_version": run.dataset.version,
                "dataset_hash": run.dataset.dataset_hash[:12] if run.dataset.dataset_hash else "-",
                "score": metrics.get(metric),
                "test_score": metrics.get("test_score"),
                "cv_mean": metrics.get("cv_mean"),
                "accuracy": metrics.get("accuracy"),
                "f1_score": metrics.get("f1_score"),
                "auc": metrics.get("auc"),
                "r2": metrics.get("r2"),
                "rmse": metrics.get("rmse"),
                "training_config": metrics.get("training_config", {}),
            }
        )

    lower_is_better = metric in {"rmse"}

    def _sort_key(item):
        val = item["score"]
        if isinstance(val, (int, float)):
            return (0, val if lower_is_better else -val)
        return (1, 0)

    rows.sort(key=_sort_key)
    return render(
        request,
        "analytics_engine/model_leaderboard.html",
        {
            "rows": rows,
            "selected_metric": metric,
            "allowed_metrics": allowed_metrics,
        },
    )


@login_required
def dataset_report_pdf_view(request, dataset_id: int):
    dataset = _get_dataset_or_404_for_user(request.user, dataset_id)

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception:
        messages.error(request, "PDF dependency missing (reportlab). Install and retry.")
        return redirect("dataset_overview", dataset_id=dataset.id)

    latest_analysis = AnalysisRun.objects.filter(dataset=dataset).order_by("-created_at").first()
    model_runs = MLModelRun.objects.filter(dataset=dataset).order_by("-created_at")[:10]
    visualizations = Visualization.objects.filter(dataset=dataset).order_by("-created_at")[:10]

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"AEDA Report - {dataset.name} (v{dataset.version})", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Project: {dataset.project.name}", styles["Normal"]))
    story.append(Paragraph(f"Rows: {dataset.row_count} | Columns: {dataset.column_count}", styles["Normal"]))
    story.append(Paragraph(f"Dataset Hash: {dataset.dataset_hash or '-'}", styles["Normal"]))
    story.append(Spacer(1, 12))

    quality = (dataset.summary_json or {}).get("quality_scorecard", {})
    quality_data = [
        ["Quality Metric", "Value"],
        ["Overall Score", quality.get("overall_score", "-")],
        ["Completeness %", quality.get("completeness_pct", "-")],
        ["Duplicate Rows %", quality.get("duplicate_row_pct", "-")],
        ["Numeric Outlier %", quality.get("numeric_outlier_pct", "-")],
    ]
    q_table = Table(quality_data)
    q_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(Paragraph("Data Quality Scorecard", styles["Heading2"]))
    story.append(q_table)
    story.append(Spacer(1, 12))

    if latest_analysis:
        story.append(Paragraph("Latest EDA Summary", styles["Heading2"]))
        story.append(Paragraph(f"Analysis Status: {latest_analysis.status}", styles["Normal"]))
        story.append(Paragraph(f"Applied Transformations: {latest_analysis.applied_transformations}", styles["Normal"]))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Model Leaderboard (This Dataset)", styles["Heading2"]))
    model_data = [["Run", "Model", "Target", "Test Score", "CV Mean", "Created"]]
    for run in model_runs:
        metrics = run.metrics_json or {}
        model_data.append([
            str(run.id),
            run.get_model_type_display(),
            run.target_column,
            str(metrics.get("test_score", "-")),
            str(metrics.get("cv_mean", "-")),
            run.created_at.strftime("%Y-%m-%d %H:%M"),
        ])
    m_table = Table(model_data)
    m_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(m_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Generated Visualizations", styles["Heading2"]))
    viz_data = [["Title", "Type", "Created"]]
    for viz in visualizations:
        viz_data.append([viz.title, viz.chart_type, viz.created_at.strftime("%Y-%m-%d %H:%M")])
    if len(viz_data) == 1:
        viz_data.append(["-", "-", "-"])
    v_table = Table(viz_data)
    v_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(v_table)

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()

    response = HttpResponse(pdf, content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="aeda_report_dataset_{dataset.id}.pdf"'
    return response
