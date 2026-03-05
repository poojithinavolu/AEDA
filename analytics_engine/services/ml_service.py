from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from analytics_engine.models import MLModelRun
from analytics_engine.utils.data_io import load_dataset_frame


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    return value


def _default_training_config() -> dict:
    return {
        "test_size": 0.2,
        "random_state": 42,
        "cv_folds": 5,
        "max_iter": 1000,
        "n_estimators": 200,
        "svm_c": 1.0,
        "svm_kernel": "rbf",
        "auto_tune": True,
    }


def _build_preprocessor(model_type: str, numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    # Tree-based models generally do not benefit from scaling.
    if model_type == "random_forest":
        numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    else:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def _build_model(model_type: str, cfg: dict):
    if model_type == "linear_regression":
        return LinearRegression()
    if model_type == "logistic_regression":
        return LogisticRegression(
            max_iter=int(cfg["max_iter"]),
            random_state=int(cfg["random_state"]),
            class_weight="balanced",
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(cfg["n_estimators"]),
            random_state=int(cfg["random_state"]),
            class_weight="balanced",
            n_jobs=-1,
        )
    if model_type == "svm":
        svm_max_iter = max(int(cfg["max_iter"]), 3000)
        return SVC(
            C=float(cfg["svm_c"]),
            kernel=str(cfg["svm_kernel"]),
            max_iter=svm_max_iter,
            class_weight="balanced",
            probability=True,
            random_state=int(cfg["random_state"]),
        )
    raise ValueError("Unsupported model type")


def _fit_with_optional_tuning(clf: Pipeline, X_train, y_train, model_type: str, cfg: dict, is_regression: bool):
    auto_tune = bool(cfg.get("auto_tune", True))
    if not auto_tune:
        clf.fit(X_train, y_train)
        return clf, None

    if model_type == "random_forest":
        grid = {
            "model__n_estimators": [int(cfg["n_estimators"]), max(50, int(cfg["n_estimators"]) // 2), int(cfg["n_estimators"]) * 2],
            "model__max_depth": [None, 6, 12, 20],
            "model__min_samples_leaf": [1, 2, 4],
        }
    elif model_type == "logistic_regression":
        grid = {
            "model__C": [0.1, 0.5, 1.0, 2.0, 5.0],
            "model__solver": ["lbfgs"],
        }
    elif model_type == "svm":
        grid = {
            "model__C": [0.1, 1.0, 3.0, 10.0],
            "model__kernel": [str(cfg["svm_kernel"]), "rbf", "linear"],
        }
    else:
        clf.fit(X_train, y_train)
        return clf, None

    if is_regression:
        scoring = "r2"
    else:
        scoring = "roc_auc" if pd.Series(y_train).nunique(dropna=True) == 2 else "accuracy"

    cv_folds = int(cfg["cv_folds"])
    if is_regression:
        max_cv = min(cv_folds, max(2, len(y_train)))
    else:
        min_class_count = int(pd.Series(y_train).value_counts().min())
        max_cv = min(cv_folds, max(2, min_class_count))

    search = GridSearchCV(clf, param_grid=grid, scoring=scoring, cv=max_cv, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, {"best_params": search.best_params_, "best_score": float(search.best_score_), "scoring": scoring}


def _artifact_path_for_run(run_id: int) -> Path:
    folder = Path(settings.MEDIA_ROOT) / "model_artifacts"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"ml_run_{run_id}.joblib"


def _fit_pipeline_from_existing_run(model_run: MLModelRun) -> Pipeline:
    dataset = model_run.dataset
    model_type = model_run.model_type
    target_column = model_run.target_column

    df = load_dataset_frame(dataset.file.path, dataset.file_type)
    if target_column not in df.columns:
        raise ValueError("Target column not found in dataset for this model run.")

    df = df.dropna(subset=[target_column]).copy()
    y = df[target_column]
    X = df.drop(columns=[target_column]).copy()

    cfg = _default_training_config()
    cfg.update((model_run.metrics_json or {}).get("training_config", {}))

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    preprocessor = _build_preprocessor(model_type, numeric_features, categorical_features)
    model = _build_model(model_type, cfg)
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    is_regression = model_type == "linear_regression"
    if is_regression and not pd.api.types.is_numeric_dtype(y):
        raise ValueError("Linear Regression requires a numeric target column.")
    if not is_regression and pd.Series(y).nunique(dropna=True) < 2:
        raise ValueError("Classification target must have at least 2 classes.")

    stratify = None if is_regression else y
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=float(cfg["test_size"]),
        random_state=int(cfg["random_state"]),
        stratify=stratify,
    )
    clf, _ = _fit_with_optional_tuning(clf, X_train, y_train, model_type, cfg, is_regression)
    return clf


def _ensure_model_artifact(model_run: MLModelRun) -> Path:
    metrics = model_run.metrics_json or {}
    artifact_path = metrics.get("artifact_path")
    if artifact_path:
        path = Path(artifact_path)
        if path.exists():
            return path

    # Legacy runs (or deleted artifacts): rebuild model pipeline from stored run metadata.
    clf = _fit_pipeline_from_existing_run(model_run)
    rebuilt_path = _artifact_path_for_run(model_run.id)
    dump(clf, rebuilt_path)

    metrics["artifact_path"] = str(rebuilt_path)
    notes = metrics.get("notes", [])
    if not isinstance(notes, list):
        notes = []
    notes.append("Model artifact auto-rebuilt for legacy run compatibility.")
    metrics["notes"] = notes[-8:]
    model_run.metrics_json = _json_safe(metrics)
    model_run.save(update_fields=["metrics_json"])
    return rebuilt_path


def _feature_names_from_preprocessor(preprocessor, transformed_width: int) -> list[str]:
    try:
        names = preprocessor.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        return [f"feature_{idx}" for idx in range(transformed_width)]


def _compute_explainability(clf: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    sample_size = min(len(X_test), 120)
    if sample_size <= 2:
        return {"method": "none", "notes": ["Not enough rows for explainability analysis."]}

    x_sample = X_test.head(sample_size)
    y_sample = y_test.head(sample_size)
    preprocessor = clf.named_steps["preprocessor"]
    model = clf.named_steps["model"]

    transformed = preprocessor.transform(x_sample)
    feature_names = _feature_names_from_preprocessor(preprocessor, transformed.shape[1])

    try:
        import shap

        explainer = shap.Explainer(model, transformed, feature_names=feature_names)
        shap_values = explainer(transformed)
        values = shap_values.values
        if values.ndim == 3:
            values = values[:, :, min(1, values.shape[2] - 1)]

        mean_abs = np.abs(values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:12]
        global_importance = [
            {"feature": feature_names[int(i)], "importance": float(mean_abs[int(i)])}
            for i in top_idx
        ]

        row_vals = values[0]
        local_idx = np.argsort(np.abs(row_vals))[::-1][:8]
        row_explanation = [
            {
                "feature": feature_names[int(i)],
                "contribution": float(row_vals[int(i)]),
                "direction": "up" if float(row_vals[int(i)]) >= 0 else "down",
            }
            for i in local_idx
        ]
        return {
            "method": "shap",
            "global_importance": global_importance,
            "local_example": row_explanation,
            "sample_size": sample_size,
        }
    except Exception:
        # Fallback keeps feature-importance available if SHAP is missing/incompatible.
        perm = permutation_importance(clf, x_sample, y_sample, n_repeats=8, random_state=42, n_jobs=-1)
        mean_imp = perm.importances_mean
        top_idx = np.argsort(mean_imp)[::-1][:12]
        global_importance = [
            {"feature": str(x_sample.columns[int(i)]), "importance": float(mean_imp[int(i)])}
            for i in top_idx
        ]
        return {
            "method": "permutation_fallback",
            "global_importance": global_importance,
            "local_example": [],
            "sample_size": sample_size,
            "notes": ["SHAP not available; used permutation importance fallback."],
        }


def train_model(dataset, owner, model_type: str, target_column: str, training_config: dict | None = None) -> MLModelRun:
    df = load_dataset_frame(dataset.file.path, dataset.file_type)
    if target_column not in df.columns:
        raise ValueError("Target column not found")

    # ML training uses original uploaded dataset only; LLM feature transforms are not forced.
    df = df.dropna(subset=[target_column]).copy()
    y = df[target_column]
    X = df.drop(columns=[target_column]).copy()

    cfg = _default_training_config()
    if training_config:
        cfg.update(training_config)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocessor = _build_preprocessor(model_type, numeric_features, categorical_features)
    model = _build_model(model_type, cfg)
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    is_regression = model_type == "linear_regression"

    if is_regression and not pd.api.types.is_numeric_dtype(y):
        raise ValueError("Linear Regression requires a numeric target column.")
    if not is_regression and pd.Series(y).nunique(dropna=True) < 2:
        raise ValueError("Classification target must have at least 2 classes.")

    stratify = None if is_regression else y
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(cfg["test_size"]),
        random_state=int(cfg["random_state"]),
        stratify=stratify,
    )

    clf, tune_info = _fit_with_optional_tuning(clf, X_train, y_train, model_type, cfg, is_regression)

    y_pred = clf.predict(X_test)

    metrics = {
        "task_type": "regression" if is_regression else "classification",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "training_config": cfg,
        "train_score": float(clf.score(X_train, y_train)),
        "test_score": float(clf.score(X_test, y_test)),
        "data_source": "original_uploaded_dataset",
        "feature_scaling_for_model": "disabled" if model_type == "random_forest" else "enabled",
        "dataset_version": int(dataset.version),
        "dataset_hash": dataset.dataset_hash or "",
    }
    if tune_info:
        metrics["tuning"] = tune_info

    confusion = None
    roc_payload = {}
    notes = []

    if is_regression:
        metrics["r2"] = float(r2_score(y_test, y_pred))
        metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
        metrics["mse"] = float(mean_squared_error(y_test, y_pred))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics["confusion_available"] = False
        metrics["roc_available"] = False
        notes.append("Confusion matrix and ROC/AUC are classification-only metrics.")
    else:
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["f1_score"] = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        metrics["precision"] = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        metrics["recall"] = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))

        unique_classes = pd.Series(y_test).dropna().unique()
        if len(unique_classes) == 2 and hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_proba, pos_label=unique_classes[1])
            metrics["auc"] = float(roc_auc_score(y_test, y_proba))
            roc_payload = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            }
            metrics["roc_available"] = True
        else:
            metrics["roc_available"] = False
            notes.append("ROC/AUC shown only for binary classification with probability scores.")

        confusion = confusion_matrix(y_test, y_pred).tolist()
        metrics["confusion_available"] = True

    cv_folds = int(cfg["cv_folds"])
    if cv_folds >= 2:
        try:
            scoring = "r2" if is_regression else "accuracy"
            if is_regression:
                max_cv = min(cv_folds, max(2, len(y_train)))
            else:
                min_class_count = int(pd.Series(y_train).value_counts().min())
                max_cv = min(cv_folds, max(2, min_class_count))
            if max_cv >= 2:
                cv_scores = cross_val_score(clf, X_train, y_train, cv=max_cv, scoring=scoring)
                metrics["cv_metric"] = scoring
                metrics["cv_folds_used"] = int(max_cv)
                metrics["cv_scores"] = [float(v) for v in cv_scores.tolist()]
                metrics["cv_mean"] = float(np.mean(cv_scores))
                metrics["cv_std"] = float(np.std(cv_scores))
        except Exception:
            notes.append("Cross-validation was skipped due to data constraints.")

    preview = []
    for idx in range(min(8, len(y_test))):
        preview.append({"actual": str(y_test.iloc[idx]), "predicted": str(y_pred[idx])})
    metrics["prediction_preview"] = preview
    metrics["explainability"] = _compute_explainability(clf, X_test, y_test)
    if notes:
        metrics["notes"] = notes

    run = MLModelRun.objects.create(
        owner=owner,
        dataset=dataset,
        analysis_run=dataset.analysis_runs.filter(owner=owner).order_by("-created_at").first(),
        model_type=model_type,
        target_column=target_column,
        feature_columns=_json_safe(list(X.columns)),
        metrics_json=_json_safe(metrics),
        confusion_matrix_json=_json_safe(confusion or []),
        roc_curve_json=_json_safe(roc_payload),
    )

    artifact_path = _artifact_path_for_run(run.id)
    dump(clf, artifact_path)
    run_metrics = run.metrics_json or {}
    run_metrics["artifact_path"] = str(artifact_path)
    run.metrics_json = _json_safe(run_metrics)
    run.save(update_fields=["metrics_json"])
    return run


def predict_with_trained_model(model_run: MLModelRun, input_payload: dict):
    path = _ensure_model_artifact(model_run)

    clf = load(path)
    feature_cols = model_run.feature_columns or []

    row = {}
    for col in feature_cols:
        row[col] = input_payload.get(col)
    frame = pd.DataFrame([row])

    pred = clf.predict(frame)
    result = {"prediction": pred.tolist()}

    if hasattr(clf, "predict_proba"):
        try:
            result["probabilities"] = clf.predict_proba(frame).tolist()
        except Exception:
            pass

    return _json_safe(result)
