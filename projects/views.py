from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render

from analytics_engine.models import AnalysisRun
from analytics_engine.services.analysis_service import analyze_dataset_and_create_run

from .forms import DatasetUploadForm, ProjectForm, ProjectMemberInviteForm
from .models import Dataset, ProjectMembership
from .permissions import (
    can_edit_project,
    can_manage_members,
    get_accessible_datasets,
    get_accessible_projects,
    get_project_or_none,
    get_user_role,
)


@login_required
def project_list_view(request):
    form = ProjectForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        project = form.save(commit=False)
        project.owner = request.user
        project.save()
        ProjectMembership.objects.get_or_create(
            project=project,
            user=request.user,
            defaults={"role": ProjectMembership.ROLE_OWNER, "invited_by": request.user},
        )
        messages.success(request, "Project created.")
        return redirect("project_list")

    projects = get_accessible_projects(request.user).prefetch_related("datasets", "memberships")
    return render(request, "projects/project_list.html", {"projects": projects, "form": form})


@login_required
def dataset_upload_view(request):
    initial = {}
    project_id = request.GET.get("project")
    if project_id and str(project_id).isdigit():
        project = get_project_or_none(request.user, int(project_id))
        if project and can_edit_project(project, request.user):
            initial["project"] = project

    form = DatasetUploadForm(request.POST or None, request.FILES or None, user=request.user, initial=initial)
    if request.method == "POST" and form.is_valid():
        dataset = form.save(commit=False)
        if not can_edit_project(dataset.project, request.user):
            messages.error(request, "You do not have permission to upload to this project.")
            return redirect("dataset_upload")

        dataset.owner = request.user
        dataset.version = Dataset.objects.filter(project=dataset.project, name=dataset.name).count() + 1
        dataset.save()
        dataset.dataset_hash = dataset.compute_hash()
        dataset.save(update_fields=["dataset_hash"])

        analyze_dataset_and_create_run(dataset=dataset, user=request.user)
        messages.success(request, "Dataset uploaded and analyzed.")
        return redirect("dataset_overview", dataset_id=dataset.id)

    return render(request, "projects/dataset_upload.html", {"form": form})


@login_required
def dataset_overview_view(request, dataset_id: int):
    dataset = get_object_or_404(get_accessible_datasets(request.user).select_related("project"), id=dataset_id)
    latest_run = AnalysisRun.objects.filter(dataset=dataset).order_by("-created_at").first()
    profile = latest_run.dataset_profile if latest_run else {}

    dtypes_items = list((profile.get("dtypes") or {}).items())
    missing_items = list((profile.get("missing_values") or {}).items())
    summary_stats = profile.get("summary_stats") or {}
    summary_rows = []
    for column, stats in summary_stats.items():
        summary_rows.append(
            {
                "column": column,
                "count": stats.get("count"),
                "mean": stats.get("mean"),
                "std": stats.get("std"),
                "min": stats.get("min"),
                "p25": stats.get("25%"),
                "median": stats.get("50%"),
                "p75": stats.get("75%"),
                "max": stats.get("max"),
            }
        )

    sample_rows = profile.get("sample_rows") or []
    sample_columns = profile.get("columns") or []
    quality_scorecard = (profile.get("quality_scorecard") or dataset.summary_json.get("quality_scorecard") or {})
    user_role = get_user_role(dataset.project, request.user)

    return render(
        request,
        "projects/dataset_overview.html",
        {
            "dataset": dataset,
            "latest_run": latest_run,
            "profile": profile,
            "dtypes_items": dtypes_items,
            "missing_items": missing_items,
            "summary_rows": summary_rows,
            "sample_rows": sample_rows,
            "sample_columns": sample_columns,
            "quality_scorecard": quality_scorecard,
            "user_role": user_role,
        },
    )


@login_required
def dataset_list_view(request):
    datasets = get_accessible_datasets(request.user).select_related("project", "owner")
    return render(request, "projects/dataset_list.html", {"datasets": datasets})


@login_required
def project_detail_view(request, project_id: int):
    project = get_object_or_404(get_accessible_projects(request.user), id=project_id)
    can_manage = can_manage_members(project, request.user)
    can_edit = can_edit_project(project, request.user)

    invite_form = ProjectMemberInviteForm(request.POST or None)
    if request.method == "POST":
        action = request.POST.get("action")
        if action == "invite_member":
            if not can_manage:
                messages.error(request, "Only project owner can manage members.")
                return redirect("project_detail", project_id=project.id)

            if invite_form.is_valid():
                invited_user = invite_form.cleaned_data["username"]
                role = invite_form.cleaned_data["role"]
                if invited_user.id == project.owner_id:
                    messages.info(request, "Owner already has full access.")
                else:
                    ProjectMembership.objects.update_or_create(
                        project=project,
                        user=invited_user,
                        defaults={"role": role, "invited_by": request.user},
                    )
                    messages.success(request, f"{invited_user.username} added as {role}.")
                return redirect("project_detail", project_id=project.id)

        if action == "remove_member":
            if not can_manage:
                messages.error(request, "Only project owner can manage members.")
                return redirect("project_detail", project_id=project.id)

            membership_id = request.POST.get("membership_id")
            if membership_id and str(membership_id).isdigit():
                membership = project.memberships.filter(id=int(membership_id)).first()
                if membership and membership.user_id != project.owner_id:
                    membership.delete()
                    messages.success(request, "Member removed from project.")
            return redirect("project_detail", project_id=project.id)

        if action == "delete_dataset":
            if not can_edit:
                messages.error(request, "You do not have permission to delete datasets in this project.")
                return redirect("project_detail", project_id=project.id)

            dataset_id = request.POST.get("dataset_id")
            if dataset_id and str(dataset_id).isdigit():
                dataset = project.datasets.filter(id=int(dataset_id)).first()
                if dataset:
                    dataset_name = dataset.name
                    dataset.delete()
                    messages.success(request, f"Dataset '{dataset_name}' deleted.")
            return redirect("project_detail", project_id=project.id)

    datasets = Dataset.objects.filter(project=project).order_by("-created_at")
    dataset_ids = list(datasets.values_list("id", flat=True))
    runs = AnalysisRun.objects.filter(dataset_id__in=dataset_ids).order_by("dataset_id", "-created_at")

    latest_by_dataset = {}
    count_by_dataset = {}
    for run in runs:
        count_by_dataset[run.dataset_id] = count_by_dataset.get(run.dataset_id, 0) + 1
        if run.dataset_id not in latest_by_dataset:
            latest_by_dataset[run.dataset_id] = run

    dataset_rows = []
    for dataset in datasets:
        dataset_rows.append(
            {
                "dataset": dataset,
                "latest_run": latest_by_dataset.get(dataset.id),
                "analysis_runs_count": count_by_dataset.get(dataset.id, 0),
            }
        )

    members = project.memberships.select_related("user", "invited_by")
    user_role = get_user_role(project, request.user)

    context = {
        "project": project,
        "dataset_rows": dataset_rows,
        "dataset_count": datasets.count(),
        "analysis_count": runs.count(),
        "members": members,
        "invite_form": invite_form,
        "can_manage_members": can_manage,
        "can_edit": can_edit,
        "user_role": user_role,
    }
    return render(request, "projects/project_detail.html", context)
