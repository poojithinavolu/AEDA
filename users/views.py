from django.contrib import messages
from django.contrib.auth import login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from analytics_engine.models import AnalysisRun
from projects.permissions import get_accessible_datasets, get_accessible_projects

from .forms import RegisterForm, StyledPasswordChangeForm, UserProfileForm


def register_view(request):
    if request.user.is_authenticated:
        return redirect("dashboard")

    form = RegisterForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save()
        login(request, user)
        return redirect("dashboard")

    return render(request, "users/register.html", {"form": form})


@login_required
def dashboard_view(request):
    projects_qs = get_accessible_projects(request.user)
    datasets_qs = get_accessible_datasets(request.user)
    projects_count = projects_qs.count()
    datasets_count = datasets_qs.count()
    analysis_count = AnalysisRun.objects.filter(dataset__in=datasets_qs).count()

    return render(
        request,
        "users/dashboard.html",
        {
            "projects_count": projects_count,
            "datasets_count": datasets_count,
            "analysis_count": analysis_count,
        },
    )


@login_required
def logout_view(request):
    # Accept both POST and GET to avoid 405 when clients hit /logout/ directly.
    logout(request)
    return redirect("login")


@login_required
def profile_view(request):
    profile_form = UserProfileForm(request.POST or None, instance=request.user, prefix="profile")
    password_form = StyledPasswordChangeForm(user=request.user, data=request.POST or None, prefix="password")

    if request.method == "POST":
        action = request.POST.get("action")
        if action == "save_profile":
            if profile_form.is_valid():
                profile_form.save()
                messages.success(request, "Profile updated successfully.")
                return redirect("profile")
        elif action == "change_password":
            if password_form.is_valid():
                user = password_form.save()
                update_session_auth_hash(request, user)
                messages.success(request, "Password changed successfully.")
                return redirect("profile")

    projects_count = get_accessible_projects(request.user).count()
    datasets_count = get_accessible_datasets(request.user).count()
    analysis_count = AnalysisRun.objects.filter(dataset__in=get_accessible_datasets(request.user)).count()

    return render(
        request,
        "users/profile.html",
        {
            "profile_form": profile_form,
            "password_form": password_form,
            "projects_count": projects_count,
            "datasets_count": datasets_count,
            "analysis_count": analysis_count,
        },
    )
