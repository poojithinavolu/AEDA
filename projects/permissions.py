from django.db.models import Q

from .models import Dataset, Project, ProjectMembership


EDIT_ROLES = {ProjectMembership.ROLE_OWNER, ProjectMembership.ROLE_ANALYST}
VIEW_ROLES = {ProjectMembership.ROLE_OWNER, ProjectMembership.ROLE_ANALYST, ProjectMembership.ROLE_VIEWER}


def accessible_projects_q(user):
    return Q(owner=user) | Q(memberships__user=user)


def editable_projects_q(user):
    return Q(owner=user) | Q(memberships__user=user, memberships__role__in=[ProjectMembership.ROLE_OWNER, ProjectMembership.ROLE_ANALYST])


def get_accessible_projects(user):
    return Project.objects.filter(accessible_projects_q(user)).distinct()


def get_editable_projects(user):
    return Project.objects.filter(editable_projects_q(user)).distinct()


def get_accessible_datasets(user):
    return Dataset.objects.filter(project__in=get_accessible_projects(user)).distinct()


def get_dataset_or_none(user, dataset_id: int):
    return get_accessible_datasets(user).filter(id=dataset_id).select_related("project").first()


def get_project_or_none(user, project_id: int):
    return get_accessible_projects(user).filter(id=project_id).first()


def get_user_role(project: Project, user):
    if project.owner_id == user.id:
        return ProjectMembership.ROLE_OWNER
    membership = project.memberships.filter(user=user).first()
    return membership.role if membership else None


def can_edit_project(project: Project, user) -> bool:
    role = get_user_role(project, user)
    return role in EDIT_ROLES


def can_view_project(project: Project, user) -> bool:
    role = get_user_role(project, user)
    return role in VIEW_ROLES


def can_manage_members(project: Project, user) -> bool:
    return project.owner_id == user.id
