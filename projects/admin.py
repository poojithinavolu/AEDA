from django.contrib import admin

from .models import Dataset, Project, ProjectMembership


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "owner", "created_at")
    search_fields = ("name", "owner__username")


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "project", "owner", "file_type", "created_at")
    list_filter = ("file_type",)
    search_fields = ("name", "owner__username", "project__name")


@admin.register(ProjectMembership)
class ProjectMembershipAdmin(admin.ModelAdmin):
    list_display = ("project", "user", "role", "invited_by", "created_at")
    list_filter = ("role",)
    search_fields = ("project__name", "user__username")
