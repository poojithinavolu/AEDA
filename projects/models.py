import os
import uuid
import hashlib

from django.conf import settings
from django.db import models


def dataset_upload_path(instance, filename: str) -> str:
    ext = os.path.splitext(filename)[1]
    return f"datasets/user_{instance.owner_id}/project_{instance.project_id}/{uuid.uuid4().hex}{ext}"


class Project(models.Model):
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, related_name="projects", on_delete=models.CASCADE)
    name = models.CharField(max_length=180)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("owner", "name")
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.name} ({self.owner})"


class ProjectMembership(models.Model):
    ROLE_OWNER = "owner"
    ROLE_ANALYST = "analyst"
    ROLE_VIEWER = "viewer"
    ROLE_CHOICES = (
        (ROLE_OWNER, "Owner"),
        (ROLE_ANALYST, "Analyst"),
        (ROLE_VIEWER, "Viewer"),
    )

    project = models.ForeignKey(Project, related_name="memberships", on_delete=models.CASCADE)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, related_name="project_memberships", on_delete=models.CASCADE)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default=ROLE_VIEWER)
    invited_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        related_name="project_invitations_sent",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("project", "user")
        ordering = ["project_id", "user_id"]

    def __str__(self) -> str:
        return f"{self.project.name} - {self.user.username} ({self.role})"


class Dataset(models.Model):
    FILETYPE_CHOICES = (
        ("csv", "CSV"),
        ("xlsx", "Excel"),
        ("xls", "Excel Legacy"),
    )

    owner = models.ForeignKey(settings.AUTH_USER_MODEL, related_name="datasets", on_delete=models.CASCADE)
    project = models.ForeignKey(Project, related_name="datasets", on_delete=models.CASCADE)
    name = models.CharField(max_length=180)
    file = models.FileField(upload_to=dataset_upload_path)
    file_type = models.CharField(max_length=10, choices=FILETYPE_CHOICES)
    row_count = models.PositiveIntegerField(default=0)
    column_count = models.PositiveIntegerField(default=0)
    schema_json = models.JSONField(default=dict, blank=True)
    summary_json = models.JSONField(default=dict, blank=True)
    version = models.PositiveIntegerField(default=1)
    dataset_hash = models.CharField(max_length=64, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.name} v{self.version} - {self.project.name}"

    def compute_hash(self) -> str:
        if not self.file:
            return ""
        digest = hashlib.sha256()
        with self.file.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
