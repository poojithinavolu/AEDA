from django.conf import settings
from django.db import models

from projects.models import Dataset


class AnalysisRun(models.Model):
    STATUS_CHOICES = (
        ("queued", "Queued"),
        ("running", "Running"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    )

    owner = models.ForeignKey(settings.AUTH_USER_MODEL, related_name="analysis_runs", on_delete=models.CASCADE)
    dataset = models.ForeignKey(Dataset, related_name="analysis_runs", on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="queued")
    dataset_profile = models.JSONField(default=dict, blank=True)
    llm_feature_plan = models.JSONField(default=dict, blank=True)
    applied_transformations = models.JSONField(default=dict, blank=True)
    processed_columns = models.JSONField(default=list, blank=True)
    transformed_file = models.FileField(upload_to="processed/", blank=True, null=True)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]


class Visualization(models.Model):
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, related_name="visualizations", on_delete=models.CASCADE)
    dataset = models.ForeignKey(Dataset, related_name="visualizations", on_delete=models.CASCADE)
    analysis_run = models.ForeignKey(AnalysisRun, related_name="visualizations", on_delete=models.CASCADE)
    title = models.CharField(max_length=180)
    chart_type = models.CharField(max_length=50)
    spec_json = models.JSONField(default=dict)
    html_snippet = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]


class MLModelRun(models.Model):
    MODEL_CHOICES = (
        ("linear_regression", "Linear Regression"),
        ("logistic_regression", "Logistic Regression"),
        ("random_forest", "Random Forest"),
        ("svm", "SVM"),
    )

    owner = models.ForeignKey(settings.AUTH_USER_MODEL, related_name="ml_runs", on_delete=models.CASCADE)
    dataset = models.ForeignKey(Dataset, related_name="ml_runs", on_delete=models.CASCADE)
    analysis_run = models.ForeignKey(AnalysisRun, related_name="ml_runs", on_delete=models.SET_NULL, null=True, blank=True)
    model_type = models.CharField(max_length=40, choices=MODEL_CHOICES)
    target_column = models.CharField(max_length=180)
    feature_columns = models.JSONField(default=list)
    metrics_json = models.JSONField(default=dict)
    confusion_matrix_json = models.JSONField(default=list, blank=True)
    roc_curve_json = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]


class DatasetCopilotMessage(models.Model):
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_CHOICES = (
        (ROLE_USER, "User"),
        (ROLE_ASSISTANT, "Assistant"),
    )

    owner = models.ForeignKey(settings.AUTH_USER_MODEL, related_name="copilot_messages", on_delete=models.CASCADE)
    dataset = models.ForeignKey(Dataset, related_name="copilot_messages", on_delete=models.CASCADE)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField()
    metadata_json = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
