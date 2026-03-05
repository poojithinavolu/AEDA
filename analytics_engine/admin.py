from django.contrib import admin

from .models import AnalysisRun, DatasetCopilotMessage, MLModelRun, Visualization


@admin.register(AnalysisRun)
class AnalysisRunAdmin(admin.ModelAdmin):
    list_display = ("id", "dataset", "owner", "status", "created_at")
    list_filter = ("status",)


@admin.register(Visualization)
class VisualizationAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "chart_type", "dataset", "owner", "created_at")


@admin.register(MLModelRun)
class MLModelRunAdmin(admin.ModelAdmin):
    list_display = ("id", "model_type", "target_column", "dataset", "owner", "created_at")


@admin.register(DatasetCopilotMessage)
class DatasetCopilotMessageAdmin(admin.ModelAdmin):
    list_display = ("dataset", "owner", "role", "created_at")
    search_fields = ("dataset__name", "owner__username", "content")
