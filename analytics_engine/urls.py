from django.urls import path

from . import views

urlpatterns = [
    path("insights/<int:dataset_id>/", views.auto_insights_view, name="auto_insights"),
    path("visualizations/<int:dataset_id>/", views.visualization_list_view, name="visualizations"),
    path("copilot/<int:dataset_id>/", views.dataset_copilot_view, name="dataset_copilot"),
    path("models/train/", views.model_training_view, name="model_training"),
    path("models/results/<int:run_id>/", views.model_results_view, name="model_results"),
    path("models/predict/", views.model_predict_view, name="model_predict"),
    path("models/leaderboard/", views.model_leaderboard_view, name="model_leaderboard"),
    path("reports/dataset/<int:dataset_id>/", views.dataset_report_pdf_view, name="dataset_report_pdf"),
]
