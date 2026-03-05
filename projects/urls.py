from django.urls import path

from . import views

urlpatterns = [
    path("", views.project_list_view, name="project_list"),
    path("datasets/", views.dataset_list_view, name="dataset_list"),
    path("datasets/upload/", views.dataset_upload_view, name="dataset_upload"),
    path("datasets/<int:dataset_id>/", views.dataset_overview_view, name="dataset_overview"),
    path("<int:project_id>/", views.project_detail_view, name="project_detail"),
]
