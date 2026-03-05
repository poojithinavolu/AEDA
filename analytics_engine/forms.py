from django import forms

from analytics_engine.models import MLModelRun
from projects.models import Dataset
from projects.permissions import get_accessible_datasets, get_editable_projects


class ModelTrainingForm(forms.Form):
    MODEL_CHOICES = (
        ("linear_regression", "Linear Regression"),
        ("logistic_regression", "Logistic Regression"),
        ("random_forest", "Random Forest"),
        ("svm", "SVM"),
    )

    dataset = forms.ModelChoiceField(queryset=Dataset.objects.none())
    model_type = forms.ChoiceField(choices=MODEL_CHOICES)
    target_column = forms.CharField(max_length=180)
    test_size = forms.FloatField(min_value=0.1, max_value=0.5, initial=0.2)
    random_state = forms.IntegerField(min_value=0, initial=42)
    cv_folds = forms.IntegerField(min_value=2, max_value=10, initial=5)
    max_iter = forms.IntegerField(min_value=50, max_value=20000, initial=1000)
    n_estimators = forms.IntegerField(min_value=10, max_value=2000, initial=200)
    svm_c = forms.FloatField(min_value=0.01, max_value=1000.0, initial=1.0)
    svm_kernel = forms.ChoiceField(
        choices=(
            ("rbf", "RBF"),
            ("linear", "Linear"),
            ("poly", "Polynomial"),
            ("sigmoid", "Sigmoid"),
        ),
        initial="rbf",
    )
    auto_tune = forms.BooleanField(required=False, initial=True)

    def __init__(self, *args, user=None, **kwargs):
        super().__init__(*args, **kwargs)
        if user is not None:
            self.fields["dataset"].queryset = get_accessible_datasets(user).filter(project__in=get_editable_projects(user)).distinct()
        for _, field in self.fields.items():
            existing = field.widget.attrs.get("class", "")
            if isinstance(field, forms.BooleanField):
                field.widget.attrs["class"] = f"{existing} form-check-input".strip()
            else:
                field.widget.attrs["class"] = f"{existing} form-control".strip()


class ModelSelectionForm(forms.Form):
    dataset = forms.ModelChoiceField(queryset=Dataset.objects.none())
    model_run = forms.ModelChoiceField(queryset=MLModelRun.objects.none())

    def __init__(self, *args, user=None, dataset_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        if user is not None:
            datasets_qs = get_accessible_datasets(user)
            self.fields["dataset"].queryset = datasets_qs
            runs = MLModelRun.objects.filter(dataset__in=datasets_qs).select_related("dataset").order_by("-created_at")
            if dataset_id:
                runs = runs.filter(dataset_id=dataset_id)
            else:
                runs = runs.none()
            self.fields["model_run"].queryset = runs
            self.fields["model_run"].label_from_instance = (
                lambda obj: f"Run #{obj.id} | {obj.dataset.name} | {obj.get_model_type_display()} | target={obj.target_column} | {obj.created_at.strftime('%Y-%m-%d %H:%M')}"
            )
            self.fields["model_run"].empty_label = "Select a trained model run"
            self.fields["dataset"].empty_label = "Select dataset first"
        for _, field in self.fields.items():
            existing = field.widget.attrs.get("class", "")
            field.widget.attrs["class"] = f"{existing} form-control".strip()
