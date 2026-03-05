from django import forms
from django.contrib.auth import get_user_model

from .models import Dataset, Project, ProjectMembership
from .permissions import get_editable_projects

User = get_user_model()


class ProjectForm(forms.ModelForm):
    class Meta:
        model = Project
        fields = ("name", "description")


class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ("project", "name", "file")

    def __init__(self, *args, user=None, **kwargs):
        super().__init__(*args, **kwargs)
        if user is not None:
            self.fields["project"].queryset = get_editable_projects(user)

    def clean_file(self):
        file_obj = self.cleaned_data["file"]
        filename = file_obj.name.lower()
        if filename.endswith(".csv"):
            self.cleaned_data["file_type"] = "csv"
        elif filename.endswith(".xlsx"):
            self.cleaned_data["file_type"] = "xlsx"
        elif filename.endswith(".xls"):
            self.cleaned_data["file_type"] = "xls"
        else:
            raise forms.ValidationError("Only CSV or Excel files are supported.")
        return file_obj

    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.file_type = self.cleaned_data["file_type"]
        if commit:
            instance.save()
        return instance


class ProjectMemberInviteForm(forms.Form):
    username = forms.CharField(max_length=150)
    role = forms.ChoiceField(
        choices=(
            (ProjectMembership.ROLE_ANALYST, "Analyst"),
            (ProjectMembership.ROLE_VIEWER, "Viewer"),
        )
    )

    def clean_username(self):
        username = self.cleaned_data["username"].strip()
        try:
            return User.objects.get(username=username)
        except User.DoesNotExist as exc:
            raise forms.ValidationError("User with this username does not exist.") from exc

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for _, field in self.fields.items():
            existing = field.widget.attrs.get("class", "")
            field.widget.attrs["class"] = f"{existing} form-control".strip()
