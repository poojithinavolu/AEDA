document.addEventListener("DOMContentLoaded", () => {
  const alerts = document.querySelectorAll(".alert");
  setTimeout(() => {
    alerts.forEach((el) => el.remove());
  }, 4000);

  const datasetSelect = document.getElementById("id_dataset");
  if (datasetSelect && window.location.pathname.includes("/analytics/models/predict/")) {
    datasetSelect.addEventListener("change", () => {
      const datasetId = datasetSelect.value;
      if (!datasetId) {
        window.location.href = window.location.pathname;
        return;
      }
      window.location.href = `${window.location.pathname}?dataset=${encodeURIComponent(datasetId)}`;
    });
  }
});
