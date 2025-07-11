
function toggleFolderView(id) {
  const body = document.getElementById(id);
  if (body.style.display === 'none') {
    body.style.display = 'block';
  } else {
    body.style.display = 'none';
  }
}

window.addEventListener('DOMContentLoaded', function() {
  const chartDataEl = document.getElementById('chartData');
  if (chartDataEl) {
    const dataObj = JSON.parse(chartDataEl.textContent);
    const labels = Object.keys(dataObj);
    const values = Object.values(dataObj);
    new Chart(document.getElementById('defectPieChart'), {
      type: 'pie',
      data: {
        labels: labels,
        datasets: [{
          data: values,
          backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#8bc34a', '#9c27b0', '#607d8b']
        }]
      },
      options: {
        plugins: {
          legend: { position: 'right' },
          datalabels: { color: '#fff' }
        }
      }
    });
  }
});
