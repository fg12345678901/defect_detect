// static/js/main.js

document.addEventListener('DOMContentLoaded', function() {
  // File upload validation
  const fileInput = document.getElementById('singleFile');
  if (fileInput) {
    fileInput.addEventListener('change', function() {
      const file = fileInput.files[0];
      if (file) {
        const allowedTypes = ["image/png", "image/jpeg", "image/jpg", "image/bmp"];
        if (!allowedTypes.includes(file.type)) {
          alert("只能上传PNG, JPG或BMP图片!");
          fileInput.value = '';
        } else {
          const preview = document.getElementById('imagePreview');
          if (preview) {
            preview.src = URL.createObjectURL(file);
            preview.style.display = 'block';
            preview.parentElement.style.display = 'block';
            preview.classList.remove('fade-in');
            void preview.offsetWidth;
            preview.classList.add('fade-in');
          }
        }
      }
    });
  }

  // Multiple file upload - show count
  const batchInput = document.querySelector('input[name="batch_folder"]');
  if (batchInput) {
    batchInput.addEventListener('change', function() {
      const files = this.files;
      const fileCount = files.length;
      const countDisplay = document.getElementById('fileCountDisplay');
      if (fileCount > 100) {
        alert("一次最多上传100张图片，请减少文件数量后重试。");
        this.value = '';
        return;
      }
      let totalSize = 0;
      for (let i = 0; i < fileCount; i++) {
        totalSize += files[i].size;
      }
      const maxSizeMB = 450;
      if (totalSize > maxSizeMB * 1024 * 1024) {
        const actualSizeMB = Math.round(totalSize / (1024 * 1024));
        alert(`文件总大小(${actualSizeMB}MB)超过了最大限制(${maxSizeMB}MB)，请减少文件数量或选择较小的文件。`);
        this.value = '';
        return;
      }
      if (countDisplay) {
        const totalSizeMB = Math.round(totalSize / (1024 * 1024) * 10) / 10;
        countDisplay.innerHTML = `已选择 <strong>${fileCount}</strong> 个文件 (总大小: <strong>${totalSizeMB}MB</strong>)`;
        countDisplay.style.display = 'block';
      }
    });
  }

  document.querySelectorAll('.custom-file-input').forEach(input => {
    input.addEventListener('change', function() {
      const fileName = this.files[0]?.name || '选择文件';
      const label = this.nextElementSibling;
      label.textContent = fileName;
    });
  });

  const chartCanvas = document.getElementById('defectPieChart');
  if (chartCanvas) {
    const chartDataElement = document.getElementById('chartData');
    if (chartDataElement) {
      try {
        const chartData = JSON.parse(chartDataElement.textContent);
        createInteractivePieChart(chartCanvas, chartData);
      } catch (error) {
        console.error('Error parsing chart data:', error);
      }
    }
  }

  document.querySelectorAll('.card, .img-container').forEach((element, index) => {
    element.classList.add('fade-in');
    element.style.animationDelay = `${index * 0.1}s`;
  });

  if (typeof $().tooltip === 'function') {
    $('[data-toggle="tooltip"]').tooltip();
  }

  document.querySelectorAll('.image-item').forEach(item => {
    item.addEventListener('click', function() {
      const imgSrc = this.querySelector('img').src;
      const imgName = this.querySelector('.small').textContent;
      const modal = document.createElement('div');
      modal.classList.add('modal', 'fade');
      modal.id = 'imagePreviewModal';
      modal.setAttribute('tabindex', '-1');
      modal.innerHTML = `
        <div class="modal-dialog modal-lg">
          <div class="modal-content">
            <div class="modal-header">
              <h5 class="modal-title">${imgName}</h5>
              <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                <span aria-hidden="true">&times;</span>
              </button>
            </div>
            <div class="modal-body text-center">
              <img src="${imgSrc}" class="img-fluid" alt="${imgName}">
            </div>
          </div>
        </div>`;
      document.body.appendChild(modal);
      $('#imagePreviewModal').modal('show');
      $('#imagePreviewModal').on('hidden.bs.modal', function() {
        document.body.removeChild(modal);
      });
    });
  });
});

function createInteractivePieChart(canvas, data) {
  const colors = [
    '#e74c3c',
    '#2ecc71',
    '#3498db',
    '#f39c12'
  ];

  function getDefectColor(index) {
    const typeMap = {
      '夹杂物': 0,
      '补丁': 1,
      '划痕': 2,
      '其他缺陷': 3
    };
    for (const [key, value] of Object.entries(typeMap)) {
      if (index.includes(key)) {
        return colors[value];
      }
    }
    return '#95a5a6';
  }

  const labels = Object.keys(data);
  const values = Object.values(data);
  const backgroundColors = labels.map(label => getDefectColor(label));

  new Chart(canvas, {
    type: 'pie',
    data: {
      labels: labels,
      datasets: [{
        data: values,
        backgroundColor: backgroundColors,
        borderColor: 'white',
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { padding: 20, font: { size: 14 } }
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              const label = context.label || '';
              const value = context.raw || 0;
              const total = context.dataset.data.reduce((a, b) => a + b, 0);
              const percentage = Math.round((value / total) * 100);
              return `${label}: ${value} 张 (${percentage}%)`;
            }
          }
        },
        title: {
          display: true,
          text: '缺陷类型分布',
          font: { size: 18 },
          padding: { top: 10, bottom: 30 }
        }
      },
      animation: { animateScale: true, animateRotate: true, duration: 1000 }
    }
  });
}

function highlightDefects() {
  const defectElements = document.querySelectorAll('.defect-element');
  defectElements.forEach(element => {
    const defectType = element.dataset.defectType;
    if (defectType) {
      element.classList.add(`defect-highlight-${defectType}`);
    }
  });
}

function toggleFolderView(folderId) {
  const folderContent = document.getElementById(folderId);
  const folderIcon = document.querySelector(`[data-folder="${folderId}"] i`);
  if (folderContent.style.display === 'none') {
    folderContent.style.display = 'block';
    folderIcon.classList.remove('fa-folder');
    folderIcon.classList.add('fa-folder-open');
  } else {
    folderContent.style.display = 'none';
    folderIcon.classList.remove('fa-folder-open');
    folderIcon.classList.add('fa-folder');
  }
}
