// static/js/main.js

document.addEventListener('DOMContentLoaded', function() {
  // 文件上传验证
  const fileInput = document.getElementById('singleFile');
  if (fileInput) {
    fileInput.addEventListener('change', function() {
      const file = fileInput.files[0];
      if (file) {
        const allowedTypes = ["image/png", "image/jpeg", "image/jpg", "image/bmp"];
        if (!allowedTypes.includes(file.type)) {
          alert("只能上传PNG, JPG或BMP图片!");
          fileInput.value = '';
        }
      }
    });
  }

  // 自定义文件输入
  document.querySelectorAll('.custom-file-input').forEach(input => {
    input.addEventListener('change', function() {
      let fileName = '选择文件';
      if (this.files.length === 1) {
        fileName = this.files[0].name;
      } else if (this.files.length > 1) {
        fileName = `已选择 ${this.files.length} 个文件`;
      }
      const label = this.nextElementSibling;
      if (label) {
        label.textContent = fileName;
      }
    });
  });

  // 添加动画效果
  document.querySelectorAll('.card, .img-container').forEach((element, index) => {
    element.classList.add('fade-in');
    element.style.animationDelay = `${index * 0.1}s`;
  });

  // 初始化工具提示
  if (typeof $().tooltip === 'function') {
    $('[data-toggle="tooltip"]').tooltip();
  }

  // 图片点击放大预览
  document.querySelectorAll('.image-item').forEach(item => {
    item.addEventListener('click', function() {
      const imgSrc = this.querySelector('img').src;
      const imgName = this.querySelector('.small')?.textContent || 'Image';
      createImageModal(imgSrc, imgName);
    });
  });

  // 手风琴图标切换
  document.querySelectorAll('.accordion .btn-link').forEach(btn => {
    btn.addEventListener('click', function() {
      const icon = this.querySelector('i.fa-chevron-down');
      if (icon) {
        setTimeout(() => {
          const isExpanded = this.getAttribute('aria-expanded') === 'true';
          icon.style.transform = isExpanded ? 'rotate(180deg)' : 'rotate(0)';
        }, 100);
      }
    });
  });
});

// 创建图片预览模态框
function createImageModal(imgSrc, imgName) {
  // 检查是否已存在模态框
  const existingModal = document.getElementById('imagePreviewModal');
  if (existingModal) {
    existingModal.remove();
  }

  const modal = document.createElement('div');
  modal.classList.add('modal', 'fade');
  modal.id = 'imagePreviewModal';
  modal.setAttribute('tabindex', '-1');
  modal.innerHTML = `
    <div class="modal-dialog modal-lg modal-dialog-centered">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title">${imgName}</h5>
          <button type="button" class="close" data-dismiss="modal" aria-label="Close">
            <span aria-hidden="true">&times;</span>
          </button>
        </div>
        <div class="modal-body text-center p-0">
          <img src="${imgSrc}" class="img-fluid" alt="${imgName}">
        </div>
      </div>
    </div>`;

  document.body.appendChild(modal);

  // 使用jQuery显示模态框
  if (typeof $ !== 'undefined') {
    $('#imagePreviewModal').modal('show');
    $('#imagePreviewModal').on('hidden.bs.modal', function() {
      document.body.removeChild(modal);
    });
  }
}

// 切换文件夹视图
function toggleFolderView(folderId) {
  const folderContent = document.getElementById(folderId);
  const folderHeader = folderContent?.previousElementSibling;
  const icon = folderHeader?.querySelector('i.fa-chevron-down');

  if (folderContent && icon) {
    if (folderContent.style.display === 'none') {
      folderContent.style.display = 'block';
      icon.style.transform = 'rotate(180deg)';
    } else {
      folderContent.style.display = 'none';
      icon.style.transform = 'rotate(0)';
    }
  }
}

// 平滑滚动
function smoothScroll(target) {
  const element = document.querySelector(target);
  if (element) {
    element.scrollIntoView({
      behavior: 'smooth',
      block: 'start'
    });
  }
}

// 导航栏平滑滚动
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    e.preventDefault();
    const target = this.getAttribute('href');
    if (target && target !== '#') {
      smoothScroll(target);
    }
  });
});

// 动态加载效果
function addLoadingEffect(element) {
  element.classList.add('loading');
  element.innerHTML = '<div class="spinner-border" role="status"><span class="sr-only">加载中...</span></div>';
}

function removeLoadingEffect(element, content) {
  element.classList.remove('loading');
  element.innerHTML = content;
}

// 文件大小格式化
function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// 时间格式化
function formatTime(seconds) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  if (hours > 0) {
    return `${hours}小时 ${minutes}分钟 ${secs}秒`;
  } else if (minutes > 0) {
    return `${minutes}分钟 ${secs}秒`;
  } else {
    return `${secs}秒`;
  }
}

// 添加通知功能
function showNotification(message, type = 'info') {
  const notification = document.createElement('div');
  notification.className = `alert alert-${type} alert-dismissible fade show notification`;
  notification.innerHTML = `
    ${message}
    <button type="button" class="close" data-dismiss="alert" aria-label="Close">
      <span aria-hidden="true">&times;</span>
    </button>
  `;

  // 添加样式
  notification.style.position = 'fixed';
  notification.style.top = '20px';
  notification.style.right = '20px';
  notification.style.zIndex = '9999';
  notification.style.minWidth = '300px';

  document.body.appendChild(notification);

  // 自动关闭
  setTimeout(() => {
    notification.classList.remove('show');
    setTimeout(() => notification.remove(), 150);
  }, 5000);
}

// 表单验证增强
function validateForm(form) {
  const inputs = form.querySelectorAll('[required]');
  let isValid = true;

  inputs.forEach(input => {
    if (!input.value.trim()) {
      input.classList.add('is-invalid');
      isValid = false;
    } else {
      input.classList.remove('is-invalid');
    }
  });

  return isValid;
}

// 添加表单提交确认
document.querySelectorAll('form').forEach(form => {
  form.addEventListener('submit', function(e) {
    if (!validateForm(this)) {
      e.preventDefault();
      showNotification('请填写所有必填字段', 'warning');
    }
  });
});
