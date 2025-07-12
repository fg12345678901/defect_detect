from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import os
import json
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
import threading
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

from models.classifier import create_classifier
from models.segmenter import get_unet_model
from utils.viz import visualize_prediction_3d, TASK_COLOR_MAPS

app = Flask(__name__)

# ---- Configuration ----
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / 'static' / 'uploads'
PRED_DIR = BASE_DIR / 'static' / 'pred_vis'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

# 加载任务信息和颜色映射
TASK_INFO_PATH = BASE_DIR / 'tasks_info.json'
with open(TASK_INFO_PATH, 'r', encoding='utf-8') as f:
    TASK_INFO = json.load(f)

SOLAR_COLOR_MAP_PATH = BASE_DIR.parent / 'dataset' / 'solar-panel' / 'color_map.json'
if SOLAR_COLOR_MAP_PATH.exists():
    with open(SOLAR_COLOR_MAP_PATH, 'r', encoding='utf-8') as f:
        SOLAR_COLOR_MAP = {int(k): tuple(v) for k, v in json.load(f).items()}
else:
    SOLAR_COLOR_MAP = {}

# Mapping from task to segmentation model path and number of classes
SEGMENT_MODELS = {
    'magnetic': {'path': 'logs/magnetic_model/best_model.pth', 'classes': 6},
    'phone': {'path': 'logs/phone_model_final/best_model.pth', 'classes': 4},
    'solar-panel': {'path': 'logs/solar-panel_model/best_model.pth', 'classes': 29},
    'steel': {'path': 'logs/steel_model/best_model_v11.pth', 'classes': 5},
}
# Classification model path
CLASSIFIER_PATH = 'logs/Classify/best_model.pth'

# Pre-load models lazily
_loaded_segmenters = {}
_classifier = None
_progress = {
    'total': 0,
    'processed': 0,
    'done': False,
    'stats': None,
    'class_counts': None,
    'class_img_map': None,
    'task': None,
    'results': None
}

def load_segmenter(task):
    if task not in _loaded_segmenters:
        info = SEGMENT_MODELS[task]
        model = get_unet_model('efficientnet-b4', None, 3, info['classes'])
        if os.path.exists(info['path']):
            model.load_state_dict(torch.load(info['path'], map_location='cpu'))
        model.eval()
        _loaded_segmenters[task] = model
    return _loaded_segmenters[task]

def load_classifier():
    global _classifier
    if _classifier is None:
        model = create_classifier(len(SEGMENT_MODELS))
        if os.path.exists(CLASSIFIER_PATH):
            model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location='cpu'))
        model.eval()
        _classifier = model
    return _classifier

# transforms
cls_tf = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def seg_predict(image_path, task, out_name):
    model = load_segmenter(task)
    info = SEGMENT_MODELS[task]
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    tensor = tf(img).unsqueeze(0)
    h, w = tensor.shape[-2:]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    tensor_pad = F.pad(tensor, (0, pad_w, 0, pad_h))
    with torch.no_grad():
        logits, _ = model(tensor_pad)
        logits = logits[..., :h, :w]
    pred = logits.argmax(1).squeeze(0).cpu().numpy()
    pred_3d = np.zeros((info['classes'], pred.shape[0], pred.shape[1]), dtype=np.uint8)
    for c in range(info['classes']):
        pred_3d[c] = (pred == c)
    visualize_prediction_3d(img_np, pred_3d, out_name, str(PRED_DIR), task=task)
    return PRED_DIR / out_name, pred

@app.route('/')
def index():
    year = datetime.now().year
    return render_template('index.html', tasks=list(SEGMENT_MODELS.keys()), current_year=year)

@app.route('/task/<task>')
def task_page(task):
    if task not in SEGMENT_MODELS:
        return redirect(url_for('index'))
    year = datetime.now().year
    return render_template('task.html', task=task, current_year=year)

@app.route('/task/<task>/single', methods=['POST'])
def task_single(task):
    if task not in SEGMENT_MODELS:
        return redirect(url_for('index'))
    file = request.files.get('image')
    if not file:
        return redirect(url_for('task_page', task=task))
    save_path = UPLOAD_DIR / file.filename
    file.save(save_path)
    out_name = f"{task}_{file.filename}"
    pred_path, _ = seg_predict(save_path, task, out_name)
    year = datetime.now().year
    return render_template(
        'single_result.html',
        original_image=f'uploads/{file.filename}',
        result_image=f'pred_vis/{out_name}',
        current_year=year
    )

@app.route('/task/<task>/start_batch', methods=['POST'])
def start_batch(task):
    if task not in SEGMENT_MODELS:
        return redirect(url_for('index'))
    files = request.files.getlist('images')
    if not files:
        return redirect(url_for('task_page', task=task))
    info = SEGMENT_MODELS[task]
    saved_paths = []
    for f in files:
        fname = Path(f.filename)
        save_path = UPLOAD_DIR / fname
        save_path.parent.mkdir(parents=True, exist_ok=True)
        f.save(save_path)
        saved_paths.append(save_path)
    _progress['total'] = len(saved_paths)
    _progress['processed'] = 0
    _progress['done'] = False
    _progress['task'] = task
    _progress['stats'] = None
    _progress['class_counts'] = None
    _progress['class_img_map'] = None
    _progress['results'] = None

    def worker(path_list):
        stats = {'total_images': len(path_list), 'defect_images': 0}
        class_counts = {f'class_{i}': 0 for i in range(1, info["classes"])}
        class_img_map = {f'class_{i}': [] for i in range(1, info["classes"])}
        for path in path_list:
            fname = Path(path).name
            out_name = f"{task}_{fname}"
            _, mask = seg_predict(path, task, out_name)
            unique = np.unique(mask)
            if np.any(mask > 0):
                stats['defect_images'] += 1
            for c in unique:
                if c == 0:
                    continue
                key = f'class_{c}'
                class_counts[key] += 1
                class_img_map[key].append(out_name)
            _progress['processed'] += 1
        _progress['stats'] = stats
        _progress['class_counts'] = class_counts
        _progress['class_img_map'] = class_img_map
        _progress['done'] = True

    threading.Thread(target=worker, args=(saved_paths,)).start()
    return render_template('progress.html', result_url=url_for('result_batch'))

@app.route('/progress_status')
def progress_status():
    return jsonify({
        'total': _progress['total'],
        'processed': _progress['processed'],
        'done': _progress['done']
    })

@app.route('/result_batch')
def result_batch():
    if not _progress['done']:
        return redirect(url_for('index'))
    year = datetime.now().year
    return render_template(
        'batch_result.html',
        stats=_progress['stats'],
        class_counts_mapped=_progress['class_counts'],
        class_img_map_mapped=_progress['class_img_map'],
        current_year=year
    )


@app.route('/download_report')
def download_report():
    if not _progress['done'] or _progress.get('stats') is None:
        return redirect(url_for('index'))
    report_dir = BASE_DIR / 'static' / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = report_dir / 'report.pdf'
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    c.setFont("Helvetica", 16)
    c.drawString(50, 800, "检测报告")
    stats = _progress['stats']
    y = 760
    c.setFont("Helvetica", 12)

    if _progress['task'] == 'classify':
        total = sum(v['total_images'] for v in stats.values())
        c.drawString(50, y, f"总图片数: {total}")
        y -= 20
        for task, s in stats.items():
            c.drawString(50, y, f"{task}: {s['total_images']} 张, 有缺陷 {s['defect_images']}")
            y -= 20
        if _progress.get('class_counts'):
            for task, counts in _progress['class_counts'].items():
                y -= 20
                c.drawString(50, y, f"{task} 缺陷统计:")
                y -= 20
                for cls, cnt in counts.items():
                    if cnt > 0:
                        c.drawString(70, y, f"{cls}: {cnt}")
                        y -= 20
    else:
        c.drawString(50, y, f"总图片数: {stats['total_images']}")
        y -= 20
        c.drawString(50, y, f"有缺陷图片数: {stats['defect_images']}")
        if _progress.get('class_counts'):
            y -= 40
            c.drawString(50, y, "各缺陷类型统计:")
            y -= 20
            for cls, cnt in _progress['class_counts'].items():
                c.drawString(70, y, f"{cls}: {cnt}")
                y -= 20
    c.save()
    return send_file(pdf_path, as_attachment=True)

@app.route('/classify', methods=['GET'])
def classify_upload():
    year = datetime.now().year
    return render_template('classify.html', current_year=year)


@app.route('/classify/start', methods=['POST'])
def start_classify():
    files = request.files.getlist('images')
    if not files:
        return redirect(url_for('classify_upload'))
    saved_paths = []
    for f in files:
        fname = Path(f.filename)
        save_path = UPLOAD_DIR / fname
        save_path.parent.mkdir(parents=True, exist_ok=True)
        f.save(save_path)
        saved_paths.append(save_path)
    _progress['total'] = len(saved_paths)
    _progress['processed'] = 0
    _progress['done'] = False
    _progress['task'] = 'classify'
    _progress['stats'] = None
    _progress['class_counts'] = None
    _progress['class_img_map'] = None
    _progress['results'] = []

    def worker(path_list):
        classifier = load_classifier()
        # 初始化统计结构
        task_stats = {t: {'total_images': 0, 'defect_images': 0} for t in SEGMENT_MODELS}
        task_class_counts = {t: {f'class_{i}': 0 for i in range(1, SEGMENT_MODELS[t]['classes'])} for t in SEGMENT_MODELS}
        for path in path_list:
            img = Image.open(path).convert('RGB')
            x = cls_tf(img).unsqueeze(0)
            with torch.no_grad():
                out = classifier(x)
            pred_idx = out.argmax(1).item()
            task = list(SEGMENT_MODELS.keys())[pred_idx]
            out_name = f"cls_{task}_{Path(path).name}"
            _, mask = seg_predict(path, task, out_name)
            _progress['results'].append({'task': task, 'image': out_name})
            task_stats[task]['total_images'] += 1
            if np.any(mask > 0):
                task_stats[task]['defect_images'] += 1
            unique = np.unique(mask)
            for c in unique:
                if c == 0:
                    continue
                key = f'class_{c}'
                task_class_counts[task][key] += 1
            _progress['processed'] += 1
        _progress['stats'] = task_stats
        _progress['class_counts'] = task_class_counts
        _progress['done'] = True

    threading.Thread(target=worker, args=(saved_paths,)).start()
    return render_template('progress.html', result_url=url_for('result_classify'))


@app.route('/result_classify')
def result_classify():
    if not _progress['done'] or _progress.get('results') is None:
        return redirect(url_for('index'))
    year = datetime.now().year
    stats = _progress.get('stats', {})
    class_counts = _progress.get('class_counts', {})
    # 任务颜色 (固定)
    task_colors = {
        'steel': '#e74c3c',
        'phone': '#3498db',
        'magnetic': '#2ecc71',
        'solar-panel': '#9b59b6'
    }
    task_counts = {TASK_INFO['tasks'][t]['task_name_cn']: s['total_images'] for t, s in stats.items() if s['total_images'] > 0}

    class_charts = {}
    for t, counts in class_counts.items():
        name_cn = TASK_INFO['tasks'][t]['task_name_cn']
        names_map = TASK_INFO['tasks'][t]['class_names']
        color_map = SOLAR_COLOR_MAP if t == 'solar-panel' else TASK_COLOR_MAPS.get(t, {})
        data = {}
        colors = {}
        for key, cnt in counts.items():
            if cnt == 0:
                continue
            cls_id = int(key.split('_')[1])
            label = names_map.get(str(cls_id), {}).get('cn', key)
            data[label] = cnt
            rgb = color_map.get(cls_id, (128, 128, 128))
            colors[label] = f'rgb({rgb[0]}, {rgb[1]}, {rgb[2]})'
        class_charts[t] = {'name_cn': name_cn, 'counts': data, 'colors': colors}

    results_by_task = {}
    for item in _progress['results']:
        results_by_task.setdefault(item['task'], []).append(item['image'])

    return render_template(
        'classify_result.html',
        results_by_task=results_by_task,
        task_counts=task_counts,
        task_colors=task_colors,
        class_charts=class_charts,
        current_year=year
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
