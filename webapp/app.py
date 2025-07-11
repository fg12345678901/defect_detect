from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
from datetime import datetime

from models.classifier import create_classifier
from models.segmenter import get_unet_model
from utils.viz import visualize_prediction_3d

app = Flask(__name__)

# ---- Configuration ----
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / 'uploads'
PRED_DIR = BASE_DIR / 'static' / 'pred_vis'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

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
    return render_template('index.html', tasks=list(SEGMENT_MODELS.keys()))

@app.route('/task/<task>')
def task_page(task):
    if task not in SEGMENT_MODELS:
        return redirect(url_for('index'))
    return render_template('task.html', task=task)

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
    return send_file(pred_path, mimetype='image/png')

@app.route('/task/<task>/batch', methods=['POST'])
def task_batch(task):
    if task not in SEGMENT_MODELS:
        return redirect(url_for('index'))
    files = request.files.getlist('images')
    if not files:
        return redirect(url_for('task_page', task=task))
    info = SEGMENT_MODELS[task]
    stats = {'total_images': len(files), 'defect_images': 0}
    class_counts = {f'class_{i}': 0 for i in range(1, info["classes"])}
    class_img_map = {f'class_{i}': [] for i in range(1, info["classes"])}
    out_files = []
    for file in files:
        save_path = UPLOAD_DIR / file.filename
        file.save(save_path)
        out_name = f"{task}_{file.filename}"
        pred_path, mask = seg_predict(save_path, task, out_name)
        out_files.append(out_name)
        unique = np.unique(mask)
        if np.any(mask > 0):
            stats['defect_images'] += 1
        for c in unique:
            if c == 0:
                continue
            key = f'class_{c}'
            class_counts[key] += 1
            class_img_map[key].append(out_name)
    year = datetime.now().year
    return render_template(
        'batch_result.html',
        stats=stats,
        class_counts_mapped=class_counts,
        class_img_map_mapped=class_img_map,
        current_year=year
    )

@app.route('/classify', methods=['GET', 'POST'])
def classify_upload():
    if request.method == 'GET':
        return render_template('classify.html')
    files = request.files.getlist('images')
    if not files:
        return redirect(url_for('classify_upload'))
    classifier = load_classifier()
    results = []
    for file in files:
        save_path = UPLOAD_DIR / file.filename
        file.save(save_path)
        img = Image.open(save_path).convert('RGB')
        x = cls_tf(img).unsqueeze(0)
        with torch.no_grad():
            out = classifier(x)
        pred_idx = out.argmax(1).item()
        task = list(SEGMENT_MODELS.keys())[pred_idx]
        out_name = f"cls_{task}_{file.filename}"
        pred_path, _ = seg_predict(save_path, task, out_name)
        results.append({'task': task, 'image': out_name})
    return render_template('classify_result.html', results=results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
