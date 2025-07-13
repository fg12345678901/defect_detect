from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import os
import json
import shutil
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
import threading
from datetime import datetime
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pdfkit
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import base64
import io

from models.classifier import create_classifier
from models.segmenter import get_unet_model
from utils.viz import visualize_prediction_3d, TASK_COLOR_MAPS

app = Flask(__name__)

# ---- Configuration ----
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
PRED_DIR = BASE_DIR / "static" / "pred_vis"
CATEGORIZED_DIR = BASE_DIR / "static" / "categorized"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
CATEGORIZED_DIR.mkdir(parents=True, exist_ok=True)

# 注册中文字体，依次尝试系统中的常见字体
try:
    candidate_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    font_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            font_path = p
            break
    if font_path:
        pdfmetrics.registerFont(TTFont("SimHei", font_path))
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.sans-serif"] = [
            "SimHei",
            "Noto Sans CJK SC",
            "WenQuanYi Zen Hei",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


def render_pdf_from_html(html_content, pdf_path):
    """Render PDF using wkhtmltopdf via pdfkit for better Chinese support."""
    wkhtml = shutil.which("wkhtmltopdf") or "/usr/bin/wkhtmltopdf"
    config = pdfkit.configuration(wkhtmltopdf=wkhtml) if os.path.exists(wkhtml) else None
    options = {
        "encoding": "UTF-8",
        "enable-local-file-access": None,
    }
    pdfkit.from_string(html_content, str(pdf_path), configuration=config, options=options)


# 加载任务信息和颜色映射
TASK_INFO_PATH = BASE_DIR / "tasks_info.json"
with open(TASK_INFO_PATH, "r", encoding="utf-8") as f:
    TASK_INFO = json.load(f)

SOLAR_COLOR_MAP_PATH = BASE_DIR.parent / "dataset" / "solar-panel" / "color_map.json"
if SOLAR_COLOR_MAP_PATH.exists():
    with open(SOLAR_COLOR_MAP_PATH, "r", encoding="utf-8") as f:
        SOLAR_COLOR_MAP = {int(k): tuple(v) for k, v in json.load(f).items()}
else:
    SOLAR_COLOR_MAP = {}

# 交互式图表使用的颜色映射（与前端保持一致）
CHART_COLOR_MAPS = {
    "steel": {
        1: (255, 99, 132),
        2: (75, 192, 192),
        3: (54, 162, 235),
        4: (255, 206, 86),
    },
    "phone": {
        1: (153, 102, 255),
        2: (255, 159, 64),
        3: (75, 192, 192),
    },
    "magnetic": {
        1: (255, 99, 132),
        2: (75, 192, 192),
        3: (54, 162, 235),
        4: (255, 206, 86),
        5: (153, 102, 255),
    },
}

# Mapping from task to segmentation model path and number of classes
SEGMENT_MODELS = {
    "magnetic": {"path": "logs/magnetic_model/best_model.pth", "classes": 6},
    "phone": {"path": "logs/phone_model_final/best_model.pth", "classes": 4},
    "solar-panel": {"path": "logs/solar-panel_model/best_model.pth", "classes": 29},
    "steel": {"path": "logs/steel_model/best_model_v11.pth", "classes": 5},
}
# Classification model path
CLASSIFIER_PATH = "logs/Classify/best_model.pth"

# Pre-load models lazily
_loaded_segmenters = {}
_classifier = None
_progress = {
    "total": 0,
    "processed": 0,
    "done": False,
    "stats": None,
    "class_counts": None,
    "class_img_map": None,
    "task": None,
    "results": None,
}


def load_segmenter(task):
    if task not in _loaded_segmenters:
        info = SEGMENT_MODELS[task]
        model = get_unet_model("efficientnet-b4", None, 3, info["classes"])
        if os.path.exists(info["path"]):
            model.load_state_dict(torch.load(info["path"], map_location="cpu"))
        model.eval()
        _loaded_segmenters[task] = model
    return _loaded_segmenters[task]


def load_classifier():
    global _classifier
    if _classifier is None:
        model = create_classifier(len(SEGMENT_MODELS))
        if os.path.exists(CLASSIFIER_PATH):
            model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location="cpu"))
        model.eval()
        _classifier = model
    return _classifier


# transforms
cls_tf = transforms.Compose(
    [
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def seg_predict(image_path, task, out_name):
    model = load_segmenter(task)
    info = SEGMENT_MODELS[task]
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = tf(img).unsqueeze(0)
    h, w = tensor.shape[-2:]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    tensor_pad = F.pad(tensor, (0, pad_w, 0, pad_h))
    with torch.no_grad():
        logits, _ = model(tensor_pad)
        logits = logits[..., :h, :w]
    pred = logits.argmax(1).squeeze(0).cpu().numpy()
    pred_3d = np.zeros((info["classes"], pred.shape[0], pred.shape[1]), dtype=np.uint8)
    for c in range(info["classes"]):
        pred_3d[c] = pred == c
    visualize_prediction_3d(img_np, pred_3d, out_name, str(PRED_DIR), task=task)
    return PRED_DIR / out_name, pred


def categorize_images(stats, class_img_map, task):
    """根据检测结果将图片按缺陷类型分类到不同文件夹"""
    # 清空分类目录
    if CATEGORIZED_DIR.exists():
        shutil.rmtree(CATEGORIZED_DIR)
    CATEGORIZED_DIR.mkdir(parents=True, exist_ok=True)

    # 创建无缺陷目录
    no_defect_dir = CATEGORIZED_DIR / "无缺陷"
    no_defect_dir.mkdir(exist_ok=True)

    # 获取任务的类名映射
    task_info = TASK_INFO["tasks"].get(task, {})
    class_names = task_info.get("class_names", {})

    # 为每个缺陷类创建目录
    for cls_key, img_list in class_img_map.items():
        if img_list:
            cls_id = cls_key.split("_")[1]
            cls_name_cn = class_names.get(cls_id, {}).get("cn", cls_key)
            class_dir = CATEGORIZED_DIR / cls_name_cn
            class_dir.mkdir(exist_ok=True)

            # 复制检测结果图到对应目录
            for img_name in img_list:
                src_path = PRED_DIR / img_name
                if src_path.exists():
                    shutil.copy2(src_path, class_dir / img_name)


def generate_pie_chart_base64(data, task, title="缺陷分布", donut=False):
    """生成饼图并返回base64编码, 可选是否生成环形图"""
    if not data or all(v == 0 for v in data.values()):
        return None

    # 获取颜色映射
    if task == "solar-panel":
        color_map = {}
    else:
        color_map = CHART_COLOR_MAPS.get(task, {})
    name_to_id = {info["cn"]: int(cid) for cid, info in TASK_INFO["tasks"].get(task, {}).get("class_names", {}).items()}

    labels = []
    sizes = []
    colors = []

    for k, v in data.items():
        if v > 0:
            labels.append(k)
            sizes.append(v)
            # 根据类别编号选择颜色
            cls_id = None
            if "class_" in k:
                cls_id = int(k.split("_")[1])
            else:
                cls_id = name_to_id.get(k)
            if task == "solar-panel":
                # 按照顺序生成渐变色，与前端保持一致
                idx = len(colors)
                hue = (idx * 360 / len([vv for vv in data.values() if vv > 0])) % 360
                import colorsys

                r, g, b = colorsys.hsv_to_rgb(hue / 360, 0.7, 0.8)
                colors.append([r, g, b])
            else:
                if cls_id is not None:
                    rgb = color_map.get(cls_id, (128, 128, 128))
                    r, g, b = rgb
                    colors.append([r / 255.0, g / 255.0, b / 255.0])
                else:
                    colors.append([0.5, 0.5, 0.5])

    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 创建饼图
    fig, ax = plt.subplots(figsize=(8, 6))
    if labels:
        wedgeprops = {"width": 0.4, "edgecolor": "white"} if donut else None
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            textprops={"fontsize": 12},
            wedgeprops=wedgeprops,
        )
        # 设置标签字体
        for text in texts:
            text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(10)
            autotext.set_weight("bold")

    ax.set_title(title, fontsize=16, weight="bold", pad=20)

    # 转换为base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return image_base64


@app.route("/")
def index():
    year = datetime.now().year
    # 将智能分类放在最前面
    tasks_ordered = ["classify"] + list(SEGMENT_MODELS.keys())
    return render_template(
        "index.html",
        tasks=list(SEGMENT_MODELS.keys()),
        tasks_ordered=tasks_ordered,
        task_info=TASK_INFO,
        current_year=year,
    )


@app.route("/task/<task>")
def task_page(task):
    if task not in SEGMENT_MODELS:
        return redirect(url_for("index"))
    year = datetime.now().year
    task_info = TASK_INFO["tasks"].get(task, {})
    return render_template("task.html", task=task, task_info=task_info, current_year=year)


@app.route("/task/<task>/single", methods=["POST"])
def task_single(task):
    if task not in SEGMENT_MODELS:
        return redirect(url_for("index"))
    file = request.files.get("image")
    if not file:
        return redirect(url_for("task_page", task=task))
    save_path = UPLOAD_DIR / file.filename
    file.save(save_path)
    out_name = f"{task}_{file.filename}"
    pred_path, mask = seg_predict(save_path, task, out_name)

    # 获取检测到的缺陷类型
    task_info = TASK_INFO["tasks"].get(task, {})
    class_names = task_info.get("class_names", {})
    detected_classes = []
    for c in np.unique(mask):
        if c > 0:
            cls_name = class_names.get(str(c), {}).get("cn", f"类别{c}")
            detected_classes.append(cls_name)

    year = datetime.now().year
    return render_template(
        "single_result.html",
        original_image=f"uploads/{file.filename}",
        result_image=f"pred_vis/{out_name}",
        detected_classes=detected_classes,
        task=task,
        task_info=task_info,
        current_year=year,
    )


@app.route("/task/<task>/start_batch", methods=["POST"])
def start_batch(task):
    if task not in SEGMENT_MODELS:
        return redirect(url_for("index"))
    files = request.files.getlist("images")
    if not files:
        return redirect(url_for("task_page", task=task))
    info = SEGMENT_MODELS[task]
    saved_paths = []
    for f in files:
        fname = Path(f.filename)
        save_path = UPLOAD_DIR / fname
        save_path.parent.mkdir(parents=True, exist_ok=True)
        f.save(save_path)
        saved_paths.append(save_path)
    _progress["total"] = len(saved_paths)
    _progress["processed"] = 0
    _progress["done"] = False
    _progress["task"] = task
    _progress["stats"] = None
    _progress["class_counts"] = None
    _progress["class_img_map"] = None
    _progress["results"] = None

    def worker(path_list):
        task_info = TASK_INFO["tasks"].get(task, {})
        class_names = task_info.get("class_names", {})

        stats = {"total_images": len(path_list), "defect_images": 0}
        class_counts = {}
        class_img_map = {}

        # 初始化类别统计
        for i in range(1, info["classes"]):
            cls_name = class_names.get(str(i), {}).get("cn", f"类别{i}")
            class_counts[cls_name] = 0
            class_img_map[f"class_{i}"] = []

        for path in path_list:
            fname = Path(path).name
            out_name = f"{task}_{fname}"
            _, mask = seg_predict(path, task, out_name)
            unique = np.unique(mask)
            if np.any(mask > 0):
                stats["defect_images"] += 1
            for c in unique:
                if c == 0:
                    continue
                cls_name = class_names.get(str(c), {}).get("cn", f"类别{c}")
                class_counts[cls_name] += 1
                class_img_map[f"class_{c}"].append(out_name)
            _progress["processed"] += 1

        # 分类存储图片
        categorize_images(stats, class_img_map, task)

        _progress["stats"] = stats
        _progress["class_counts"] = class_counts
        _progress["class_img_map"] = class_img_map
        _progress["done"] = True

    threading.Thread(target=worker, args=(saved_paths,)).start()
    return render_template("progress.html", result_url=url_for("result_batch"))


@app.route("/progress_status")
def progress_status():
    return jsonify(
        {
            "total": _progress["total"],
            "processed": _progress["processed"],
            "done": _progress["done"],
        }
    )


@app.route("/result_batch")
def result_batch():
    if not _progress["done"]:
        return redirect(url_for("index"))

    task = _progress["task"]
    task_info = TASK_INFO["tasks"].get(task, {})

    # 生成饼图
    pie_chart_base64 = generate_pie_chart_base64(
        _progress["class_counts"],
        task,
        f"{task_info.get('task_name_cn', task)}缺陷分布",
        donut=True,
    )

    year = datetime.now().year
    return render_template(
        "batch_result.html",
        stats=_progress["stats"],
        class_counts_mapped=_progress["class_counts"],
        class_img_map_mapped=_progress["class_img_map"],
        pie_chart_base64=pie_chart_base64,
        task=task,
        task_info=task_info,
        current_year=year,
    )


@app.route("/download_report")
def download_report():
    if not _progress["done"] or _progress.get("stats") is None:
        return redirect(url_for("index"))

    # 使用report.html模板生成HTML内容
    task = _progress.get("task")
    if task == "classify":
        # 自动分类报告需要特殊处理
        stats = _progress["stats"]
        total = sum(v["total_images"] for v in stats.values())
        total_defects = sum(v["defect_images"] for v in stats.values())

        task_counts = {}
        for t, s in stats.items():
            if s["total_images"] > 0:
                task_cn = TASK_INFO["tasks"][t]["task_name_cn"]
                task_counts[task_cn] = s["total_images"]

        task_pie_base64 = None
        if task_counts:
            plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False

            labels = list(task_counts.keys())
            sizes = list(task_counts.values())
            colors = [
                f"#{r:02x}{g:02x}{b:02x}"
                for r, g, b in [
                    (231, 76, 60),
                    (52, 152, 219),
                    (46, 204, 113),
                    (155, 89, 182),
                ][: len(labels)]
            ]

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(
                sizes,
                labels=labels,
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
                textprops={"fontsize": 12},
            )
            ax.set_title("任务类型分布", fontsize=16, weight="bold", pad=20)

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
            buf.seek(0)
            task_pie_base64 = base64.b64encode(buf.getvalue()).decode()
            plt.close(fig)

        # 准备报告数据
        report_data = {
            "task": "classify",
            "stats": {"total_images": total, "defect_images": total_defects},
            "task_stats": stats,
            "class_counts_all": _progress["class_counts"],
            "task_pie_charts": {
                t: generate_pie_chart_base64(
                    c,
                    t,
                    f"{TASK_INFO['tasks'][t]['task_name_cn']}缺陷分布",
                    donut=False,
                )
                for t, c in _progress["class_counts"].items()
                if any(v > 0 for v in c.values())
            },
            "task_pie_base64": task_pie_base64,
            "current_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_year": datetime.now().year,
            "task_info": TASK_INFO,
        }

        html_content = render_template("report.html", **report_data)
    else:
        # 单任务报告
        report_data = {
            "stats": _progress["stats"],
            "class_counts_mapped": _progress["class_counts"],
            "class_img_map_mapped": _progress["class_img_map"],
            "pie_chart_base64": generate_pie_chart_base64(
                _progress["class_counts"],
                task,
                f"{TASK_INFO['tasks'][task]['task_name_cn']}缺陷分布",
                donut=False,
            ),
            "current_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_year": datetime.now().year,
            "task": task,
            "task_info": TASK_INFO["tasks"].get(task, {}),
        }

        # 渲染report.html模板
        html_content = render_template("report.html", **report_data)

    # 将 HTML 转为 PDF
    report_dir = BASE_DIR / "static" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = report_dir / f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'

    render_pdf_from_html(html_content, pdf_path)

    return send_file(
        pdf_path,
        as_attachment=True,
        download_name=f"检测报告_{datetime.now().strftime('%Y%m%d')}.pdf",
    )


@app.route("/classify", methods=["GET"])
def classify_upload():
    year = datetime.now().year
    return render_template("classify.html", task_info=TASK_INFO, current_year=year)


@app.route("/classify/start", methods=["POST"])
def start_classify():
    files = request.files.getlist("images")
    if not files:
        return redirect(url_for("classify_upload"))
    saved_paths = []
    for f in files:
        fname = Path(f.filename)
        save_path = UPLOAD_DIR / fname
        save_path.parent.mkdir(parents=True, exist_ok=True)
        f.save(save_path)
        saved_paths.append(save_path)
    _progress["total"] = len(saved_paths)
    _progress["processed"] = 0
    _progress["done"] = False
    _progress["task"] = "classify"
    _progress["stats"] = None
    _progress["class_counts"] = None
    _progress["class_img_map"] = None
    _progress["results"] = []

    def worker(path_list):
        classifier = load_classifier()
        # 初始化统计结构
        task_stats = {t: {"total_images": 0, "defect_images": 0} for t in SEGMENT_MODELS}
        task_class_counts = {}
        task_class_img_map = {}

        for t in SEGMENT_MODELS:
            task_info = TASK_INFO["tasks"].get(t, {})
            class_names = task_info.get("class_names", {})
            task_class_counts[t] = {}
            task_class_img_map[t] = {}
            task_class_img_map[t]["no_defect"] = []
            for i in range(1, SEGMENT_MODELS[t]["classes"]):
                cls_name = class_names.get(str(i), {}).get("cn", f"类别{i}")
                task_class_counts[t][cls_name] = 0
                task_class_img_map[t][f"class_{i}"] = []

        # 创建分类结果存储目录
        classify_result_dir = CATEGORIZED_DIR / "auto_classify"
        if classify_result_dir.exists():
            shutil.rmtree(classify_result_dir)
        classify_result_dir.mkdir(parents=True, exist_ok=True)

        # 为每个任务创建目录
        for t in SEGMENT_MODELS:
            task_dir = classify_result_dir / TASK_INFO["tasks"][t]["task_name_cn"]
            task_dir.mkdir(exist_ok=True)
            (task_dir / "无缺陷").mkdir(exist_ok=True)
            # 为每个缺陷类型创建目录
            for i in range(1, SEGMENT_MODELS[t]["classes"]):
                cls_name = TASK_INFO["tasks"][t]["class_names"].get(str(i), {}).get("cn", f"类别{i}")
                (task_dir / cls_name).mkdir(exist_ok=True)

        for path in path_list:
            img = Image.open(path).convert("RGB")
            x = cls_tf(img).unsqueeze(0)
            with torch.no_grad():
                out = classifier(x)
            pred_idx = out.argmax(1).item()
            task = list(SEGMENT_MODELS.keys())[pred_idx]
            out_name = f"cls_{task}_{Path(path).name}"
            _, mask = seg_predict(path, task, out_name)

            _progress["results"].append({"task": task, "image": out_name, "has_defect": np.any(mask > 0)})

            task_stats[task]["total_images"] += 1
            task_cn = TASK_INFO["tasks"][task]["task_name_cn"]

            if np.any(mask > 0):
                task_stats[task]["defect_images"] += 1
                # 复制到对应缺陷文件夹
                for c in np.unique(mask):
                    if c > 0:
                        cls_name = TASK_INFO["tasks"][task]["class_names"].get(str(c), {}).get("cn", f"类别{c}")
                        dst_path = classify_result_dir / task_cn / cls_name / out_name
                        shutil.copy2(PRED_DIR / out_name, dst_path)
            else:
                # 复制到无缺陷文件夹
                dst_path = classify_result_dir / task_cn / "无缺陷" / out_name
                shutil.copy2(PRED_DIR / out_name, dst_path)
                task_class_img_map[task]["no_defect"].append(out_name)

            task_info = TASK_INFO["tasks"].get(task, {})
            class_names = task_info.get("class_names", {})
            unique = np.unique(mask)
            for c in unique:
                if c == 0:
                    continue
                cls_name = class_names.get(str(c), {}).get("cn", f"类别{c}")
                task_class_counts[task][cls_name] += 1
                task_class_img_map[task][f"class_{c}"].append(out_name)
            _progress["processed"] += 1

        _progress["stats"] = task_stats
        _progress["class_counts"] = task_class_counts
        _progress["class_img_map"] = task_class_img_map
        _progress["done"] = True

    threading.Thread(target=worker, args=(saved_paths,)).start()
    return render_template("progress.html", result_url=url_for("result_classify"))


@app.route("/result_classify")
def result_classify():
    if not _progress["done"] or _progress.get("results") is None:
        return redirect(url_for("index"))
    year = datetime.now().year
    stats = _progress.get("stats", {})
    class_counts = _progress.get("class_counts", {})
    class_img_map = _progress.get("class_img_map", {})

    # 任务颜色 (柔和的颜色)
    task_colors = {
        "steel": "#e74c3c",
        "phone": "#3498db",
        "magnetic": "#2ecc71",
        "solar-panel": "#9b59b6",
    }

    # 准备任务分布数据
    task_counts = {}
    for t, s in stats.items():
        if s["total_images"] > 0:
            task_cn = TASK_INFO["tasks"][t]["task_name_cn"]
            task_counts[task_cn] = s["total_images"]

    # 生成任务分布饼图
    task_pie_base64 = None
    if task_counts:
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        labels = list(task_counts.keys())
        sizes = list(task_counts.values())
        colors = [task_colors.get(t, "#95a5a6") for t in stats.keys() if stats[t]["total_images"] > 0]

        fig, ax = plt.subplots(figsize=(8, 6))
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
        # 设置字体
        for text in texts:
            text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(10)
            autotext.set_weight("bold")

        ax.set_title("任务类型分布", fontsize=16, weight="bold", pad=20)

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor="white")
        buffer.seek(0)
        task_pie_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)

    # 准备各任务的缺陷分布数据和图表
    class_charts = {}
    for t, counts in class_counts.items():
        if any(cnt > 0 for cnt in counts.values()):
            # 生成该任务的缺陷分布饼图
            pie_base64 = generate_pie_chart_base64(
                counts,
                t,
                f"{TASK_INFO['tasks'][t]['task_name_cn']}缺陷分布",
                donut=True,
            )
            if pie_base64:
                class_charts[t] = {
                    "name_cn": TASK_INFO["tasks"][t]["task_name_cn"],
                    "pie_chart": pie_base64,
                }

    # 准备结果展示
    results_by_task = {}
    for item in _progress["results"]:
        task = item["task"]
        if task not in results_by_task:
            results_by_task[task] = []
        results_by_task[task].append(item)

    return render_template(
        "classify_result.html",
        results_by_task=results_by_task,
        task_counts=task_counts,
        task_pie_base64=task_pie_base64,
        class_charts=class_charts,
        stats=stats,
        class_counts=class_counts,
        class_img_map=class_img_map,
        task_info=TASK_INFO,
        current_year=year,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
