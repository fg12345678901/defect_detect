# defect_detect

本项目提供工业表面缺陷检测的 Web 演示界面，运行前需安装部分依赖。

```bash
pip install flask torch torchvision pdfkit

# 系统需安装 `wkhtmltopdf` 用于生成 PDF：
sudo apt-get install wkhtmltopdf
```

运行 `webapp/app.py` 即可启动服务。