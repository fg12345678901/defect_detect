# defect_detect

本项目提供工业表面缺陷检测的 Web 演示界面，运行前需安装部分依赖。

```bash
pip install flask torch torchvision xhtml2pdf
# 为了在生成的 PDF 中正确显示中文，需要安装中文字体
sudo apt-get install -y fonts-noto-cjk
# Windows 用户请确保系统安装有 "SimHei" 或其他中文字体，以便 PDF 正确显示中文。
```

运行 `webapp/app.py` 即可启动服务。
