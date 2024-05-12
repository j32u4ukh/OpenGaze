# OpenGaze

目前的縮放基本上就是跟著整體縮放而已，並沒有隨著距離越遠而縮放程度增加

#### 將已安裝套件輸出到 requirements.txt

```bash
pip freeze > requirements.txt
```

#### 目前遇到的錯誤

應該是需要安裝 QT 相關的 plugins，但目前安裝指令都遇到問題。

指令先在容器內測試，沒問題再修正 Dockerfile

```
root@1f3ce84aa49d:/home/opengaze# python __init__.py 
qt.qpa.xcb: could not connect to display 
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/usr/local/lib/python3.12/site-packages/cv2/qt/plugins" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: xcb.

Aborted
```

```
apt-get install xorg openbox

apt-get install -y \
    libopencv-dev \
    libx11-dev



先打开调试信息
export QT_DEBUG_PLUGINS=1

apt-get install -y xvfb




xvfb-run python3 -m pyqtgraph.util.get_resolution

apt-cache search xxx



apt-get install -y libgl1-mesa-glx libsm6 libxrender1 libxext-dev



apt-get install -y libtiff5-dev libjpeg-dev libjasper-dev libpng-dev libopenexr6-dev libgdal1-dev libwebp-dev libvpx-dev libswscale-dev libx264-dev


```