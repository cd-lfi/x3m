# hobot_dnn
本目录的示例运行在板端(X3M)。当你的板端镜像比较旧时：
```python
import sys

sys.path.append('/usr/lib/hobot-srcampy')
```

[旭日X3派用户手册](https://developer.horizon.ai/api/v1/fileData/documents_pi/index.html)

## nv12 -> bgr
```python
import cv2 as cv
import numpy as np
from hobot_dnn import pyeasy_dnn as dnn
from hobot_vio import libsrcampy as srcampy

cam = srcampy.Camera()
width, height = 1920, 1080
cam.open_cam(0, 1, 30, width, height)

# nv12格式的图像数据 - 参数 2 代表从硬件模块IPU中获取
origin_image = cam.get_img(2, width=1920, height=1080)
origin_nv12 = np.frombuffer(origin_image, dtype=np.uint8).reshape(1620, 1920)
origin_bgr = cv2.cvtColor(origin_nv12, cv2.COLOR_YUV420SP2BGR)
```
