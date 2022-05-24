import colorsys
import traceback

import cv2 as cv
import numpy as np
from hobot_dnn import pyeasy_dnn as dnn
from hobot_vio import libsrcampy as srcampy


def print_properties(pro):
    print("tensor type:", pro.tensor_type)
    print("data type:", pro.dtype)
    print("layout:", pro.layout)
    print("shape:", pro.shape)


def get_colors(classes):
    num_classes = len(classes)
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = map(lambda x: tuple(int(i * 255) for i in x), colors)
    return list(colors)


def get_hw(pro):
    if pro.layout == "NCHW":
        return pro.shape[2], pro.shape[3]
    else:
        return pro.shape[1], pro.shape[2]


def bgr2nv12_opencv(image):
    height, width = image.shape[0], image.shape[1]
    area = height * width
    yuv420p = cv.cvtColor(
        image, cv.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

    nv12 = np.zeros_like(yuv420p)
    nv12[:height * width] = y
    nv12[height * width:] = uv_packed
    return nv12


def main_camera_usb():
    models = dnn.load("./fcos_512x512_nv12.bin")
    print_properties(models[0].inputs[0].properties)

    cap = cv.VideoCapture(8)
    if cap.isOpened():
        try:
            # 设置图像格式为 MJPEG / 分辨率为 512x512
            codec = cv.VideoWriter_fourcc("M", "J", "P", "G")
            cap.set(cv.CAP_PROP_FOURCC, codec)
            cap.set(cv.CAP_PROP_FPS, 30)
            cap.set(cv.CAP_PROP_FRAME_WIDTH, 512)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, 512)

            ret, bgr = cap.read()
            nv12_data = bgr2nv12_opencv(bgr)

            outputs = models[0].forward(nv12_data)
        except Exception as e:
            print(f"ERROR: {traceback.format_exc()}")
        cap.release()


def main_camera_mipi():
    models = dnn.load("./fcos_512x512_nv12.bin")
    print_properties(models[0].inputs[0].properties)

    cam = srcampy.Camera()
    # 打开 F37, 初始化视频 pipeline 0, 设置帧率30fps, 缩放图像为 512x512
    h, w = get_hw(models[0].inputs[0].properties)
    cam.open_cam(0, 1, 30, w, h)

    # 从相机获取分辨率为 512x512 的nv12格式的图像数据, 参数 2 代表从硬件模块IPU中获取
    nv12_data = cam.get_img(2, 512, 512)
    # 把图像数据转成 numpy 数据类型
    nv12_data = np.frombuffer(nv12_data, dtype=np.uint8)

    outputs = models[0].forward(nv12_data)
