import cv2 as cv
import numpy as np
from hobot_dnn import pyeasy_dnn as dnn


def print_properties(pro):
    print("tensor type:", pro.tensor_type)
    print("data type:", pro.dtype)
    print("layout:", pro.layout)
    print("shape:", pro.shape)


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


def main_bgr():
    # 单输入NV12模型
    models = dnn.load("./mobilenet_unet_1024x2048_nv12.bin")
    # 获取模型输入信息
    print_properties(models[0].inputs[0].properties)
    input_dtype = models[0].inputs[0].properties.dtype
    # 构造模型NV12输入数据 (h*1.5,w)
    bgr = cv.imread("./test.jpg", 1)
    resized_data = cv.resize(bgr, (2048, 1024), interpolation=cv.INTER_AREA)
    tensor = bgr2nv12_opencv(resized_data)
    outputs = models[0].forward(tensor)
    # 获取模型输出数据，类型为numpy数据
    output_buffer = outputs[0].buffer
    print(f"{output_buffer.shape=}")
    return output_buffer


if __name__ == "__main__":
    main_bgr()
