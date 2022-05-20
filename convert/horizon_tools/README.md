- [高性能模型设计建议](https://developer.horizon.ai/api/v1/fileData/documents/ai_toolchain_develop/horizon_ai_toolchain_user_guide/chapter_3_model_conversion.html#id17)
- [在开发板上评测模型的推理性能](https://developer.horizon.ai/api/v1/fileData/hrt_model_exec/index.html)
- [量化工具链使用说明 » 模型转换](https://developer.horizon.ai/api/v1/fileData/documents/ai_toolchain_develop/horizon_ai_toolchain_user_guide/chapter_3_model_conversion.html)

## data
`./calibration_data_rgb_f32` build with:
```sh
# work on: convert/horizon_tools/
                     src_dir                          dst_dir                          HxW               NCHW/NHWC
python preprocess.py coco2017_cat_dog/validation/data calibration_data_rgb_f32 --imgsz 640x640 --channel first
```

## docker
```sh
docker pull openexplorer/ai_toolchain_centos_7_xj3:v2.0.4

# run
cd /workspace/docker/model_convert && \
IMAGE=openexplorer/ai_toolchain_centos_7_xj3:v2.0.4 && \
docker run -it --rm --network=host --ipc=host -v "$(pwd)":/workspace ${IMAGE} bash

# src
git clone -q --depth 1 https://github.com/cd-lfi/x3m.git
```

### onnx check
```sh
# yolov5: lite
onnx_model=/workspace/models/yolov5m_relu_sigmoid_fast_nc_15.onnx && \
input_node=images && \
input_shape=1x3x640x640 && \
bash onnx_check.sh ${onnx_model} ${input_node} ${input_shape}

# horizon.ai: model_zoo/mapper/detection/yolov5_onnx_optimized
onnx_model=/workspace/models/yolov5_onnx_optimized/YOLOv5l.onnx && \
input_node=data && \
input_shape=1x3x672x672 && \
bash onnx_check.sh ${onnx_model} ${input_node} ${input_shape}
```

### onnx makert
```sh
onnx_model=/workspace/models/yolov5_onnx_optimized/YOLOv5l.onnx && \
input_name=images && \
input_shape=640x640 && \
input_type_rt=nv12 && \
core_num=1 && \
bash onnx_makert_f32.sh ${onnx_model} ${input_name} ${input_shape} ${input_type_rt} ${core_num}
```

## board
https://developer.horizon.ai/api/v1/fileData/hrt_model_exec/index.html
```sh
# X3M_SDK_UBUNTU/ai_toolchain_package/
# - v2.0.4/ai_toolchain/hrt_tools/
chmod +x hrt_model_exec

hrt_model_exec -v

model=models/yolov5_672x672_nv12.bin

hrt_model_exec model_info --model_file=${model}

# separate by comma, each represents one input.
hrt_model_exec infer --model_file=${model} --input_file=test.jpg

# core id, 0 for any core, 1 for core 0, 2 for core 1.
hrt_model_exec perf --model_file=${model} -core_id 1 --profile_path='.'
```

## YOLOv5
```sh
docker pull flystarhe/yolov5:6.1-torch1.10-cuda11.3

# run
cd /workspace/docker/model_convert && \
IMAGE=flystarhe/yolov5:6.1-torch1.10-cuda11.3 && \
docker run --gpus all -d --network=host --ipc=host --name yolov5 -v "$(pwd)":/workspace ${IMAGE}

# attach
docker exec -it yolov5 bash

# src
git clone -q -b clip --depth 1 https://github.com/cd-lfi/yolov5.git
pip install onnx onnx-simplifier onnxruntime-gpu

# export onnx
weights=/workspace/models/yolov5m_relu_sigmoid_fast_nc_15.pt
python export.py --weights ${weights} --img 640 --opset 11 --include onnx
# /workspace/models/yolov5m_relu_sigmoid_fast_nc_15.onnx
```
