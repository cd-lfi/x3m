## docker
```sh
docker pull openexplorer/ai_toolchain_centos_7_xj3:v2.0.4

# run
cd /workspace/docker/model_convert && \
IMAGE=openexplorer/ai_toolchain_centos_7_xj3:v2.0.4 && \
docker run -it --rm --network=host --ipc=host -v "$(pwd)":/workspace ${IMAGE} bash

# src
src_path=/workspace/x3m

# todo
cd /workspace && \
rm -rf ${src_path} && mkdir -p ${src_path} && cd ${src_path} && \
git clone -q --depth 1 https://github.com/cd-lfi/x3m.git .

# yolov5: lite2
model_type=onnx
march=bernoulli2
onnx_model=/workspace/models/yolov5_lite/yolov5m_relu_sigmoid_fast_nc_15.onnx
input_node=images
input_shape=1x3x640x640
output=log.checker
hb_mapper checker --model-type ${model_type} \
                  --march ${march} \
                  --model ${onnx_model} \
                  --input-shape ${input_node} ${input_shape} \
                  --output ${output}

# horizon.ai: model_zoo/mapper/detection/yolov5_onnx_optimized
model_type=onnx
march=bernoulli2
onnx_model=/workspace/models/yolov5_onnx_optimized/YOLOv5l.onnx
input_node=data
input_shape=1x3x672x672
output=log.checker
hb_mapper checker --model-type ${model_type} \
                  --march ${march} \
                  --model ${onnx_model} \
                  --input-shape ${input_node} ${input_shape} \
                  --output ${output}

# yolov5: lite2
hb_mapper makertbin --config ${config_file}  \
                    --model-type  ${model_type}
```

## YOLOv5 - lite
```sh
docker pull flystarhe/yolov5:6.1-torch1.10-cuda11.3

# run
cd /workspace/docker/model_convert && \
IMAGE=flystarhe/yolov5:6.1-torch1.10-cuda11.3 && \
docker run --gpus all -d --network=host --ipc=host --name yolov5 -v "$(pwd)":/workspace ${IMAGE}

# attach
docker exec -it yolov5 bash

# src
src_path=/workspace/yolov5

# prepare
cd /workspace && \
rm -rf ${src_path} && mkdir -p ${src_path} && cd ${src_path} && \
git clone -q -b lite2 --depth 1 https://github.com/cd-lfi/yolov5.git . && \
pip install onnx onnx-simplifier onnxruntime-gpu

# export onnx
cd ${src_path} && \
weights=/workspace/models/yolov5_lite/yolov5m_relu_sigmoid_fast_nc_15.pt && \
python export.py --weights ${weights} --img 640 --opset 11 --include onnx --train
# /workspace/models/yolov5_lite/yolov5m_relu_sigmoid_fast_nc_15.onnx

# deeplearning results
aws s3 cp s3://lfi-algo-data-us-west-2/runs/deeplearning_20220517_082001 deeplearning_20220517_082001 --no-progress --recursive
# last.pt
aws s3 cp s3://lfi-algo-data-us-west-2/runs/deeplearning_20220517_082001/fiftyone_coco_nc_15/yolov5m_relu_sigmoid_fast/weights/last.pt \
    yolov5_lite/yolov5m_relu_sigmoid_fast_nc_15.pt
```

`Detect()`
```sh
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # `reshape` on cpu / use `split`
            x[i] = [xi.permute(0, 2, 3, 1).contiguous() for xi in torch.split(x[i], self.no, dim=1)][0]
```


```python
import wandb
run = wandb.init()
artifact = run.use_artifact('flystarhe/fiftyone_coco_nc_15/run_37xbidin_model:v29', type='model')
artifact_dir = artifact.download()
```
