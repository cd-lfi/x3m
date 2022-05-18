#!/bin/bash
set -e

# ai_toolchain_package/v2.0.4.tar.xz/ai_toolchain/horizon_model_convert_sample/04_detection/03_yolov5/mapper/01_check.sh

model_type=onnx
march=bernoulli2

onnx_model=${1:-/workspace/models/yolov5_lite/yolov5m_relu_sigmoid_fast_nc_15.onnx}
input_node=${2:-images}
input_shape=${3:-1x3x640x640}
output=${4:-log.checker}

hb_mapper checker --model-type ${model_type} \
                  --march ${march} \
                  --model ${onnx_model} \
                  --input-shape ${input_node} ${input_shape} \
                  --output ${output}
