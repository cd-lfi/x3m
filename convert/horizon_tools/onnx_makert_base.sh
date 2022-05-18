#!/bin/bash
set -e


model_type='onnx'
config_file='./config.yaml'


onnx_model=${1:-yolov5m.onnx}
input_name=${2:-images}
input_shape=${3:-640x640}
input_type_rt=${4:-nv12}


file_prefix="${input_shape}_${input_type_rt}"
working_dir="tmp/make_rt_$(basename ${onnx_model} .onnx)"


rm -rf ${working_dir} && mkdir -p ${working_dir}


cat > ${config_file} <<EOF
# https://developer.horizon.ai/api/v1/fileData/documents/ai_toolchain_develop/horizon_ai_toolchain_user_guide/chapter_3_model_conversion.html#model-conversion
model_parameters:
  onnx_model: '${onnx_model}'
  march: 'bernoulli2'
  output_model_file_prefix: '${file_prefix}'
  working_dir: '${working_dir}'
  layer_out_dump: False
  log_level: 'debug'

input_parameters:
  input_name: '${input_name}'
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'
  input_type_rt: '${input_type_rt}'
  input_layout_rt: 'NHWC'
  input_shape: '1x3x${input_shape}'
  input_batch: 1
  norm_type: 'data_mean_and_scale'
  mean_value: '0.0'
  scale_value: '0.003921568627451'

calibration_parameters:
  cal_data_dir: './calibration_data_rgb_f32'
  preprocess_on: False
  calibration_type: 'default'

compiler_parameters:
  compile_mode: 'latency'
  debug: False
  core_num: 1
  optimize_level: 'O3'
EOF


hb_mapper makertbin --config ${config_file} --model-type ${model_type}
rm -rf ${config_file}
