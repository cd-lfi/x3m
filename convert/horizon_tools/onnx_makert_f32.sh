#!/bin/bash
set -e


onnx_model=${1:-yolov5m.onnx}
input_name=${2:-images}
input_shape=${3:-640x640}
input_type_rt=${4:-nv12}
core_num=${5:-1}


working_dir="tmp/makert_$(basename ${onnx_model} .onnx)"
file_prefix="out_${input_shape}_${input_type_rt}"
config_file="${working_dir}/config.yaml"


rm -rf ${working_dir} && mkdir -p ${working_dir}


cat > ${config_file} <<EOF
# https://developer.horizon.ai/api/v1/fileData/documents/ai_toolchain_develop/horizon_ai_toolchain_user_guide/chapter_3_model_conversion.html#model-conversion
model_parameters:
  onnx_model: '${onnx_model}'
  march: 'bernoulli2'
  working_dir: '${working_dir}'
  output_model_file_prefix: '${file_prefix}'
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
  norm_type: 'data_scale'
  mean_value: ''
  scale_value: '0.003921568627451'

calibration_parameters:
  cal_data_dir: './calibration_data_rgb_f32'
  calibration_type: 'default'
  max_percentile: '0.99999'
  per_channel: False

compiler_parameters:
  compile_mode: 'latency'
  debug: False
  core_num: ${core_num}
  optimize_level: 'O3'
EOF


hb_mapper makertbin --config ${config_file} --model-type onnx

