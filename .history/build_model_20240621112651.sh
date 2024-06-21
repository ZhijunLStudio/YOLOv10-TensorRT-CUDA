#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

conda activate torch 

# 定义模型文件名的变量
model_name="yolov10s" 

# 导出为onnx格式
yolo mode=export model=./checkpoints/${model_name}.pt format=onnx dynamic=False simplify=True opset=13 

# 导出为engine格式
/opt/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/bin/trtexec --onnx=./${model_name}.onnx \
  --saveEngine=./${model_name}_fp32.engine  --fp16  
  # --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:4x3x640x640