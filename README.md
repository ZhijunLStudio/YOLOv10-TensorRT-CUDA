# YOLOv10-TensorRT-CUDA

CUDA预处理包括letterbox/BGR转RGB/归一化/hwc转chw等

由于yolov10采用的是nms-free，因此后处理仅需要对坐标作个映射即可

## 原始yolov10的pt推理结果与本仓库的int8推理结果对比

![](images/image1.jpg)

![](images/image2.jpg)

![](images/image3.jpg)

## 环境依赖
- TensorRT-8.6.1.6
- CUDA-12.1
- OpenCV-4.5.0
- ubuntu-20.04（windows亦可）
- pytorch-2.2.1
- ultralytics-8.2.38



## 第一步: pt转onnx

这一步按`yolo mode=export model=./checkpoints/${model_name}.pt format=onnx dynamic=False simplify=True opset=13`导出即可

## 第二步: onnx转trt

fp32/fp16可以使用trtexec直接进行导出，int8的话掉点太多，不建议使用该方法

```shell
/opt/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/bin/trtexec --onnx=./${model_name}.onnx \
  --saveEngine=./${model_name}_fp16.engine  --fp16  
```

如果需要转int8，使用build_model/onnx_to_trt.py脚本进行导出即可，如下示例

```shell
python ./build_model/onnx_to_trt.py \
-m best.onnx -d int8 --img-size 320 --batch-size 1 \
--calib-img-dir data_calib \
--verbose
```


## 第三步: 修改配置项

在`include/config`下有一些自定义选项，需要根据自己训练模型进行调整

```c++
const int num_class = 80;
static const int INPUT_W = 640;
static const int INPUT_H = 640;
static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };
```

## 第四步: 编译/运行项目

编译前记得根据自己的环境，配置下CMakeLists.txt

```shell
mkdir build
cd build
cmake ..
make
```

运行程序

默认推理为图片文件夹，如有其他需求更改`yolov10_infer.cpp`即可

```shell
./yolov10 you.engine  image_folder_path
```

注：本仓库代码为单batch推理，后续考虑加入多batch推理

python版的推理代码在`tools/infer_pt.py`

## 参考仓库
https://github.com/wang-xinyu/tensorrtx 

https://github.com/FeiYull/TensorRT-Alpha

https://github.com/meituan/YOLOv6
