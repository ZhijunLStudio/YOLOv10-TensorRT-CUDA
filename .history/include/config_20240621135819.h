#ifndef CONFIG_H
#define CONFIG_H

const int num_class = 80;
inline const int INPUT_W = 640;
inline const int INPUT_H = 640;
inline static int kGpuId = 0;
inline const char* kInputTensorName = "images";
inline const char* kOutputTensorName = "output0";
inline const int kMaxInputImageSize = 3000 * 3000;
const static float kConfThresh = 0.3f;
inline const int kBatchSize = 1;
inline static int kMaxNumOutputBbox = 300;
inline static int NumCoords = 6; // 每个预测框的维度数量
inline static float NormMeans[3] = { 0.f, 0.f, 0.f };
inline static float NormStds[3] = { 1.f, 1.f, 1.f };
inline static float NormScale = 255.f;
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

#endif // CONFIG_H

