#ifndef CONFIG_H
#define CONFIG_H

const int num_class = 80;
inline const int INPUT_W = 640;
inline const int INPUT_H = 640;
inline static int kGpuId = 0;
inline const char* kInputTensorName = "images";
inline const char* kOutputTensorName = "output0";
inline const int kMaxInputImageSize = 3000 * 3000;
const static float kConfThresh = 0.5f;
inline const int kBatchSize = 1;
inline static int kMaxNumOutputBbox = 300;
inline static int NumCoords = 6; // 每个预测框的维度数量
#endif // CONFIG_H

