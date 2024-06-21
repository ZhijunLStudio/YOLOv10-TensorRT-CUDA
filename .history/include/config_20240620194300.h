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
inline static float means[3] = { 0.f, 0.f, 0.f };
inline         float stds[3] = { 1.f, 1.f, 1.f };
#endif // CONFIG_H

