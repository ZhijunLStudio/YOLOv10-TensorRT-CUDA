#pragma once
#include"common_include.h"


#define CHECK(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line);

#define BLOCK_SIZE 8

//note: resize rgb with padding
void resizeDevice(const int& batch_size, float* src, int src_width, int src_height,
    float* dst, int dstWidth, int dstHeight,
    float paddingValue, utils::AffineMat matrix);


void bgr2rgbDevice(const int& batch_size, float* src, int srcWidth, int srcHeight,
    float* dst, int dstWidth, int dstHeight);

void normDevice(const int& batch_size, float* src, int srcWidth, int srcHeight,
    float* dst, int dstWidth, int dstHeight, , float* scale);

void hwc2chwDevice(const int& batch_size, float* src, int srcWidth, int srcHeight,
    float* dst, int dstWidth, int dstHeight);

