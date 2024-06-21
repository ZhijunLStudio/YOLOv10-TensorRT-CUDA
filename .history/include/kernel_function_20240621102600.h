// #pragma once
// #include"common_include.h"
// #include"types.h"


// // #define CHECK(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

// bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line);

// #define BLOCK_SIZE 8

// //note: resize rgb with padding
// void resizeDevice(const int& batch_size, float* src, int src_width, int src_height,
//     float* dst, int dstWidth, int dstHeight,
//     float paddingValue, AffineMat matrix);


// void bgr2rgbDevice(const int& batch_size, float* src, int srcWidth, int srcHeight,
//     float* dst, int dstWidth, int dstHeight);

// void normDevice(const int& batch_size, float* src, int srcWidth, int srcHeight,
//     float* dst, int dstWidth, int dstHeight);

// void hwc2chwDevice(const int& batch_size, float* src, int srcWidth, int srcHeight,
//     float* dst, int dstWidth, int dstHeight);



#pragma once
#include "common_include.h"
#include "types.h"

// #define CHECK(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

// bool DEPLOY_DECL __check_cuda_runtime(cudaError_t code, const char *op,
//                                       const char *file, int line);

#define BLOCK_SIZE 8

// note: resize rgb with padding
void DEPLOY_DECL resizeDevice(const int &batch_size, float *src, int src_width,
                              int src_height, float *dst, int dstWidth,
                              int dstHeight, float paddingValue,
                              utils::AffineMat matrix);

// overload:resize rgb with padding, but src's type is uin8
void DEPLOY_DECL resizeDevice(const int &batch_size, unsigned char *src,
                              int src_width, int src_height, float *dst,
                              int dstWidth, int dstHeight, float paddingValue,
                              utils::AffineMat matrix);

// overload: resize rgb/gray without padding
void DEPLOY_DECL resizeDevice(const int &batchSize, float *src, int srcWidth,
                              int srcHeight, float *dst, int dstWidth,
                              int dstHeight, utils::ColorMode mode,
                              utils::AffineMat matrix);

void DEPLOY_DECL bgr2rgbDevice(const int &batch_size, float *src, int srcWidth,
                               int srcHeight, float *dst, int dstWidth,
                               int dstHeight);

void DEPLOY_DECL normDevice(const int &batch_size, float *src, int srcWidth,
                            int srcHeight, float *dst, int dstWidth,
                            int dstHeight, utils::InitParameter norm_param);

void DEPLOY_DECL hwc2chwDevice(const int &batch_size, float *src, int srcWidth,
                               int srcHeight, float *dst, int dstWidth,
                               int dstHeight);

void DEPLOY_DECL decodeDevice(utils::InitParameter param, float *src,
                              int srcWidth, int srcHeight, int srcLength,
                              float *dst, int dstWidth, int dstHeight);

// nms fast
void DEPLOY_DECL nmsDeviceV1(utils::InitParameter param, float *src,
                             int srcWidth, int srcHeight, int srcArea);

// nms sort
void DEPLOY_DECL nmsDeviceV2(utils::InitParameter param, float *src,
                             int srcWidth, int srcHeight, int srcArea, int *idx,
                             float *conf);

void DEPLOY_DECL copyWithPaddingDevice(const int &batchSize, float *src,
                                       int srcWidth, int srcHeight, float *dst,
                                       int dstWidth, int dstHeight,
                                       float paddingValue, int padTop,
                                       int padLeft);
