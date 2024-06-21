#pragma once
#include"types.h"


#define BLOCK_SIZE 8

//note: resize rgb with padding
void resizeDevice(const int &batchSize, unsigned char *src, int srcWidth,
    int srcHeight, float *dst, int dstWidth, int dstHeight,
    float paddingValue, AffineMat matrix);


void bgr2rgbDevice(const int& batch_size, float* src, int srcWidth, int srcHeight,
    float* dst, int dstWidth, int dstHeight);

void normDevice(const int& batch_size, float* src, int srcWidth, int srcHeight,
    float* dst, int dstWidth, int dstHeight);

void hwc2chwDevice(const int& batch_size, float* src, int srcWidth, int srcHeight,
    float* dst, int dstWidth, int dstHeight);

