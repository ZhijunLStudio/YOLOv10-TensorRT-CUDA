#pragma once

#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "types.h"
#include <map>


void cuda_preprocess_init(int max_image_size);

void cuda_preprocess_destroy();

void cuda_preprocess(uint8_t *src, int src_width, int src_height, float *dst, int dst_width, int dst_height, cudaStream_t stream, AffineMat& s2d, AffineMatrix& d2s);



