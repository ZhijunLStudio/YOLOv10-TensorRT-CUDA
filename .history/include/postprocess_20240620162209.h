#pragma once

#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "types.h"

std::vector<std::vector<Detection>> postProcess(float* inference_output, int batch_size, int num_boxes, int num_coords);

void draw_boxes(cv::Mat& img, const std::vector<Detection>& detections, const cv::Mat& inverse_transform)