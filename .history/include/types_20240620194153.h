#pragma once
#include "config.h"

// 定义检测结果结构体
struct Detection {
    float left;
    float top;
    float right;
    float bottom;
    float confidence;
    int class_id;
};

struct AffineMatrix {
    float value[6];
};

    struct AffineMat
    {
        float v0, v1, v2;
        float v3, v4, v5;
    };
