#include "preprocess.h"
#include "cuda_utils.h"
#include "kernel_function.h"


static unsigned char *img_buffer_host = nullptr;
static unsigned char *img_buffer_device = nullptr;

// 新的全局变量
static float *m_input_src_device = nullptr;
static float *m_input_resize_device = nullptr;
static float *m_input_rgb_device = nullptr;
static float *m_input_norm_device = nullptr;
static float *m_input_hwc_device = nullptr;
static AffineMat m_dst2src;


// 打印 AffineMat 的值
void printAffineMat(const AffineMat& mat) {
    std::cout << "AffineMat values:" << std::endl;
    std::cout << "v0: " << mat.v0 << std::endl;
    std::cout << "v1: " << mat.v1 << std::endl;
    std::cout << "v2: " << mat.v2 << std::endl;
    std::cout << "v3: " << mat.v3 << std::endl;
    std::cout << "v4: " << mat.v4 << std::endl;
    std::cout << "v5: " << mat.v5 << std::endl;
}



void cuda_preprocess(uint8_t *src, int src_width, int src_height, float *dst, int dst_width, int dst_height,
                     cudaStream_t stream, AffineMat& s2d, AffineMat& d2s) {
    // 计算源图像的大小
    int img_size = src_width * src_height * 3;
    // 将源图像数据复制到主机上的 pinned 内存中
    memcpy(img_buffer_host, src, img_size);
    // 将源图像数据从主机上的 pinned 内存复制到设备内存中
    CUDA_CHECK(cudaMemcpyAsync(img_buffer_device, img_buffer_host, img_size, cudaMemcpyHostToDevice, stream));

    // 计算缩放比例
    // float scale = std::min(dst_height / (float) src_height, dst_width / (float) src_width);

    // // 计算从源坐标到目标坐标的仿射变换矩阵
    // s2d.v0 = scale;
    // s2d.v1 = 0;
    // s2d.v2 = -scale * src_width * 0.5 + dst_width * 0.5;
    // s2d.v3 = 0;
    // s2d.v4 = scale;
    // s2d.v5 = -scale * src_height * 0.5 + dst_height * 0.5;
    
    // // 使用 OpenCV 函数计算从目标坐标到源坐标的仿射变换矩阵
    // float s2d_values[6] = { s2d.v0, s2d.v1, s2d.v2, s2d.v3, s2d.v4, s2d.v5 };
    // cv::Mat m2x3_s2d(2, 3, CV_32F, s2d_values);
    // float d2s_values[6] = { 0 };
    // cv::Mat m2x3_d2s(2, 3, CV_32F, d2s_values);
    // cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

    // // 将计算出的仿射变换矩阵复制到 `d2s` 中
    // memcpy(&d2s, m2x3_d2s.ptr<float>(0), sizeof(AffineMat));

    // // 设置仿射变换矩阵
    // m_dst2src.v0 = d2s.v0;
    // m_dst2src.v1 = d2s.v1;
    // m_dst2src.v2 = d2s.v2;
    // m_dst2src.v3 = d2s.v3;
    // m_dst2src.v4 = d2s.v4;
    // m_dst2src.v5 = d2s.v5;

    float a = float(dst_height) / src_height;
    float b = float(dst_width) / src_width;
    float scale = a < b ? a : b;
    cv::Mat src2dst =
        (cv::Mat_<float>(2, 3) << scale, 0.f,
        (-scale * src_width + dst_width + scale - 1) * 0.5, 0.f, scale,
        (-scale * src_height + dst_height + scale - 1) * 0.5);
    cv::Mat dst2src = cv::Mat::zeros(2, 3, CV_32FC1);
    std::cout << src2dst.at<float>(0, 1) << "111" << src2dst.at<float>(1, 0)
                << std::endl;
    cv::invertAffineTransform(src2dst, dst2src);

    m_dst2src.v0 = dst2src.ptr<float>(0)[0];
    m_dst2src.v1 = dst2src.ptr<float>(0)[1];
    m_dst2src.v2 = dst2src.ptr<float>(0)[2];
    m_dst2src.v3 = dst2src.ptr<float>(1)[0];
    m_dst2src.v4 = dst2src.ptr<float>(1)[1];
    m_dst2src.v5 = dst2src.ptr<float>(1)[2];

    // 打印 m_dst2src 的值
    printAffineMat(m_dst2src);

    // 执行预处理
    resizeDevice(1, img_buffer_device, src_width, src_height, dst, dst_width, dst_height, 114, m_dst2src);
    // bgr2rgbDevice(1, m_input_resize_device, dst_width, dst_height, m_input_rgb_device, dst_width, dst_height);
    // normDevice(1, m_input_rgb_device, dst_width, dst_height, m_input_norm_device, dst_width, dst_height);
    // hwc2chwDevice(1, m_input_norm_device, dst_width, dst_height, dst, dst_width, dst_height);
}

// void cuda_preprocess(uint8_t *src, int src_width, int src_height, float *dst, int dst_width, int dst_height,
//                      cudaStream_t stream, AffineMat& s2d, AffineMat& d2s) {
//     // 计算源图像的大小
//     int img_size = src_width * src_height * 3;
//     // 将源图像数据复制到主机上的 pinned 内存中
//     memcpy(img_buffer_host, src, img_size);
//     // 将源图像数据从主机上的 pinned 内存复制到设备内存中
//     CUDA_CHECK(cudaMemcpyAsync(img_buffer_device, img_buffer_host, img_size, cudaMemcpyHostToDevice, stream));

//     // 计算缩放比例
//     float scale = std::min(dst_height / (float) src_height, dst_width / (float) src_width);

//     // 计算从源坐标到目标坐标的仿射变换矩阵
//     s2d.v0 = scale;
//     s2d.v1 = 0;
//     s2d.v2 = -scale * src_width * 0.5 + dst_width * 0.5;
//     s2d.v3 = 0;
//     s2d.v4 = scale;
//     s2d.v5 = -scale * src_height * 0.5 + dst_height * 0.5;
    
//     // 使用 OpenCV 函数计算从目标坐标到源坐标的仿射变换矩阵
//     float s2d_values[6] = { s2d.v0, s2d.v1, s2d.v2, s2d.v3, s2d.v4, s2d.v5 };
//     cv::Mat m2x3_s2d(2, 3, CV_32F, s2d_values);
//     float d2s_values[6] = { 0 };
//     cv::Mat m2x3_d2s(2, 3, CV_32F, d2s_values);
//     cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

//     // 将计算出的仿射变换矩阵复制到 `d2s` 中
//     memcpy(&d2s, m2x3_d2s.ptr<float>(0), sizeof(AffineMat));

//     // 设置仿射变换矩阵
//     m_dst2src.v0 = d2s.v0;
//     m_dst2src.v1 = d2s.v1;
//     m_dst2src.v2 = d2s.v2;
//     m_dst2src.v3 = d2s.v3;
//     m_dst2src.v4 = d2s.v4;
//     m_dst2src.v5 = d2s.v5;

//     // 执行预处理
//     resizeDevice(1, m_input_src_device, src_width, src_height, m_input_resize_device, dst_width, dst_height, 114, m_dst2src);
//     bgr2rgbDevice(1, m_input_resize_device, dst_width, dst_height, m_input_rgb_device, dst_width, dst_height);
//     normDevice(1, m_input_rgb_device, dst_width, dst_height, m_input_norm_device, dst_width, dst_height);
//     hwc2chwDevice(1, m_input_norm_device, dst_width, dst_height, dst, dst_width, dst_height);
// }


void cuda_preprocess_init(int max_image_size) {
    CUDA_CHECK(cudaMallocHost((void **) &img_buffer_host, max_image_size * 3));
    CUDA_CHECK(cudaMalloc((void **) &img_buffer_device, max_image_size * 3));

    CUDA_CHECK(cudaMalloc((void **) &m_input_src_device, max_image_size * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &m_input_resize_device, max_image_size * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &m_input_rgb_device, max_image_size * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &m_input_norm_device, max_image_size * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &m_input_hwc_device, max_image_size * 3 * sizeof(float)));
}

void cuda_preprocess_destroy() {
    
    CUDA_CHECK(cudaFree(img_buffer_device));


    CUDA_CHECK(cudaFree(m_input_src_device));
    CUDA_CHECK(cudaFree(m_input_resize_device));
    CUDA_CHECK(cudaFree(m_input_rgb_device));
    CUDA_CHECK(cudaFree(m_input_norm_device));
    CUDA_CHECK(cudaFree(m_input_hwc_device));
}