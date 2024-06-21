#include "preprocess.h"
#include "cuda_utils.h"
#include "kernel_function.h"


static uint8_t *img_buffer_host = nullptr;
static uint8_t *img_buffer_device = nullptr;

// 新的全局变量
static float *m_input_src_device = nullptr;
static float *m_input_resize_device = nullptr;
static float *m_input_rgb_device = nullptr;
static float *m_input_norm_device = nullptr;
static float *m_input_hwc_device = nullptr;
static AffineMat m_dst2src;



__global__ void
warpaffine_kernel(uint8_t *src, int src_line_size, int src_width, int src_height, float *dst, int dst_width,
                  int dst_height, uint8_t const_value_st, AffineMatrix d2s, int edge) {
    // 定义一个全局函数，用于执行仿射变换
    // 参数：
    //   src: 源图像数据指针
    //   src_line_size: 源图像每行字节数
    //   src_width: 源图像宽度
    //   src_height: 源图像高度
    //   dst: 目标图像数据指针
    //   dst_width: 目标图像宽度
    //   dst_height: 目标图像高度
    //   const_value_st: 用于填充超出范围像素的常数值
    //   d2s: 从目标坐标到源坐标的仿射变换矩阵
    //   edge: 线程执行的范围

    // 计算当前线程的索引
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    // 如果当前线程索引超出范围，则退出函数
    if (position >= edge) return;
    // 从仿射变换矩阵中获取变换参数
    float m_x1 = d2s.value[0];
    float m_y1 = d2s.value[1];
    float m_z1 = d2s.value[2];
    float m_x2 = d2s.value[3];
    float m_y2 = d2s.value[4];
    float m_z2 = d2s.value[5];
    // 计算目标图像中的坐标
    int dx = position % dst_width;
    int dy = position / dst_width;
    // 使用仿射变换矩阵将目标坐标映射到源坐标
    float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
    float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;
    // 定义三个变量，用于存储源图像中对应像素的 RGB 值
    float c0, c1, c2;

    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
        // out of range
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    }
    // 如果映射后的源坐标在源图像范围内，则进行双线性插值 
    else {
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;
        // 定义一个常量数组，用于填充超出范围的像素
        uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
        // 计算双线性插值的权重
        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        // 定义四个指针，指向源图像中对应像素的地址
        uint8_t *v1 = const_value;
        uint8_t *v2 = const_value;
        uint8_t *v3 = const_value;
        uint8_t *v4 = const_value;

        // 根据源坐标计算四个指针的地址
        if (y_low >= 0) {
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3;

            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }

        if (y_high < src_height) {
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;

            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }
        // 使用双线性插值计算源图像中对应像素的 RGB 值
        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }

    // bgr to rgb
    float t = c2;
    c2 = c0;
    c0 = t;

    // normalization
    c0 = c0 / 255.0f;
    c1 = c1 / 255.0f;
    c2 = c2 / 255.0f;

    // rgbrgbrgb to rrrgggbbb
    int area = dst_width * dst_height;
    float *pdst_c0 = dst + dy * dst_width + dx;
    float *pdst_c1 = pdst_c0 + area;
    float *pdst_c2 = pdst_c1 + area;
    // 将 RGB 值存储到目标图像中
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}


// void cuda_preprocess(uint8_t *src, int src_width, int src_height, float *dst, int dst_width, int dst_height,
//                      cudaStream_t stream, AffineMatrix& s2d, AffineMatrix& d2s) {
//     // 计算源图像的大小
//     int img_size = src_width * src_height * 3;
//     // 将源图像数据复制到主机上的 pinned 内存中
//     memcpy(img_buffer_host, src, img_size);
//     // 将源图像数据从主机上的 pinned 内存复制到设备内存中
//     CUDA_CHECK(cudaMemcpyAsync(img_buffer_device, img_buffer_host, img_size, cudaMemcpyHostToDevice, stream));

//     // 计算缩放比例
//     float scale = std::min(dst_height / (float) src_height, dst_width / (float) src_width);

//     // 计算从源坐标到目标坐标的仿射变换矩阵
//     s2d.value[0] = scale;
//     s2d.value[1] = 0;
//     s2d.value[2] = -scale * src_width * 0.5 + dst_width * 0.5;
//     s2d.value[3] = 0;
//     s2d.value[4] = scale;
//     s2d.value[5] = -scale * src_height * 0.5 + dst_height * 0.5;
//     // 使用 OpenCV 函数计算从目标坐标到源坐标的仿射变换矩阵
//     cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
//     cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
//     cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);
//     // 将计算出的仿射变换矩阵复制到 `d2s` 中
//     memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));
//     // 计算线程块和线程数
//     int jobs = dst_height * dst_width;
//     int threads = 256;
//     int blocks = ceil(jobs / (float) threads);
//     // 调用 `warpaffine_kernel` 函数执行仿射变换
//     warpaffine_kernel<<<blocks, threads, 0, stream>>>(
//             img_buffer_device, src_width * 3, src_width,
//             src_height, dst, dst_width,
//             dst_height, 128, d2s, jobs);
// }

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
    cv::Mat dst2src = cv::Mat::zeros(2, 3, CV_32FC13);
    std::cout << src2dst.at<float>(0, 1) << "111" << src2dst.at<float>(1, 0)
                << std::endl;
    cv::invertAffineTransform(src2dst, dst2src);

    m_dst2src.v0 = dst2src.ptr<float>(0)[0];
    m_dst2src.v1 = dst2src.ptr<float>(0)[1];
    m_dst2src.v2 = dst2src.ptr<float>(0)[2];
    m_dst2src.v3 = dst2src.ptr<float>(1)[0];
    m_dst2src.v4 = dst2src.ptr<float>(1)[1];
    m_dst2src.v5 = dst2src.ptr<float>(1)[2];

    // 执行预处理
    resizeDevice(1, m_input_src_device, src_width, src_height, dst, dst_width, dst_height, 114, m_dst2src);
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
    CUDA_CHECK(cudaFreeHost(img_buffer_host));

    CUDA_CHECK(cudaFree(m_input_src_device));
    CUDA_CHECK(cudaFree(m_input_resize_device));
    CUDA_CHECK(cudaFree(m_input_rgb_device));
    CUDA_CHECK(cudaFree(m_input_norm_device));
    CUDA_CHECK(cudaFree(m_input_hwc_device));
}