#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "cuda_utils.h"
#include "logging.h"
#include "preprocess.h"
#include "postprocess.h"


Logger gLogger;
using namespace nvinfer1;

const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;


// 反序列化引擎
void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "读取 " << engine_name << " 出错！" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    *runtime = createInferRuntime(gLogger);
    assert(*runtime);
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;

    std::cerr << "引擎反序列化成功" << std::endl;
}




// 解析命令行参数
bool parse_args_deserialize(int argc, char** argv, std::string& engine, std::string& img_dir, std::string& cuda_post_process) {
    if (argc == 5) {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
        cuda_post_process = std::string(argv[4]);
        return true;
    }
    return false;
}

// 准备缓冲区
void prepare_buffer(ICudaEngine* engine, float** input_buffer_device, float** output_buffer_device,
                    float** output_buffer_host, float** decode_ptr_host, float** decode_ptr_device) {
    assert(engine->getNbBindings() == 2);
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);

    std::cout << "inputIndex: " << inputIndex << std::endl;
    std::cout << "outputIndex: " << outputIndex << std::endl;
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    CUDA_CHECK(cudaMalloc((void**)input_buffer_device, kBatchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)output_buffer_device, kBatchSize * kOutputSize * sizeof(float)));
    *output_buffer_host = new float[kBatchSize * kOutputSize];

    std::cerr << "缓冲区准备成功" << std::endl;
}


// 推理
void infer(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output, int batchsize,
           float* decode_ptr_host, float* decode_ptr_device, int model_bboxes) {
    std::cout << "已经进入推理啦" << std::endl;
    auto t0 = std::chrono::system_clock::now();
    if (!context.enqueueV2(buffers, stream, nullptr)) {
        std::cerr << "Failed to enqueue the inference task" << std::endl;
        return;
    }

    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], kBatchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::system_clock::now();
    std::cout << "将推理任务添加到上下文耗时: " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << "μs\n";
    std::cout << "推理完成\n";
}

int main(int argc, char** argv) {
    cudaSetDevice(kGpuId);
    std::string engine_name = "";
    std::string img_dir;
    std::string cuda_post_process = "c";

    if (!parse_args_deserialize(argc, argv, engine_name, img_dir, cuda_post_process)) {
        std::cerr << "参数不正确！" << std::endl;
        std::cerr << "./yolov8 -d [.engine] [image folder] [c/g]  // 反序列化计划文件并运行推理" << std::endl;
        return -1;
    }

    std::cerr << "参数解析成功" << std::endl;

    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;

    deserialize_engine(engine_name, &runtime, &engine, &context);

    float* buffers[2];
    float* input_buffer_device = nullptr;
    float* output_buffer_device = nullptr;
    float* output_buffer_host = nullptr;
    float* decode_ptr_host = nullptr;
    float* decode_ptr_device = nullptr;


    prepare_buffer(engine, &input_buffer_device, &output_buffer_device, &output_buffer_host, &decode_ptr_host, &decode_ptr_device);

    cuda_preprocess_init(kMaxInputImageSize);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::cerr << "CUDA 流创建成功" << std::endl;

    buffers[0] = input_buffer_device;
    buffers[1] = output_buffer_device;

    std::vector<cv::String> filenames;
    cv::glob(img_dir, filenames);

    std::string output_file;

    for (const auto& filename : filenames) {
        auto T1 = std::chrono::system_clock::now();
        cv::Mat img = cv::imread(filename);
        if (img.empty()) {
            std::cerr << "无法读取图像: " << filename << std::endl;
            continue;
        }
        auto T2 = std::chrono::system_clock::now();

        // 定义仿射变换矩阵
        AffineMatrix s2d, d2s;

        // 使用 cuda_preprocess 进行预处理
        cuda_preprocess(img.data, img.cols, img.rows, input_buffer_device, INPUT_W, INPUT_H, stream, s2d, d2s);
        auto T3 = std::chrono::system_clock::now();
        
        infer(*context, stream, (void**)buffers, output_buffer_host, kBatchSize, decode_ptr_host, decode_ptr_device, kMaxNumOutputBbox);
        auto T4 = std::chrono::system_clock::now();

        // 后处理
        auto results = postProcess(output_buffer_host, kBatchSize, kMaxNumOutputBbox, NumCoords);
        auto T5 = std::chrono::system_clock::now();

        // 将 AffineMatrix 转换为 cv::Mat
        cv::Mat m2x3_d2s = cv::Mat(2, 3, CV_32F, d2s.value);

        // 在原图上绘制框
        draw_boxes(img, results[0], m2x3_d2s);

        // 保存结果图片
        std::string output_filename = "output_" + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(T1.time_since_epoch()).count()) + ".jpg";
        cv::imwrite(output_filename, img);

        auto T6 = std::chrono::system_clock::now();

        // 输出每部分的耗时（用微秒计时）
        std::cout << "读取图像耗时: " << std::chrono::duration_cast<std::chrono::microseconds>(T2 - T1).count() << "μs\n";
        std::cout << "预处理耗时: " << std::chrono::duration_cast<std::chrono::microseconds>(T3 - T2).count() << "μs\n";
        std::cout << "推理耗时: " << std::chrono::duration_cast<std::chrono::microseconds>(T4 - T3).count() << "μs\n";
        std::cout << "后处理耗时: " << std::chrono::duration_cast<std::chrono::microseconds>(T5 - T4).count() << "μs\n";
        std::cout << "绘制框和保存图片耗时: " << std::chrono::duration_cast<std::chrono::microseconds>(T6 - T5).count() << "μs\n";
        std::cout << "总耗时: " << std::chrono::duration_cast<std::chrono::microseconds>(T6 - T1).count() << "μs\n";
        std::cout << "----------------------------------------" << std::endl;

    }
    // 释放资源
    if (output_buffer_host) {
        delete[] output_buffer_host;
    }
    if (decode_ptr_host) {
        delete[] decode_ptr_host;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(input_buffer_device));
    CUDA_CHECK(cudaFree(output_buffer_device));
    if (decode_ptr_device) {
        CUDA_CHECK(cudaFree(decode_ptr_device));
    }

    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}