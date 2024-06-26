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
bool parse_args_deserialize(int argc, char** argv, std::string& engine, std::string& img_dir) {
    if (argc == 3) {
        engine = std::string(argv[1]);
        img_dir = std::string(argv[2]);
        return true;
    }
    return false;
}

// 准备缓冲区
void prepare_buffer(ICudaEngine* engine, float** input_buffer_device, float** output_buffer_device,
                    float** output_buffer_host) {
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
void infer(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output, int batchsize, int model_bboxes) {
    if (!context.enqueueV2(buffers, stream, nullptr)) {
        std::cerr << "Failed to enqueue the inference task" << std::endl;
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], kBatchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "推理完成\n";
}



int main(int argc, char** argv) {
    cudaSetDevice(kGpuId);
    std::string engine_name = "";
    std::string img_dir;

    if (!parse_args_deserialize(argc, argv, engine_name, img_dir)) {
        std::cerr << "参数不正确！" << std::endl;
        std::cerr << "./yolov8 [.engine] [image folder] // 反序列化计划文件并运行推理" << std::endl;
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


    prepare_buffer(engine, &input_buffer_device, &output_buffer_device, &output_buffer_host);

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
        AffineMat s2d, d2s;

        // 预处理
        cuda_preprocess(img.data, img.cols, img.rows, input_buffer_device, INPUT_W, INPUT_H, stream, s2d, d2s);
        auto T3 = std::chrono::system_clock::now();

        // 推理
        infer(*context, stream, (void**)buffers, output_buffer_host, kBatchSize, kMaxNumOutputBbox);
        auto T4 = std::chrono::system_clock::now();
        // 后处理
        auto results = postProcess(output_buffer_host, kBatchSize, kMaxNumOutputBbox, NumCoords);
        auto T5 = std::chrono::system_clock::now();
        // 将 AffineMat 转换为 cv::Mat
        float d2s_values[6] = { d2s.v0, d2s.v1, d2s.v2, d2s.v3, d2s.v4, d2s.v5 };
        cv::Mat m2x3_d2s = cv::Mat(2, 3, CV_32F, d2s_values);
        
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

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(input_buffer_device));
    CUDA_CHECK(cudaFree(output_buffer_device));


    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}