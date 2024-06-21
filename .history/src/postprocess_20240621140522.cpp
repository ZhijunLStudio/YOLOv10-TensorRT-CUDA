#include "postprocess.h"


// 后处理函数
std::vector<std::vector<Detection>> postProcess(float* inference_output, int batch_size, int num_boxes, int num_coords) {
    std::vector<std::vector<Detection>> results(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < num_boxes; ++i) {
            float* box = inference_output + b * num_boxes * num_coords + i * num_coords;
            float left = box[0];
            float top = box[1];
            float right = box[2];
            float bottom = box[3];
            float confidence = box[4];
            int class_id = static_cast<int>(box[5]);

            if (confidence > kConfThresh) { // 过滤低置信度的框
                Detection det = {left, top, right, bottom, confidence, class_id};
                results[b].push_back(det);
            }
        }
    
    }

    return results;
}


void draw_boxes(cv::Mat& img, const std::vector<Detection>& detections, const cv::Mat& inverse_transform) {
    for (const auto& det : detections) {
        // 映射回原图坐标
        cv::Point2f points[4] = {
            cv::Point2f(det.left, det.top),
            cv::Point2f(det.right, det.top),
            cv::Point2f(det.right, det.bottom),
            cv::Point2f(det.left, det.bottom)
        };

        for (auto& point : points) {
            cv::Mat pt_mat = (cv::Mat_<float>(3, 1) << point.x, point.y, 1.0);
            cv::Mat original_pt_mat = inverse_transform * pt_mat;
            point.x = original_pt_mat.at<float>(0, 0);
            point.y = original_pt_mat.at<float>(1, 0);
        }

        cv::Rect rect(points[0], points[2]);
        cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2);

        // 获取类别名称
        std::string class_name = class_names[det.class_id];
        std::string label = class_name + ": " + std::to_string(det.confidence);

        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::rectangle(img, cv::Rect(cv::Point(rect.x, rect.y - labelSize.height),
                                    cv::Size(labelSize.width, labelSize.height + baseLine)),
                      cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(img, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}


// void draw_boxes(cv::Mat& img, const std::vector<Detection>& detections, const cv::Mat& inverse_transform) {
//     auto T1 = std::chrono::system_clock::now();

//     for (const auto& det : detections) {
//         auto T2 = std::chrono::system_clock::now();
        
//         // 映射回原图坐标
//         cv::Point2f points[4] = {
//             cv::Point2f(det.left, det.top),
//             cv::Point2f(det.right, det.top),
//             cv::Point2f(det.right, det.bottom),
//             cv::Point2f(det.left, det.bottom)
//         };

//         for (auto& point : points) {
//             cv::Mat pt_mat = (cv::Mat_<float>(3, 1) << point.x, point.y, 1.0);
//             cv::Mat original_pt_mat = inverse_transform * pt_mat;
//             point.x = original_pt_mat.at<float>(0, 0);
//             point.y = original_pt_mat.at<float>(1, 0);
//         }
        
//         auto T3 = std::chrono::system_clock::now();
//         std::cout << "映射回原图坐标耗时: " << std::chrono::duration_cast<std::chrono::microseconds>(T3 - T2).count() << "μs\n";

//         auto T4 = std::chrono::system_clock::now();
        
//         cv::Rect rect(points[0], points[2]);
//         cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2);

//         std::string label = std::to_string(det.class_id) + ": " + std::to_string(det.confidence);
//         int baseLine;
//         cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
//         cv::rectangle(img, cv::Rect(cv::Point(rect.x, rect.y - labelSize.height),
//                                     cv::Size(labelSize.width, labelSize.height + baseLine)),
//                       cv::Scalar(0, 255, 0), cv::FILLED);
//         cv::putText(img, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        
//         auto T5 = std::chrono::system_clock::now();
//         std::cout << "绘制矩形和标签耗时: " << std::chrono::duration_cast<std::chrono::microseconds>(T5 - T4).count() << "μs\n";
//     }

//     auto T6 = std::chrono::system_clock::now();
//     std::cout << "总耗时: " << std::chrono::duration_cast<std::chrono::microseconds>(T6 - T1).count() << "μs\n";
// }