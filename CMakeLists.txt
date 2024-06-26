cmake_minimum_required(VERSION 3.10)

project(yolov10)

# 引入C++17支持
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

# 设置CUDA编译器
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# 检查是否为嵌入式平台
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("embed_platform on")
  include_directories(/usr/local/cuda/targets/aarch64-linux/include)
  link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
  message("embed_platform off")
  # CUDA
  include_directories(/usr/local/cuda/include)
  link_directories(/usr/local/cuda/lib64)

  # TensorRT
  include_directories(/opt/TensorRT-8.6.1.6/include)
  link_directories(/opt/TensorRT-8.6.1.6/lib)
endif()

# 查找OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Source files
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)

# 调试代码，打印包含的源文件
message("Source files: ${SRCS}")

# 编译可执行文件
add_executable(yolov10 ${PROJECT_SOURCE_DIR}/yolov10_infer.cpp ${SRCS})
target_link_libraries(yolov10 nvinfer cudart  ${OpenCV_LIBS})
# 设置 CUDA 架构
set_property(TARGET yolov10  PROPERTY CUDA_ARCHITECTURES 89)


# 额外定义
add_definitions(-O2 -pthread)