cmake_minimum_required(VERSION 3.10)

project(Demos)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)



find_package(OpenCV)
find_package(OpenVINO)


include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/opt/intel/openvino_2023/runtime/include)

add_executable(main main5.cpp)
target_link_libraries(main PRIVATE ${OpenCV_LIBS}  openvino::runtime)