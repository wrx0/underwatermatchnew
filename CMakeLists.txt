cmake_minimum_required(VERSION 3.17)
project(underwatermatch)

set(CMAKE_CXX_STANDARD 11)
set(OpenCV_DIR "/home/wrx/opencv-4.4.0/buildnew")
find_package(OpenCV REQUIRED)
message(STATUS: "OpenCV libraries status:")
message(STATUS: "VERSION: ${OpenCV_VERSION}")
message(STATUS: "INCLUDE_PATH: ${OpenCV_INCLUDE_DIRS}")

add_executable(underwatermatch main.cpp)
target_link_libraries(underwatermatch ${OpenCV_LIBS})

add_executable(testopencv testopencv.cpp)
target_link_libraries(testopencv ${OpenCV_LIBS})