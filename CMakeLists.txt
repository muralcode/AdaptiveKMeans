cmake_minimum_required(VERSION 3.10)

project(AdaptiveKMeans)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set path to Eigen headers
set(EIGEN3_INCLUDE_DIR "C:/eigen-3.4.0")

# Include directories
include_directories(${EIGEN3_INCLUDE_DIR} include)

# Add library
add_library(AdaptiveKMeans STATIC src/AdaptiveKMeans.cpp)

# Add executable for testing
add_executable(testAdaptiveKMeans test/main.cpp)

# Link libraries
target_link_libraries(testAdaptiveKMeans AdaptiveKMeans)
