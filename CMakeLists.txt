cmake_minimum_required(VERSION 3.10)

project(AdaptiveKMeans)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Manually set the Eigen include directory
include_directories("/usr/local/include/eigen3")  # Adjust this path as necessary

# Include directories for your project
include_directories("${PROJECT_SOURCE_DIR}/include")

# Add library
add_library(AdaptiveKMeans STATIC src/AdaptiveKMeans.cpp)

# Add executable for testing
add_executable(testAdaptiveKMeans test/main.cpp)

# Link libraries
target_link_libraries(testAdaptiveKMeans AdaptiveKMeans)
