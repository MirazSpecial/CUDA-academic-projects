cmake_minimum_required(VERSION 3.19)
project(Threads_2D CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(Threads_2D Threads_2D.cu)

set_target_properties(
        Threads_2D
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)