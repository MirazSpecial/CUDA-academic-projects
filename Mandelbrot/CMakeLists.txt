cmake_minimum_required(VERSION 3.17)
project(Mandelbrot CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(Mandelbrot Mandelbrot.cu)

set_target_properties(
        Mandelbrot
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)