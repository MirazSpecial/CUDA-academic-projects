cmake_minimum_required(VERSION 3.19)
project(strumienie_i_2D CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(strumienie_i_2D main.cu)

set_target_properties(
        strumienie_i_2D
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)