cmake_minimum_required(VERSION 3.19)
project(Redukcja CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(Redukcja Redukcja.cu)

set_target_properties(
        Redukcja
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)