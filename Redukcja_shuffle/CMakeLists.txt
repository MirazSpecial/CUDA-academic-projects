cmake_minimum_required(VERSION 3.19)
project(Redukcja_shuffle CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(Redukcja_shuffle Redukcja_shuffle.cu)

set_target_properties(
        Redukcja_shuffle
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)