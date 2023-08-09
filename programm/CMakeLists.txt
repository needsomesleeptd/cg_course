cmake_minimum_required(VERSION 3.0.0)
project(dz2)

set(CMAKE_CXX_STANDARD 20)



set(Srcs "main.cpp" "math_primitives/vector/vector.cpp")
set(Hdrs  "math_primitives/vector/vector.h" "math_primitives/base_math_exception.h" "math_primitives/vector/vector_exceptions.h" math_primitives/base_math_primitive.h)

include_directories(math_primitives/vector)
include_directories(math_primitives)


add_executable(ray_tracer ${Srcs} ${Hdrs})

#Testing part
target_include_directories(ray_tracer_tests gtest_main gmock_main)

add_executable(ray_tracer_tests ${Srcs} ${Hdrs})