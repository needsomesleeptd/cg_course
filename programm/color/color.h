//
// Created by Андрей on 14.08.2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_COLOR_COLOR_H_
#define LAB_03_CG_COURSE_PROGRAMM_COLOR_COLOR_H_

#include  <cuda_runtime.h>

struct ColorRGB
{
 public:
	ColorRGB() = default;
	__host__ __device__ ColorRGB(float R_, float G_, float B_);
	__host__ __device__ ColorRGB operator*(float value);
	__host__ __device__ ColorRGB operator+(const ColorRGB& color);
	__host__ __device__ ColorRGB operator*(const ColorRGB& color);
	__host__ __device__ void normalize();
	float R;
	float G;
	float B;
};

#endif //LAB_03_CG_COURSE_PROGRAMM_COLOR_COLOR_H_
