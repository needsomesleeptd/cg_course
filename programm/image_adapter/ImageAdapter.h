//
// Created by Андрей on 14.08.2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_IMAGE_ADAPTER_IMAGEADAPTER_H_
#define LAB_03_CG_COURSE_PROGRAMM_IMAGE_ADAPTER_IMAGEADAPTER_H_
#include <vector>
#include "cudaUtils.h"
#include "../color/color.h"
class ImageAdapter
{
 public:
	__device__ ImageAdapter();
	__device__ explicit ImageAdapter(int width,int height);
	__device__ void setPixelColor(int x,int y,ColorRGB color);
	__device__ int getWidth();
	__device__ int getHeight();
	__device__ ImageAdapter getImage();
	__device__ ~ImageAdapter();
 private:
	ColorRGB* colorMatrix;
	int width;
	int height;
};

#endif //LAB_03_CG_COURSE_PROGRAMM_IMAGE_ADAPTER_IMAGEADAPTER_H_
