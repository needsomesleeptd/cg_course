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
	__host__ __device__   ImageAdapter();
	__host__ void DeviceMalloc(int width, int height);
	__host__ __device__  void HostMalloc(int width, int height);
	__host__  __device__ void setPixelColor(int x,int y,ColorRGB color);
	__host__  __device__ int getWidth();
	__host__  __device__ int getHeight();
	__host__  __device__ ImageAdapter getImage();
	__host__  __device__ ~ImageAdapter();
	ColorRGB* colorMatrix;
	int _width;
	int _height;
	bool _isHost = true;
};

#endif //LAB_03_CG_COURSE_PROGRAMM_IMAGE_ADAPTER_IMAGEADAPTER_H_
