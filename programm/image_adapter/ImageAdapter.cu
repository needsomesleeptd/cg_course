//
// Created by Андрей on 14.08.2023.
//

#include "ImageAdapter.h"
#include <iostream>

__host__ __device__ void ImageAdapter::setPixelColor(int x, int y, ColorRGB color)
{
	colorMatrix[x * _width + y] = color;
}
__host__  __device__ int ImageAdapter::getWidth()
{
	return _width;
}
__host__  __device__ int ImageAdapter::getHeight()
{
	return _height;
}
__host__  __device__ ImageAdapter ImageAdapter::getImage()
{
	return *this;
}
__host__ __device__  ImageAdapter::ImageAdapter()
{
	_width = 1;
	_height = 1;
}

__host__  ImageAdapter::~ImageAdapter()
{
	if (_isHost)
		free(colorMatrix);
	else
		cudaFree(colorMatrix);
}
void ImageAdapter::DeviceMalloc(int width, int height)
{
	/*_width = width;
	_height = height;*/
	cpuErrorCheck(cudaMalloc((void**)&colorMatrix, _width * _height * sizeof(ColorRGB)));
	//_isHost = false;
}
void ImageAdapter::HostMalloc(int width, int height)
{
	_width = width;
	_height = height;
	colorMatrix = (ColorRGB*)malloc(_width * _height * sizeof(ColorRGB));
	_isHost = true;
}


