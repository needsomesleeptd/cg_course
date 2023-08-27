//
// Created by Андрей on 14.08.2023.
//

#include "ImageAdapter.h"

__host__ __device__ void ImageAdapter::setPixelColor(int x, int y, ColorRGB color)
{
	colorMatrix[x * width + y] = color;
}
__host__  __device__ int ImageAdapter::getWidth()
{
	return width;
}
__host__  __device__ int ImageAdapter::getHeight()
{
	return height;
}
__host__  __device__ ImageAdapter ImageAdapter::getImage()
{
	return *this;
}
__host__   ImageAdapter::ImageAdapter()
{
	width = 1;
	height = 1;
	cpuErrorCheck(cudaMalloc((void**)&colorMatrix, 1 * sizeof(ColorRGB)));
}
__host__  ImageAdapter::ImageAdapter(int width, int height)
{
	width = 1;
	height = 1;
	cpuErrorCheck(cudaMalloc((void**)&colorMatrix, width * height * sizeof(ColorRGB)));
}
__host__  ImageAdapter::~ImageAdapter()
{
	cpuErrorCheck(cudaFree(colorMatrix));
}
