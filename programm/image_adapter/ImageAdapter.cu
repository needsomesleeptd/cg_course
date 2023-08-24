//
// Created by Андрей on 14.08.2023.
//

#include "ImageAdapter.h"

__device__ void ImageAdapter::setPixelColor(int x, int y, ColorRGB color)
{
	colorMatrix[x * width + y] = color;
}
__device__ int ImageAdapter::getWidth()
{
	return width;
}
__device__ int ImageAdapter::getHeight()
{
	return height;
}
__device__ ImageAdapter ImageAdapter::getImage()
{
	return *this;
}
__device__ ImageAdapter::ImageAdapter()
{
	width = 1;
	height = 1;
	gpuErrorCheck(cudaMalloc((void**)&colorMatrix, 1 * sizeof(ColorRGB)));
}
__device__ ImageAdapter::ImageAdapter(int width, int height)
{
	width = 1;
	height = 1;
	gpuErrorCheck(cudaMalloc((void**)&colorMatrix, width * height * sizeof(ColorRGB)));
}
__device__ ImageAdapter::~ImageAdapter()
{
	gpuErrorCheck(cudaFree(colorMatrix));
}
