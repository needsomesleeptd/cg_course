//
// Created by Андрей on 14.08.2023.
//

#include "ImageAdapter.h"

#include <memory>
void ImageAdapter::setPixelColor(int x, int y, ColorRGB color)
{
	image->setPixelColor(x, y, QColor(color.R, color.G, color.B));
}
int ImageAdapter::getWidth()
{
	return image->width();
}
int ImageAdapter::getHeight()
{
	return image->height();
}
std::shared_ptr<QImage> ImageAdapter::getImage()
{
	return image;
}
ImageAdapter::ImageAdapter()
{
	image = std::make_shared<QImage>();
}
ImageAdapter::ImageAdapter(int width, int height)
{
	image = std::make_shared<QImage>(width,height,QImage::Format_RGB32);
}
