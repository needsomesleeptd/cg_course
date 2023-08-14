//
// Created by Андрей on 14.08.2023.
//

#include "ImageAdapter.h"
void ImageAdapter::setPixelColor(int x, int y,ColorRGB color)
{
	image->setPixelColor(x,y,QColor(color.R,color.G,color.B));
}
int ImageAdapter::getWidth()
{
	image->width();
}
int ImageAdapter::getHeight()
{
	image->height();
}
