//
// Created by Андрей on 14.08.2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_IMAGE_ADAPTER_IMAGEADAPTER_H_
#define LAB_03_CG_COURSE_PROGRAMM_IMAGE_ADAPTER_IMAGEADAPTER_H_
#include <QImage>
#include "memory"
#include "color.h"
#include <QColor>
class ImageAdapter
{
 public:
	ImageAdapter();
	explicit ImageAdapter(int width,int height);
	void setPixelColor(int x,int y,ColorRGB color);
	int getWidth();
	int getHeight();
	std::shared_ptr<QImage> getImage();
 private:
	std::shared_ptr<QImage> image;
};

#endif //LAB_03_CG_COURSE_PROGRAMM_IMAGE_ADAPTER_IMAGEADAPTER_H_
