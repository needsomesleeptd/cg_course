//
// Created by Андрей on 14.08.2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_IMAGE_ADAPTER_IMAGEADAPTER_H_
#define LAB_03_CG_COURSE_PROGRAMM_IMAGE_ADAPTER_IMAGEADAPTER_H_
#include <QImage>
#include "memory"
#include "color.h"

class ImageAdapter
{
 public:
	void setPixelColor(int x,int y,ColorRGB color);
	int getWidth();
	int getHeight();
 private:
	std::shared_ptr<QImage> image;
};

#endif //LAB_03_CG_COURSE_PROGRAMM_IMAGE_ADAPTER_IMAGEADAPTER_H_
