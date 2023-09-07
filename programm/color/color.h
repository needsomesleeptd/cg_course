//
// Created by Андрей on 14.08.2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_COLOR_COLOR_H_
#define LAB_03_CG_COURSE_PROGRAMM_COLOR_COLOR_H_

struct ColorRGB
{
 public:
	ColorRGB() = default;
	ColorRGB(float R_, float G_, float B_);
	ColorRGB operator*(float value);
	ColorRGB operator/(float value);
	ColorRGB operator+(const ColorRGB& color);
	ColorRGB operator*(const ColorRGB& color);
	void normalize();
	float R;
	float G;
	float B;
};

#endif //LAB_03_CG_COURSE_PROGRAMM_COLOR_COLOR_H_
