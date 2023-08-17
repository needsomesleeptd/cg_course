//
// Created by Андрей on 14.08.2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_COLOR_COLOR_H_
#define LAB_03_CG_COURSE_PROGRAMM_COLOR_COLOR_H_

struct ColorRGB
{
 public:
	ColorRGB() = default;
	ColorRGB(int R_,int G_,int B_);
	ColorRGB operator*(int value);
	ColorRGB operator+(const ColorRGB& color);

	int R;
	int G;
	int B;
};

#endif //LAB_03_CG_COURSE_PROGRAMM_COLOR_COLOR_H_
