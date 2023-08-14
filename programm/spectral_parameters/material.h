//
// Created by Андрей on 14.08.2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_SPECTRAL_PARAMETERS_MATERIAL_H_
#define LAB_03_CG_COURSE_PROGRAMM_SPECTRAL_PARAMETERS_MATERIAL_H_

#include "color.h"
struct Material // TODO:: Might be better to make class with restricted access
{
	ColorRGB color;
	float k_a;
	float k_d;
	float k_s;
};

#endif //LAB_03_CG_COURSE_PROGRAMM_SPECTRAL_PARAMETERS_MATERIAL_H_
