//
// Created by Андрей on 14.08.2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_SPECTRAL_PARAMETERS_MATERIAL_H_
#define LAB_03_CG_COURSE_PROGRAMM_SPECTRAL_PARAMETERS_MATERIAL_H_

#include "color.h"
struct Material // TODO:: Might be better to make class with restricted access
{
	Material() = default;
	explicit Material(float k_a_src, float k_d_src, float k_s_src, ColorRGB color);
	float _k_a;
	float _k_d;
	float _k_s;
	ColorRGB _color;
};

#endif //LAB_03_CG_COURSE_PROGRAMM_SPECTRAL_PARAMETERS_MATERIAL_H_
