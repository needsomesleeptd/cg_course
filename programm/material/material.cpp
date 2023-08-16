//
// Created by Андрей on 14.08.2023.
//

#include "material.h"
Material::Material(float k_a_src, float k_d_src, float k_s_src, ColorRGB color)
{
	_k_a = k_a_src;
	_k_d = k_d_src;
	_k_s = k_s_src;
	_color = color;

}
