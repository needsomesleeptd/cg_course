//
// Created by Андрей on 14.08.2023.
//

#include "color.h"
ColorRGB::ColorRGB(float R_, float G_, float B_) : R(R_), G(G_), B(B_)
{

}
ColorRGB ColorRGB::operator*(float value)
{
	float changedR = R * value;
	float changedG = G * value;
	float changedB = B * value;
	return ColorRGB(changedR, changedG, changedB);

}
ColorRGB ColorRGB::operator+(const ColorRGB& color)
{
	return ColorRGB(this->R + color.R, this->G + color.G, this->B + color.B);
}
void ColorRGB::normalize()
{
	if (R > 1.0)
		R = 1;
	if (G > 1.0)
		G = 1;
	if (B > 1.0)
		B = 1;

}
ColorRGB ColorRGB::operator*(const ColorRGB& color)
{
	return {this->R * color.R, this->G * color.G, this->B * color.B};
}
