//
// Created by Андрей on 14.08.2023.
//

#include "color.h"
ColorRGB::ColorRGB(int R_, int G_, int B_) : R(R_), G(G_), B(B_)
{

}
ColorRGB ColorRGB::operator*(float value)
{
	int changedR = R * value;
	int changedG = G * value;
	int changedB = B * value;
	return ColorRGB(changedR, changedG, changedB);

}
ColorRGB ColorRGB::operator+(const ColorRGB& color)
{
	return ColorRGB(this->R + color.R, this->G + color.G, this->B + color.B);
}
void ColorRGB::normalize()
{
	if (R > 255)
		R = 255;
	if (G > 255)
		G = 255;
	if (B > 255)
		B = 255;

}
ColorRGB ColorRGB::operator*(const ColorRGB& color)
{
	return {this->R * color.R, this->G * color.G, this->B * color.B};
}
