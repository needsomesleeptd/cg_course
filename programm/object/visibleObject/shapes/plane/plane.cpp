//
// Created by Андрей on 07/09/2023.
//

#include "plane.h"
//#include <iostream>
void Plane::transform(const TransformParams& transformParams)
{
	//TODO:: implement this
}
double Plane::intersection(const Ray& ray)
{
	 float normalDDot = dot(_normal , ray.D);
	 if (fabs(normalDDot) < EPS)
		 return -1.0; //no Intersection

	 float t = dot(_normal , (_point - ray.E)) / normalDDot;
	 //std::cout<< t << "\n";
	 return t;
}
void Plane::setSpectralParams(float k_a, float k_d, float k_s)
{
	_material._k_s = k_s;
	_material._k_a = k_a;
	_material._k_d = k_d;

}
void Plane::setColorParams(const ColorRGB& color)
{
	_material._color = color;
}
Material Plane::getMaterial()
{
	return _material;
}
VecD3 Plane::getNormal(VecD3 intersectionPoint)
{
	return _normal;
}
void Plane::accept(std::shared_ptr<Visitor> visitor)
{
	visitor->visit(*this);
}
