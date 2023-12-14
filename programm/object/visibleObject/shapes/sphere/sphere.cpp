//
// Created by Андрей on 09.08.2023.
//

#include "sphere.h"
#include "vector.h"
#include <iostream> //TODO::remove after print
void Sphere::transform(const TransformParams& transformParams)
{
	//Do nothing

}

void Sphere::setSpectralParams(float k_a, float k_d, float k_s)
{
	_material._k_a = k_a;
	_material._k_d = k_d;
	_material._k_s = k_s;
}
void Sphere::setColorParams(const ColorRGB& color)
{
	_material._color = color;
}
Material Sphere::getMaterial()
{
	return _material;
}

Sphere::Sphere(const VecD3& center, double radius, const Material& material)
	: _center(center), _radius(radius), _material(material)
{

}
void Sphere::accept(std::shared_ptr<Visitor> visitor)
{
	visitor->visit(*this);
}
double Sphere::getRadius()
{
	return _radius;
}
VecD3 Sphere::getCenter()
{
	return _center;
}
void Sphere::move(VecD3 delta)
{

	_center = delta;
}
void Sphere::setMaterial(const Material& material)
{
	_material = material;
}

