//
// Created by Андрей on 29/11/2023.
//

#include "box.h"
Box::Box(const VecD3& position, glm::mat3 rotation, VecD3 halfSize, const Material& material)
{
	_position = position;
	_rotation = rotation;
	_halfSize = halfSize;
	_material = material;
}
void Box::transform(const TransformParams& transformParams)
{
	//pass
}
float Box::intersection(const Ray& ray)
{
	return -1.0;
}
void Box::setSpectralParams(float k_a, float k_d, float k_s)
{
	_material._k_a = k_a;
	_material._k_d = k_d;
	_material._k_s = k_s;
}
void Box::setColorParams(const ColorRGB& color)
{
	_material._color = color;
}
Material Box::getMaterial()
{
	return _material;
}
VecD3 Box::getNormal(VecD3 intersectionPoint)
{
	return VecD3(1.0);
}
void Box::accept(std::shared_ptr<Visitor> visitor)
{
	//do smth;
}
void Box::move(VecD3 delta)
{
	_position = delta;
}
void Box::setMaterial(const Material& material)
{
	_material = material;
}
VecD3 Box::getCenter()
{
	return _position;
}
