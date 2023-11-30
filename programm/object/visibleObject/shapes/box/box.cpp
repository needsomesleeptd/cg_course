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

	VecD3 rotate = TransformParams::toRadians(transformParams.getRotateParams());
	float angle_x = rotate.x;
	float angle_y = rotate.y;
	float angle_z = rotate.z;





	glm::mat3 rot_x = {{ 1.0, 0.0, 0.0 },
	                   { 0, cos(angle_x), -sin(angle_x) },
	                   { 0, sin(angle_x), cos(angle_x) }};

	glm::mat3 rot_y = {{ cos(angle_y), 0.0, sin(angle_y) },
	                   { 0.0, 1, 0.0 },
	                   { -sin(angle_y), 0, cos(angle_y) }};

	glm::mat3 rot_z = {{ cos(angle_z), -sin(angle_z), 0.0 },
	                   { sin(angle_z), cos(angle_x), 0.0 },
	                   { 0.0, 0.0, 1.0 }};

	_rotation = rot_x * rot_y * rot_z;
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
