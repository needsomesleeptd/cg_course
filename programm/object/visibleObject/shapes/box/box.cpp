//
// Created by Андрей on 29/11/2023.
//

#include "box.h"

glm::mat3 rotationAxisAngle( VecD3 v, float angle )
{
	float s = sin( angle ); //in radians
	float c = cos( angle );
	float ic = 1.0 - c;

	return glm::mat3( v.x*v.x*ic + c,     v.y*v.x*ic - s*v.z, v.z*v.x*ic + s*v.y,
		v.x*v.y*ic + s*v.z, v.y*v.y*ic + c,     v.z*v.y*ic - s*v.x,
		v.x*v.z*ic - s*v.y, v.y*v.z*ic + s*v.x, v.z*v.z*ic + c);
}



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

	VecD3 axis_x = VecD3(1.0,0.0,0.0);
	VecD3 axis_y = VecD3(0.0,1.0,0.0);
	VecD3 axis_z = VecD3(0.0,0.0,1.0);





	glm::mat3 rot_x = rotationAxisAngle(axis_x,angle_x);

	glm::mat3 rot_y = rotationAxisAngle(axis_y,angle_y);

	glm::mat3 rot_z = rotationAxisAngle(axis_z,angle_z);




	_rotation = rot_x  * rot_y * rot_z;
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
