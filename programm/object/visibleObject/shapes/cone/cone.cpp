//
// Created by Андрей on 04/11/2023.
//

#include "cone.h"

void Cone::transform(const TransformParams& transformParams)
{

	VecD3 rotate = TransformParams::toRadians(transformParams.getRotateParams());


	VecD3 center = getCenter();
	//_v -= center;


	_v= TransformParams::rotatePoint(_v, rotate);

	//_v += center;



}
float Cone::intersection(const Ray& ray)
{
	/*// Calculate the ray origin relative to the cone apex
	VecD3 origin = ray.E - this->_apex;

	// Calculate the coefficients for the quadratic equation
	float a = dot(ray.D, ray.D) - (1 + this->_slope * this->_slope) * dot(ray.D, this->_axis) * dot(ray.D, this->_axis);
	float b = 2 * (dot(ray.D, origin) - (1 + this->_slope * this->_slope) * dot(ray.D, this->_axis) * dot(origin, this->_axis));
	float c = dot(origin, origin) - (1 + this->_slope * this->_slope) * dot(origin, this->_axis) * dot(origin, this->_axis);

	// Calculate the discriminant
	float discriminant = b * b - 4 * a * c;

	if (discriminant < 0)
		return -1.0; // No intersection

	// Calculate the two possible values of t
	float discriminant_value = sqrt(discriminant);
	float t1 = (-b + discriminant_value) / (2 * a);
	float t2 = (-b - discriminant_value) / (2 * a);

	// Check if either of the values of t is negative
	if (t1 < 0 && t2 < 0)
		return -1.0; // No intersection

	// Find the minimum non-negative value of t
	float t = std::min(t1, t2);
	if (t < 0)
	{
		if (t1 < 0)
			t = t2;
		else if (t2 < 0)
			t = t1;
	}*/

	//TODO::implement this
	return -1.0;
}
void Cone::setSpectralParams(float k_a, float k_d, float k_s)
{
	_material._k_a = k_a;
	_material._k_d = k_d;
	_material._k_s = k_s;
}
Material Cone::getMaterial()
{
	return _material;
}
void Cone::setColorParams(const ColorRGB& color)
{
	_material._color = color;
}
void Cone::accept(std::shared_ptr<Visitor> visitor)
{
	;// do smth
}
VecD3 Cone::getNormal(VecD3 intersectionPoint)
{
	//TODO::implementinh
	return VecD3(1.0);
}
Cone::Cone(float cosa, float h, const VecD3& c, const VecD3& v, const Material& material)
{
	_cosa = cosa;
	_h = h;
	_c = c;
	_v = v;
	_material = material;
}
void Cone::move(VecD3 delta)
{
	_c = delta;
}
void Cone::setMaterial(const Material& material)
{
	_material = material;
}
VecD3 Cone::getCenter()
{
	return VecD3(_c + (_v / VecD3(2.0f, 2.0f, 2.0f)));
}
