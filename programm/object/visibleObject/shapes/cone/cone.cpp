//
// Created by Андрей on 04/11/2023.
//

#include "cone.h"
Cone::Cone(const VecD3& apex,const VecD3& axis, float slope, const Material& material)
{
	_apex = apex;
	_axis = axis;
	_slope = slope;
	_material = material;
}
void Cone::transform(const TransformParams& transformParams)
{
	;//TODO:: do smth
}
float Cone::intersection(const Ray& ray)
{
	// Calculate the ray origin relative to the cone apex
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
	}

	return t;
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
	VecD3 __axisToIntersection = intersectionPoint - this->_axis;
	VecD3 _axisProjection = this->_axis * dot(__axisToIntersection, this->_axis);

	// Check if the intersection point is on the top or bottom of the cone
	// If it is, return the _axis direction as the normal
	float distanceFrom_axis = _axisProjection.length();
	if (distanceFrom_axis < this->_slope * dot(__axisToIntersection, this->_axis))
	{
		return this->_axis;
	}

	// Calculate the direction of the normal by projecting __axisToIntersection onto the cone's surface
	VecD3 coneSurfaceProjection = __axisToIntersection - _axisProjection;
	coneSurfaceProjection = normalize(coneSurfaceProjection);

	// Calculate the normal vector
	VecD3 normal = coneSurfaceProjection - (this->_slope * this->_slope) * _axisProjection;

	// Normalize the normal vector
	normal = normalize(normal);

	return normal;
}
