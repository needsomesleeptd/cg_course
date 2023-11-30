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
float Sphere::intersection(const Ray& ray)
{
	VecD3 origin  = ray.E - this->_center;
	float a = dot(ray.D, ray.D);
	float b = 2.0f * dot(origin, ray.D);
	float c = dot(origin, origin) - _radius * _radius;
	float discriminant = b * b - 4 * a * c;
//	std::cout << ray.E[0] << ray.E[1] << ray.E[2] << " " << ray.D[0] << ray.D[1] << ray.D[2] << " " << discriminant << std::endl;
	if (discriminant < 0)
		return -1.0; //TODO::might need a special flag of intersection
	else
	{
		float discriminant_value = sqrt(discriminant);
		double t1 = (-b + discriminant_value) / (2 * a);
		double t2 = (-b - discriminant_value) / (2 * a);
		if (t1 < 0 && t2 < 0)
			return -1.0;
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
VecD3 Sphere::getNormal(VecD3 intersectionPoint)
{
	return (intersectionPoint - _center) * float(_radius);
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

