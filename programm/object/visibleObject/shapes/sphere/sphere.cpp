//
// Created by Андрей on 09.08.2023.
//

#include "sphere.h"
#include "vector.h"
void Sphere::transform(const TransformParams& transformParams)
{
	_center += transformParams.getMoveParams();
	_radius += transformParams.getScaleParams()[0];
}
double Sphere::intersection(const Ray& ray)
{
	float a = dot(ray.D, ray.D);
	float b = dot(ray.E * 2.0f, ray.D);
	float c = dot(ray.E, ray.E) - 1;
	float discriminant = b * b - 4 * a * c;
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
	_material.k_a = k_a;
	_material.k_d = k_d;
	_material.k_s = k_s;
}
void Sphere::setColorParams(const ColorRGB& color)
{
	_material.color = color;
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
