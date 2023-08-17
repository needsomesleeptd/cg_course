//
// Created by Андрей on 09.08.2023.
//

#include "ray.h"

Ray::Ray(const VecD3& eye, const VecD3& direction)
{
	E = eye;
	D = direction;
}
VecD3 Ray::getPoint(float t) const
{
	return E + D * t;
}
Ray Ray::calculateReflected(const VecD3& normalToIntersection, const VecD3& intersectionPoint) const
{
	VecD3 reflectedDirection = this->D - 2.0f * normalToIntersection * dot(this->D,normalToIntersection);
	return Ray(intersectionPoint,reflectedDirection);
}
