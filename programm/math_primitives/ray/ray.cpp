//
// Created by Андрей on 09.08.2023.
//

#include "ray.h"

Ray::Ray(const VecD3& eye, const VecD3& direction)
{
	E = eye;
	D = direction;
}
VecD3 Ray::getPoint(float t)
{
	return E + D * t;
}
