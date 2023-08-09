//
// Created by Андрей on 09.08.2023.
//

#include "sphere.h"
void Sphere::transform(const TransformParams& transformParams)
{
	_center += transformParams.getMoveParams();
	_radius += transformParams.getScaleParams()[0];
}
