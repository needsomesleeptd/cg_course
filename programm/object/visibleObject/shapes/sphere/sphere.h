//
// Created by Андрей on 09.08.2023.
//

#ifndef DZ2_CG_COURSE_PROGRAMM_SHAPES_SPHERE_SPHERE_H_
#define DZ2_CG_COURSE_PROGRAMM_SHAPES_SPHERE_SPHERE_H_

#include "vector.h"
#include "baseShape.h"
class Sphere : public BaseShape
{
	double _radius;
	VecD3 _center;
	void transform(const TransformParams& transformParams) override;
};


#endif //DZ2_CG_COURSE_PROGRAMM_SHAPES_SPHERE_SPHERE_H_
