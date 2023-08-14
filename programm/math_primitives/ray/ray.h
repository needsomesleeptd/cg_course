//
// Created by Андрей on 09.08.2023.
//

#ifndef DZ2_CG_COURSE_PROGRAMM_MATH_PRIMITIVES_RAY_RAY_H_
#define DZ2_CG_COURSE_PROGRAMM_MATH_PRIMITIVES_RAY_RAY_H_
#include "vector.h"

class Ray
{
 public:
	Ray() = default;
	Ray(const VecD3& eye, const VecD3& direction);
	double t;
	VecD3 E;
	VecD3 D;

};

#endif //DZ2_CG_COURSE_PROGRAMM_MATH_PRIMITIVES_RAY_RAY_H_
