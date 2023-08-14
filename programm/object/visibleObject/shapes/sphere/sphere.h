//
// Created by Андрей on 09.08.2023.
//

#ifndef DZ2_CG_COURSE_PROGRAMM_SHAPES_SPHERE_SPHERE_H_
#define DZ2_CG_COURSE_PROGRAMM_SHAPES_SPHERE_SPHERE_H_

#include "baseShape.h"
#include "ray.h"
#include "color.h"

class Sphere : public BaseShape
{
	double _radius;
	 VecD3 _center;
	 Material material;
 public:
	void transform(const TransformParams& transformParams) override;
	double intersection(const Ray& ray) override;
	void setSpectralParams(float k_a,float k_d,float k_s);
	void setColorParams(const ColorRGB& color);
	Material getMaterial() override;
};


#endif //DZ2_CG_COURSE_PROGRAMM_SHAPES_SPHERE_SPHERE_H_
