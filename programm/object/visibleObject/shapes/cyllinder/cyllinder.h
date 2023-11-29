//
// Created by Андрей on 29/11/2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_SHAPES_CYLLINDER_CYLLINDER_H_
#define LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_SHAPES_CYLLINDER_CYLLINDER_H_

#include "baseShape.h"
#include "ray.h"
#include "color.h"


class Cyllinder : public BaseShape
{
 public:
	VecD3 _extr_a;
	VecD3 _extr_b;
	float _ra;
	Material _material;
 public:
	Cyllinder() = default;
	Cyllinder(const VecD3& extr_a,const VecD3& extr_b,float ra,const Material& material);
	void transform(const TransformParams& transformParams) override;
	float intersection(const Ray& ray) override;
	void setSpectralParams(float k_a,float k_d,float k_s);
	void setColorParams(const ColorRGB& color);
	Material getMaterial() override;
	VecD3 getNormal(VecD3 intersectionPoint);
	virtual void accept(std::shared_ptr<Visitor> visitor);

	void move(VecD3 delta) override;
};

#endif //LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_SHAPES_CYLLINDER_CYLLINDER_H_
