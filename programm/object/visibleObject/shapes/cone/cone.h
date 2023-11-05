//
// Created by Андрей on 04/11/2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_SHAPES_CODE_CONE_H_
#define LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_SHAPES_CODE_CONE_H_

#include "baseShape.h"
#include "ray.h"
#include "color.h"

class Cone : public BaseShape
{
 private:
	VecD3 _apex;
	VecD3 _axis;
	float _slope;
	Material _material;
 public:
	Cone() = default;
	Cone(const VecD3& apex,const  VecD3& axis, float slope, const Material& material);
	void transform(const TransformParams& transformParams) override;
	float intersection(const Ray& ray) override;
	void setSpectralParams(float k_a, float k_d, float k_s);
	void setColorParams(const ColorRGB& color);
	Material getMaterial() override;
	VecD3 getNormal(VecD3 intersectionPoint);
	virtual void accept(std::shared_ptr<Visitor> visitor);
	//double getRadius(); // TODO::implement getters
	//VecD3 getCenter();

};

#endif //LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_SHAPES_CODE_CONE_H_
