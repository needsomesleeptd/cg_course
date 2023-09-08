//
// Created by Андрей on 07/09/2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_SHAPES_PLANE_PLANE_H_
#define LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_SHAPES_PLANE_PLANE_H_

#include "vector.h"
#include "baseShape.h"
class Plane : public BaseShape
{
 public:
	Plane(const VecD3& point, const VecD3& normal, const Material& material) : _point(point), _normal(normal), _material(material)
	{

	}
	void transform(const TransformParams& transformParams) override;
	float intersection(const Ray& ray) override;
	void setSpectralParams(float k_a,float k_d,float k_s);
	void setColorParams(const ColorRGB& color);
	Material getMaterial() override;
	VecD3 getNormal(VecD3 intersectionPoint);
	virtual void accept(std::shared_ptr<Visitor> visitor);
 private:
	VecD3 _point;
	VecD3 _normal;
	Material _material;
};

#endif //LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_SHAPES_PLANE_PLANE_H_
