//
// Created by Андрей on 29/11/2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_SHAPES_BOX_BOX_H_
#define LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_SHAPES_BOX_BOX_H_

#include "baseShape.h"
#include "ray.h"
#include "color.h"

class Box : public BaseShape
{
 public:
	VecD3 _position;
	glm::mat3 _rotation;
	VecD3 _halfSize;
	Material _material;
 public:
	Box() = default;
	Box(const VecD3& position, glm::mat3 rotation, VecD3 halfSize, const Material& material);
	void transform(const TransformParams& transformParams) override;
	float intersection(const Ray& ray) override;
	void setSpectralParams(float k_a, float k_d, float k_s);
	void setColorParams(const ColorRGB& color);
	Material getMaterial() override;
	VecD3 getNormal(VecD3 intersectionPoint);
	virtual void accept(std::shared_ptr<Visitor> visitor);
	void setMaterial(const Material& material) override;
	VecD3 getCenter() override;
	void move(VecD3 delta) override;
};

#endif //LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_SHAPES_BOX_BOX_H_
