//
// Created by Андрей on 09.08.2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_SHAPES_BASESHAPE_H_
#define LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_SHAPES_BASESHAPE_H_
#include "object.h"
#include "transform.h"
#include "ray.h"
#include "material.h"

class BaseShape : public VisibleObject
{
 public:
	BaseShape() = default;
	__device__ virtual ~BaseShape() = default;
	__device__ virtual void transform(const TransformParams& transformParams) = 0;
	__device__ virtual Material getMaterial() = 0;
	__device__ virtual double intersection(const Ray& ray) = 0;
	__device__ virtual VecD3 getNormal(VecD3 intersectionPoint) = 0;
	__device__ virtual void accept(std::shared_ptr<Visitor> visitor) = 0;
};

class BaseShapeFactory
{
 public:
	BaseShapeFactory() = default;
	virtual ~BaseShapeFactory() = default;

	virtual std::shared_ptr<Camera> create() = 0;
};

#endif //LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_SHAPES_BASESHAPE_H_
