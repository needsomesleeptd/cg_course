//
// Created by Андрей on 09.08.2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_SHAPES_BASESHAPE_H_
#define LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_SHAPES_BASESHAPE_H_
#include "object.h"
#include "transform.cuh"
#include "ray.h"
#include "material.h"

enum class CudaShapeType
{
	sphere
};

class BaseShape : public VisibleObject
{
 public:
	BaseShape() = default;
	virtual __device__ __host__ ~BaseShape() = default;
	virtual __device__  void transform(const TransformParams& transformParams) = 0;
	virtual __host__ __device__      Material getMaterial() = 0;
	virtual __device__  double intersection(const Ray& ray) = 0;
	virtual   __device__  VecD3 getNormal(VecD3 intersectionPoint) = 0;
	virtual void accept(std::shared_ptr<Visitor> visitor) = 0;
	virtual CudaShapeType getShapeType() = 0;
	virtual size_t getByteSize() = 0;

};

class BaseShapeFactory
{
 public:
	BaseShapeFactory() = default;
	virtual ~BaseShapeFactory() = default;

	virtual std::shared_ptr<Camera> create() = 0;
};

#endif //LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_SHAPES_BASESHAPE_H_
