//
// Created by Андрей on 26.08.2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_CUDASHAPE_H_
#define LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_CUDASHAPE_H_

#include "sphere.h"


class CudaShape
{
 public:

	CudaShapeType _shapeType;
	union
	{
		Sphere _sphere;
	};

	~CudaShape();
	CudaShape(CudaShapeType shapeType,void *cudaShape);
	CudaShape operator=(const CudaShape &other);
	CudaShape(const CudaShape &other);
	__device__ double intersection(const Ray& ray)
	{
		switch (_shapeType)
		{
		case CudaShapeType::sphere:
			return _sphere.intersection(ray);
			break;
		}
	}

	__device__ VecD3 getNormal(VecD3 intersectionPoint)
	{
		switch (_shapeType)
		{
		case CudaShapeType::sphere:
			return _sphere.getNormal(intersectionPoint);
			break;
		}
	}
	__device__ Material getMaterial()
	{
		switch (_shapeType)
		{
		case CudaShapeType::sphere:
			return _sphere.getMaterial();
			break;
		}
	}
};

#endif //LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_CUDASHAPE_H_
