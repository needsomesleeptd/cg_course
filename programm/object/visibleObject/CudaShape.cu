//
// Created by Андрей on 26.08.2023.
//

#include "CudaShape.cuh"
#include "shapes/baseShape.h"
CudaShape::CudaShape(CudaShapeType shapeType, void* cudaShape)
{
	_shapeType = shapeType;
	switch (shapeType)
	{
		case CudaShapeType::sphere:
			_sphere = *(Sphere*)cudaShape;
			break;
	}

}

CudaShape CudaShape::operator=(const CudaShape& other)
{
	this->_shapeType  = other._shapeType;
	switch (other._shapeType)
	{
	case CudaShapeType::sphere:
		this->_sphere = other._sphere;
		break;
	}
	return *this;
}
CudaShape::CudaShape(const CudaShape& other)
{
	this->_shapeType  = other._shapeType;
	switch (other._shapeType)
	{
	case CudaShapeType::sphere:
		this->_sphere = other._sphere;
		break;
	}

}
CudaShape::~CudaShape()
{
	switch (this->_shapeType)
	{
		case CudaShapeType::sphere:
			(&_sphere)->Sphere::~Sphere();
			break;
	}
}

