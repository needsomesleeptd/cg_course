//
// Created by Андрей on 26.08.2023.
//

#include "CudaShape.h"
CudaShape::CudaShape(CudaShapeType shapeType, void* cudaShape)
{
	_shapeType = shapeType;
	switch (shapeType)
	{
		case CudaShapeType::sphere:
			_sphere = *((Sphere*)cudaShape);
			break;
	}

}
