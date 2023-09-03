//
// Created by Андрей on 03.06.2023.
//

#ifndef LAB_03_OBJECT_INVISIBLEOBJECT_BASECAMERA_H_
#define LAB_03_OBJECT_INVISIBLEOBJECT_BASECAMERA_H_

#include "object.h"
#include "vector.h"

class BaseCamera : public InvisibleObject
{
 public:
	BaseCamera() = default;
	__host__ __device__  virtual VecD3 getViewPoint() = 0;
	__host__ __device__  virtual VecD3 getViewDirection() = 0;
	__host__ __device__  virtual VecD3 getUpVector() = 0;
	__host__ __device__  virtual MatD4  getInverseProjectionMatrix() = 0;
	__host__ __device__ virtual MatD4  getInverseViewMatrix() = 0;
	virtual ~BaseCamera() = default;
};

class BaseCameraFactory
{
 public:
	BaseCameraFactory() = default;
	virtual ~BaseCameraFactory() = default;

	virtual std::shared_ptr<Camera> create() = 0;
};

#endif //LAB_03_OBJECT_INVISIBLEOBJECT_BASECAMERA_H_
