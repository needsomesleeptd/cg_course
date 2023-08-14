//
// Created by Андрей on 03.06.2023.
//

#ifndef LAB_03_OBJECT_INVISIBLEOBJECT_BASECAMERA_H_
#define LAB_03_OBJECT_INVISIBLEOBJECT_BASECAMERA_H_

#include "object.h"

class BaseCamera : public InvisibleObject
{
 public:
	BaseCamera() = default;
	virtual VecD3 getViewPoint() = 0;
	virtual VecD3 getViewDirection() = 0;
	virtual VecD3 getUpVector() = 0;
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
