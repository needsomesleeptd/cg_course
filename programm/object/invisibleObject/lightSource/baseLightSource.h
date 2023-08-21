//
// Created by Андрей on 09.08.2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_OBJECT_INVISIBLEOBJECT_LIGHTSOURCE_BASELIGHTSOURCE_H_
#define LAB_03_CG_COURSE_PROGRAMM_OBJECT_INVISIBLEOBJECT_LIGHTSOURCE_BASELIGHTSOURCE_H_

#include "object.h"
#include "vector.h"

class BaseLightSource : public InvisibleObject
{
 public:
	BaseLightSource() = default;
	virtual ~BaseLightSource() = default;
	__device__ virtual VecD3 getPosition() = 0;
	__device__ virtual double getIntensivity() = 0;
	__device__ virtual void setPosition(const VecD3& newPosition) = 0;
	__device__ virtual void setIntensivity(double newIntensivity) = 0;
	__device__ virtual ColorRGB getColor() = 0;
	__device__ virtual void setColor(const ColorRGB& color) = 0;
};

class BaseLightSourceFactory
{
 public:
	BaseLightSourceFactory() = default;
	virtual ~BaseLightSourceFactory() = default;

	virtual std::shared_ptr<BaseLightSource> create() = 0;
};
#endif //LAB_03_CG_COURSE_PROGRAMM_OBJECT_INVISIBLEOBJECT_LIGHTSOURCE_BASELIGHTSOURCE_H_
