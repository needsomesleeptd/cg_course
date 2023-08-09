//
// Created by Андрей on 09.08.2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_OBJECT_INVISIBLEOBJECT_LIGHTSOURCE_BASELIGHTSOURCE_H_
#define LAB_03_CG_COURSE_PROGRAMM_OBJECT_INVISIBLEOBJECT_LIGHTSOURCE_BASELIGHTSOURCE_H_

#include "object.h"
#include "vector.h"
#include <memory>
class BaseLightSource : public InvisibleObject
{
 public:
	BaseLightSource() = default;
	virtual ~BaseLightSource() = default;
};

class BaseLightSourceFactory
{
 public:
	BaseLightSourceFactory() = default;
	virtual ~BaseLightSourceFactory() = default;

	virtual std::shared_ptr<BaseLightSource> create()  = 0;
};
#endif //LAB_03_CG_COURSE_PROGRAMM_OBJECT_INVISIBLEOBJECT_LIGHTSOURCE_BASELIGHTSOURCE_H_
