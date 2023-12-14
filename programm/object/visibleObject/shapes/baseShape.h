//
// Created by Андрей on 09.08.2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_SHAPES_BASESHAPE_H_
#define LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_SHAPES_BASESHAPE_H_
#include "object.h"
#include "transform.h"
#include "ray.h"
#include "material.h"
const float EPS = 1e-6;

class BaseShape : public VisibleObject
{
 public:
	BaseShape() = default;
	virtual ~BaseShape() = default;
	virtual void transform(const TransformParams& transformParams) = 0;
	virtual void move(VecD3 delta) = 0;
	virtual Material getMaterial() = 0;
	virtual void accept(std::shared_ptr<Visitor> visitor) = 0;
	virtual VecD3 getCenter() = 0;
	virtual void setMaterial(const Material &material) = 0;
};

class BaseShapeFactory
{
 public:
	BaseShapeFactory() = default;
	virtual ~BaseShapeFactory() = default;

	virtual std::shared_ptr<Camera> create() = 0;
};

#endif //LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_SHAPES_BASESHAPE_H_
