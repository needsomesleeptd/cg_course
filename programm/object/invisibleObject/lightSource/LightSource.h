//
// Created by Андрей on 09.08.2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_OBJECT_INVISIBLEOBJECT_LIGHTSOURCE_H_
#define LAB_03_CG_COURSE_PROGRAMM_OBJECT_INVISIBLEOBJECT_LIGHTSOURCE_H_

#include "baseLightSource.h"
#include "vector.h"

class TransformVisitor;

class LightSource : public BaseLightSource
{
 public:
	//friend void TransformVisitor::visit(LightSource& lightSorce) const;
	friend TransformVisitor;

	LightSource();
	LightSource(const VecD3& position,double intensivity);

	~LightSource() = default;

	VecD3 getPosition();
	void setPosition(const VecD3& newPosition);

	double getIntensivity();
	void setIntensivity(double newIntensivity);

	void accept(std::shared_ptr<Visitor> visitor);
 private:
	VecD3 _position;
	double _intensivity;
};

class LightSorceFactory : BaseLightSourceFactory
{
 public:
	LightSorceFactory(const VecD3& position, double intensivity);
	virtual ~LightSorceFactory() = default;

	virtual std::shared_ptr<BaseLightSource> create() override;
 private:
	VecD3 _position{ 0.0, 0.0, 0.0 };
	double _intensivity;
};

#endif //LAB_03_CG_COURSE_PROGRAMM_OBJECT_INVISIBLEOBJECT_LIGHTSOURCE_H_
