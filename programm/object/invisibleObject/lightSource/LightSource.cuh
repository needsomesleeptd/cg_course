//
// Created by Андрей on 09.08.2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_OBJECT_INVISIBLEOBJECT_LIGHTSOURCE_H_
#define LAB_03_CG_COURSE_PROGRAMM_OBJECT_INVISIBLEOBJECT_LIGHTSOURCE_H_

#include "baseLightSource.h"
#include "vector.h"
#include "color.h"

class TransformVisitor;

class LightSource : public BaseLightSource
{
 public:
	//friend void TransformVisitor::visit(LightSource& lightSorce) const;
	friend TransformVisitor;

	__host__ __device__ LightSource();
	__host__  __device__ LightSource(const  VecD3& position,double intensivity);

	~LightSource() override = default;

	__host__  __device__ VecD3 getPosition() override;
	__host__  __device__ void setPosition(const  VecD3& newPosition) override;

	__host__  __device__ double getIntensivity() override;
	__host__  __device__ void setIntensivity(double newIntensivity);

	void accept(std::shared_ptr<Visitor> visitor);
	__host__  __device__ ColorRGB getColor() override;
	__host__  __device__ void setColor(const ColorRGB& color) override;
 private:
	 VecD3 _position;
	 double _intensivity;
	 ColorRGB _color;
};

class LightSourceFactory : BaseLightSourceFactory
{
 public:
	LightSourceFactory(const  VecD3& position, double intensivity);
	virtual ~LightSourceFactory() = default;

	virtual std::shared_ptr<BaseLightSource> create() override;
 private:
	 VecD3 _position{ 0.0, 0.0, 0.0 };
	double _intensivity;
};

#endif //LAB_03_CG_COURSE_PROGRAMM_OBJECT_INVISIBLEOBJECT_LIGHTSOURCE_H_
