//
// Created by Андрей on 09.08.2023.
//

#ifndef DZ2_CG_COURSE_PROGRAMM_SHAPES_SPHERE_SPHERE_H_
#define DZ2_CG_COURSE_PROGRAMM_SHAPES_SPHERE_SPHERE_H_

#include "baseShape.h"
#include "ray.h"
#include "color.h"

class Sphere : public BaseShape
{
 private:
	 double _radius;
	 VecD3 _center;
	 Material _material;
 public:
	Sphere() = default;
	Sphere(const VecD3& center,double radius,const Material& material);
	void transform(const TransformParams& transformParams) override;
	float intersection(const Ray& ray) override;
	void setSpectralParams(float k_a,float k_d,float k_s);
	void setColorParams(const ColorRGB& color);
	Material getMaterial() override;
	VecD3 getNormal(VecD3 intersectionPoint);
	virtual void accept(std::shared_ptr<Visitor> visitor);
	double getRadius();
	VecD3 getCenter() override;
	void setMaterial(const Material& material) override;
	void move(VecD3 delta) override;


};


#endif //DZ2_CG_COURSE_PROGRAMM_SHAPES_SPHERE_SPHERE_H_
