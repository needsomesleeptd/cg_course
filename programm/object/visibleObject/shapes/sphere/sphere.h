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

	__host__ __device__ Sphere(const VecD3& center,double radius,const Material& material);
	__device__ void transform(const TransformParams& transformParams) override;
	__device__ double intersection(const Ray& ray) override;
	__host__ __device__ void setSpectralParams(float k_a,float k_d,float k_s);
	__host__ __device__ void setColorParams(const ColorRGB& color);
	__host__ __device__ Material getMaterial() override;
	__device__ VecD3 getNormal(VecD3 intersectionPoint) override;
	void accept(std::shared_ptr<Visitor> visitor) override;


};


#endif //DZ2_CG_COURSE_PROGRAMM_SHAPES_SPHERE_SPHERE_H_
