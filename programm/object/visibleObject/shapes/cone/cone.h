//
// Created by Андрей on 04/11/2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_SHAPES_CODE_CONE_H_
#define LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_SHAPES_CODE_CONE_H_

#include "baseShape.h"
#include "color.h"

class Cone : public BaseShape
{
 public:
	float _cosa;    // half cone angle
	float _h;    // height
	VecD3 _c;        // tip position
	VecD3 _v;        // axis
	Material _material;    // material
 public:
	Cone() = default;
	Cone(float cosa,float h, const VecD3 &c,const VecD3 &v, const Material& material);
	void transform(const TransformParams& transformParams) override;
	void setSpectralParams(float k_a, float k_d, float k_s);
	void setColorParams(const ColorRGB& color);
	Material getMaterial() override;
	virtual void accept(std::shared_ptr<Visitor> visitor);
	void setMaterial(const Material& material) override;
	VecD3 getCenter() override;
	void move(VecD3 delta) override;


};

#endif //LAB_03_CG_COURSE_PROGRAMM_OBJECT_VISIBLEOBJECT_SHAPES_CODE_CONE_H_
