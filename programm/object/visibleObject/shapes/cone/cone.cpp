//
// Created by Андрей on 04/11/2023.
//

#include "cone.h"

void Cone::transform(const TransformParams& transformParams)
{

	VecD3 rotate = TransformParams::toRadians(transformParams.getRotateParams());

	VecD3 center = getCenter();

	_v = TransformParams::rotatePoint(_v, rotate);

}

void Cone::setSpectralParams(float k_a, float k_d, float k_s)
{
	_material._k_a = k_a;
	_material._k_d = k_d;
	_material._k_s = k_s;
}
Material Cone::getMaterial()
{
	return _material;
}
void Cone::setColorParams(const ColorRGB& color)
{
	_material._color = color;
}
void Cone::accept(std::shared_ptr<Visitor> visitor)
{
	;// do smth
}

Cone::Cone(float cosa, float h, const VecD3& c, const VecD3& v, const Material& material)
{
	_cosa = cosa;
	_h = h;
	_c = c;
	_v = v;
	_material = material;
}
void Cone::move(VecD3 delta)
{
	_c = delta;
}
void Cone::setMaterial(const Material& material)
{
	_material = material;
}
VecD3 Cone::getCenter()
{
	return VecD3(_c + (_v / VecD3(2.0f, 2.0f, 2.0f)));
}
