//
// Created by Андрей on 29/11/2023.
//

#include <QDebug>
#include "cyllinder.h"
Cyllinder::Cyllinder(const VecD3& extr_a, const VecD3& extr_b, float ra, const Material& material)
{
	_extr_a = extr_a;
	_extr_b = extr_b;
	_ra = ra;
	_material = material;
}
void Cyllinder::transform(const TransformParams& transformParams)
{
	VecD3 rotate = TransformParams::toRadians(transformParams.getRotateParams());


	VecD3 center = getCenter();
	_extr_a -= center;
	_extr_b -= center;

	_extr_a = TransformParams::rotatePoint(_extr_a, rotate);
	_extr_b = TransformParams::rotatePoint(_extr_b, rotate);

	_extr_a += center;
	_extr_b += center;

	qDebug() << "pos_extr" << _extr_a.x << _extr_a.y << _extr_a.z;

}

void Cyllinder::setSpectralParams(float k_a, float k_d, float k_s)
{
	_material._k_a = k_a;
	_material._k_d = k_d;
	_material._k_s = k_s;
}
void Cyllinder::setColorParams(const ColorRGB& color)
{
	_material._color = color;
}
Material Cyllinder::getMaterial()
{
	return _material;
}

void Cyllinder::accept(std::shared_ptr<Visitor> visitor)
{
	//pass;
}
void Cyllinder::move(VecD3 delta)
{
	VecD3 len_vec = abs(_extr_b - _extr_a);
	_extr_b = delta + (len_vec / VecD3(2.0, 2.0, 2.0));
	_extr_a = delta - (len_vec / VecD3(2.0, 2.0, 2.0));
}
void Cyllinder::setMaterial(const Material& material)
{
	_material = material;
}
VecD3 Cyllinder::getCenter()
{
	return (_extr_a + _extr_b) / VecD3(2.0f, 2.0f, 2.0f);
}
