//
// Created by Андрей on 09.08.2023.
//

#include "LightSource.h"


LightSource::LightSource()
{
	_intensivity = 1;
	_position =  VecD3({ 0.0, 0.0, 0.0 });
	_color = ColorRGB(255,255,25);
}
 VecD3 LightSource::getPosition()
{
	return  VecD3(_position); //TODO::Might slow down
}
void LightSource::setPosition(const  VecD3& newPosition)
{
	_position = newPosition;
}
double LightSource::getIntensivity()
{
	return _intensivity;
}
void LightSource::setIntensivity(double newIntensivity)
{
	_intensivity = newIntensivity;
}
LightSource::LightSource(const  VecD3& position, double intensivity)
{
	_position = position;
	_intensivity = intensivity;
}

LightSourceFactory::LightSourceFactory(const  VecD3& position, double intensivity)
{
	_position = position;
	_intensivity = intensivity;
}

std::shared_ptr<BaseLightSource> LightSourceFactory::create()
{
	return std::make_shared<LightSource>(_position, _intensivity);
}

void LightSource::accept(std::shared_ptr<Visitor> visitor)
{
	visitor->visit(*this);
}
ColorRGB LightSource::getColor()
{
	return _color;
}
void LightSource::setColor(const ColorRGB& color)
{
	_color = color;
}
