//
// Created by Андрей on 09.08.2023.
//

#include "LightSource.cuh"


__host__  __device__ LightSource::LightSource()
{
	_intensivity = 1;
	_position =  VecD3({ 0.0, 0.0, 0.0 });
	_color = ColorRGB(1.0,1.0,1.0);
}
__host__ __device__ VecD3 LightSource::getPosition()
{
	return  VecD3(_position); //TODO::Might slow down
}
__host__  __device__ void LightSource::setPosition(const  VecD3& newPosition)
{
	_position = newPosition;
}
__host__  __device__ double LightSource::getIntensivity()
{
	return _intensivity;
}
__host__  __device__ void LightSource::setIntensivity(double newIntensivity)
{
	_intensivity = newIntensivity;
}
__host__  __device__ LightSource::LightSource(const  VecD3& position, double intensivity)
{
	_position = position;
	_intensivity = intensivity;
	ColorRGB intensivityColor(_intensivity,_intensivity,_intensivity);
	setColor(intensivityColor);
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
