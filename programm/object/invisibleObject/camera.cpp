#include "camera.h"
/*
Point Camera::getCoordinates() const noexcept {
	return _cameraStructure->getCoordinates();
}*/



void Camera::accept(std::shared_ptr<Visitor> visitor)
{
	visitor->visit(*this);
}
/*
void Camera::setCoordinates(Point &coordinates) {
	Point tmp(coordinates);
//	_coordinates = tmp;
	_cameraStructure->setCoordinates(coordinates);
}
*/
Camera::Camera(const  VecD3& coordinates, const  VecD3& direction)
	: _cameraStructure(std::make_shared<CameraStructureImp>(coordinates, direction))
{
}

CameraFactory::CameraFactory(const  VecD3& position, const  VecD3& direction)
	: _position(position), _direction(direction)
{

}

std::shared_ptr<Camera> CameraFactory::create()
{
	return std::make_shared<Camera>(std::make_shared<CameraStructureImp>(_position, _direction));
}
