#include "camera.h"
#include <QKeyEvent>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include  "Input.h"






void Camera::accept(std::shared_ptr<Visitor> visitor)
{
	visitor->visit(*this);
}

Camera::Camera(const VecD3& coordinates, const VecD3& direction)
	: _cameraStructure(std::make_shared<CameraStructureImp>(coordinates, direction))
{
}
VecD3 Camera::getViewPoint()
{
	return _cameraStructure->getCoordinates();
}
VecD3 Camera::getViewDirection()
{
	return _cameraStructure->getViewDirection();
}
VecD3 Camera::getUpVector()
{
	return _cameraStructure->getUp();
}
void Camera::setImageParams(int height, int width)
{
	_cameraStructure->setViewPortParams(height, width);
}

MatD4 Camera::getInverseProjectionMatrix()
{
	return _cameraStructure->getInverseProjectionMatrix();
}
MatD4 Camera::getInverseViewMatrix()
{
	return _cameraStructure->getInverseViewMatrix();
}

CameraFactory::CameraFactory(
	const VecD3& position,
	const VecD3& direction)
	: _position(position), _direction(direction)
{

}

std::shared_ptr<Camera> CameraFactory::create()
{
	return std::make_shared<Camera>(std::make_shared<CameraStructureImp>(_position, _direction));
}
