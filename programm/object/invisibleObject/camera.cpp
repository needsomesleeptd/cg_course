#include "camera.h"
#include <QKeyEvent>
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
/*void Camera::update(QKeyEvent* e, float time)
{
	float movementspeed = 1.0f;
	float rotationspeed = 1.0f;
	int key = e->key();
	bool moved = false;
	switch (key)
	{

	case Qt::Key_W:
		_cameraStructure->move(_cameraStructure->getViewDirection() * movementspeed * time);
		moved = true;
		break;

	case Qt::Key_A:
		_cameraStructure->move(_cameraStructure->getRight() * movementspeed * time);
		moved = true;
		break;

	case Qt::Key_S:
		_cameraStructure->move(-1.0f * _cameraStructure->getViewDirection() * movementspeed * time);
		moved = true;
		break;

	case Qt::Key_D:
		_cameraStructure->move(-1.0f * _cameraStructure->getRight() * movementspeed * time);
		moved = true;
		break;

	}
	if (moved)
	{
		_cameraStructure->updateView();
		_cameraStructure->updateProjection();
	}
}*/
MatD4 Camera::getInverseProjectionMatrix()
{
	return _cameraStructure->getInverseProjectionMatrix();
}
MatD4 Camera::getInverseViewMatrix()
{
	return _cameraStructure->getInverseViewMatrix();
}

CameraFactory::CameraFactory(const VecD3& position, const VecD3& direction)
	: _position(position), _direction(direction)
{

}

std::shared_ptr<Camera> CameraFactory::create()
{
	return std::make_shared<Camera>(std::make_shared<CameraStructureImp>(_position, _direction));
}
