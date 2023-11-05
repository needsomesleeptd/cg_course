#include "camera.h"
#include <QKeyEvent>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include  "Input.h"
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
void Camera::update(float time)
{
	Input::update();
	//std::cout << Input::mouseDelta().x() << Input::mouseDelta().y() << std::endl;


	const float transSpeed = 1.5f;
	const float rotSpeed = 0.5f;
	bool moved = false;

	std::cout << "Right is pressed";

	// Handle rotations
	/*m_camera.rotate(-rotSpeed * Input::mouseDelta().x(), _cameraStructure->getUp());
	m_camera.rotate(-rotSpeed * Input::mouseDelta().y(), _cameraStructure->getRight());*/

	VecD3 translation = { 0.0, 0.0, 0.0 };

	if (Input::keyTriggered(Qt::Key_W))
	{

		translation += _cameraStructure->getViewDirection();
	}
	if (Input::keyTriggered(Qt::Key_S))
	{
		translation -= _cameraStructure->getViewDirection();
	}
	if (Input::keyTriggered(Qt::Key_A))
	{
		translation -= _cameraStructure->getRight();
	}
	if (Input::keyTriggered(Qt::Key_D))
	{
		translation += _cameraStructure->getRight();
	}
	if (Input::keyTriggered(Qt::Key_Control))
	{
		translation -= _cameraStructure->getUp();
	}
	if (Input::keyTriggered(Qt::Key_Space))
	{
		translation += _cameraStructure->getUp();
	}
	std::cout << translation.x << " " << translation.y << " " << translation.z << std::endl;
	_cameraStructure->move(translation * transSpeed);
	std::cout << translation.x << " " << translation.y << " " << translation.z << std::endl;
	if (Input::buttonPressed(Qt::RightButton))
	{
		float delta_x = Input::mouseDelta().x() * 0.02f;
		float delta_y = Input::mouseDelta().y() * 0.02f;

		glm::vec2 delta = { delta_x, delta_y };
		if (delta.x != 0.0f || delta.y != 0.0f)
		{
			float pitchDelta = delta.y * rotSpeed;
			float yawDelta = delta.x * rotSpeed;

			glm::quat q = glm::normalize(glm::cross(glm::angleAxis(-pitchDelta, _cameraStructure->getRight()),
				glm::angleAxis(-yawDelta, glm::vec3(0.f, 1.0f, 0.0f))));

			_cameraStructure->_forward = glm::rotate(q, _cameraStructure->getViewDirection());
			moved = true;
		}
	}
	if (moved)
	{
		_cameraStructure->updateView();
		_cameraStructure->updateProjection();
	}

	//std::cout << translation.x << " " << translation.y << " " << translation.z << std::endl;

	//}
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
