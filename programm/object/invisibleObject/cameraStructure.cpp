#include "cameraStructure.h"
#include "vector.h"

CameraStructureImp::CameraStructureImp(const  VecD3& coordinates, const  VecD3& direction)
	: _coordinates(coordinates), _front(direction)
{
}

 VecD3 CameraStructureImp::getCoordinates() const
{
	return  VecD3(_coordinates);
}

/*[[nodiscard]] Matrix4 CameraStructureImp::getView()
{
	Vector3 coords = Vector3{ _coordinates.getX(), _coordinates.getY(), _coordinates.getZ() };
	Matrix4 view = glm::lookAt(coords, coords + _front, _up);
	return view;
}*/

void CameraStructureImp::transform(const TransformParams& transformParams)
{
	auto moveParams = transformParams.getMoveParams();
	_coordinates += moveParams;
	//auto rotateParams = transformParams.getRotateParams(); TODO::add rotate
	//rotate(rotateParams.getX(), rotateParams.getY());
}

void CameraStructureImp::rotate(float xOffset, float yOffset)
{
	_yaw += xOffset;
	_pitch += yOffset;
	if (_pitch > 90.0f)
		_pitch = 90.0f;
	if (_pitch < -90.0f)
		_pitch = -90.0f;
	if (_yaw > 90.0f)
		_yaw = 90.0f;
	if (_yaw < -90.0f)
		_yaw = -90.0f;
	//updateVectors();
}

void CameraStructureImp::move(const  VecD3& moveParams)
{
	_coordinates += moveParams;
}

/*void CameraStructureImp::updateVectors()
{
	Vector3 front;
	front.x = cos(glm::radians(_yaw)) * cos(glm::radians(_pitch));
	front.y = sin(glm::radians(_pitch));
	front.z = sin(glm::radians(_yaw)) * cos(glm::radians(_pitch));
	_front = normalize(front);
	_right = normalize(cross(_front, _worldUp));
	_up = normalize(cross(_right, _front));
}*/

void CameraStructureImp::setCoordinates(const  VecD3& coordinates)
{
	_coordinates = coordinates;
}

/*Matrix4 CameraStructureImp::getProjection() const
{
	return perspective(glm::radians(90.0f), _aspect, _zNear, _zFar);
}
Vector3 CameraStructureImp::setDirection(const Vector3& direction)
{
	_front = direction;
}
*/

