#include "cameraStructure.h"
#include "vector.h"
#include "glmWrapper.h"

CameraStructureImp::CameraStructureImp(const  VecD3& coordinates, const  VecD3& direction)
	: _coordinates(coordinates), _forward(direction)
{
}

 VecD3 CameraStructureImp::getCoordinates() const
{
	return  VecD3(_coordinates);
}

VecD3 CameraStructureImp::getViewDirection() const
{
	return  VecD3(_forward);
}
/*[[nodiscard]] Matrix4 CameraStructureImp::getViewDirection()
{
	Vector3 coords = Vector3{ _coordinates.getX(), _coordinates.getY(), _coordinates.getZ() };
	Matrix4 view = glm::lookAt(coords, coords + _forward, _up);
	return view;
}*/

void CameraStructureImp::transform(const TransformParams& transformParams)
{
	auto moveParams = transformParams.getMoveParams();
	_coordinates += moveParams;
	//auto rotateParams = transformParams.getRotateParams(); TODO::add rotate
	//rotate(rotateParams.getX(), rotateParams.getY());
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
	_forward = normalize(front);
	_right = normalize(cross(_forward, _worldUp));
	_up = normalize(cross(_right, _forward));
}*/

void CameraStructureImp::setCoordinates(const  VecD3& coordinates)
{
	_coordinates = coordinates;
}
VecD3 CameraStructureImp::getUp() const
{
	return VecD3(_up);
}
void CameraStructureImp::updateView()
{
	_mView = glm::lookAt(_coordinates,_coordinates + _forward,_up);//TODO::write own lookat function
	_mInverseView = glm::inverse(_mView);
}


void CameraStructureImp::setDirection(const VecD3 & direction)
{
	_forward = direction;
}
void CameraStructureImp::updateProjection()
{
	_mProjection = glm::perspectiveFov(glm::radians(_verticalFOV), (float)_viewPortWidth, (float)_viewPortHeight, _nearCLip, _farClip); //TODO::remove glm
	_mInverseProjection = glm::inverse(_mProjection);
}
void CameraStructureImp::setViewPortParams(int height, int width)
{
	_viewPortHeight = height;
	_viewPortWidth = width;
}
MatD4 CameraStructureImp::getInverseProjectionMatrix()
{
	return _mInverseProjection;
}
MatD4 CameraStructureImp::getInverseViewMatrix()
{
	return _mInverseView;
}
VecD3 CameraStructureImp::getRight() const
{
	return _right;
}



