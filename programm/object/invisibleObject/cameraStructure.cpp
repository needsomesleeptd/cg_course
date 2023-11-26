#include "cameraStructure.h"
#include "vector.h"

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


void CameraStructureImp::transform(const TransformParams& transformParams)
{
	auto moveParams = transformParams.getMoveParams();
	_coordinates += moveParams;
}


#include <iostream>

void CameraStructureImp::move(const  VecD3& moveParams)
{
	_coordinates += moveParams;
}



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
	_forward = normalise(_forward);
	_mView = glm::lookAt(glm::vec3(0.f, 0.0f, 0.0f), _forward,glm::vec3(0.f, 1.0f, 0.0f));//TODO::write own lookat function
	_mInverseView = glm::inverse(_mView);
}


void CameraStructureImp::setDirection(const VecD3 & direction)
{
	_forward = direction;
}
void CameraStructureImp::updateProjection()
{
	_mProjection = glm::perspectiveFov(_verticalFOV, (float)_viewPortWidth, (float)_viewPortHeight, _nearCLip, _farClip);
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



