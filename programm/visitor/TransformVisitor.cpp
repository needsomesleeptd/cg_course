//
// Created by Андрей on 03.06.2023.
//
#include "TransformVisitor.h"
#include "frameModel.h"
#include "camera.h"
#include "lightSource.h"

TransformVisitor::TransformVisitor(const TransformParams& transformParams) : _transformParams(transformParams)
{

}

void TransformVisitor::visit(const FrameModel &model)
{
	model._modelStructure->transform(_transformParams);

}
void  TransformVisitor::visit(const Camera &camera)
{
	camera._cameraStructure->transform(_transformParams);
}
Camera::Camera(std::shared_ptr<CameraStructureImp> cameraStructure) : _cameraStructure(cameraStructure)
{

}
void  TransformVisitor::visit(const Composite &composite)
{
	//TODO::You can do wmth here
}

void  TransformVisitor::setParams(const TransformParams& transformParams)
{
	_transformParams = transformParams;
}
void TransformVisitor::visit(const Sphere& sphere)
{

}
void TransformVisitor::visit(const LightSource& lightSorce)
{
	lightSorce->transform(_transformParams);
}


//factory starts here

TransformVisitorFactory::TransformVisitorFactory(const TransformParams & transformParams) : _transformParams(transformParams)
{
}

std::unique_ptr<Visitor> TransformVisitorFactory::create()
{
	return std::make_unique<TransformVisitor>(_transformParams);
}

