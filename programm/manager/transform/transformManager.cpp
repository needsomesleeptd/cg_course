#include "transformManager.h"
#include "frameModel.h"
#include "camera.h"
#include "TransformVisitor.h"


void TransformManager::moveObject(const std::shared_ptr <BaseObject> &object, const double &dx, const double &dy,const double &dz) {
	Point move(dx, dy,dz);
	TransformParams params;
	params.setMoveParams(move);
	std::shared_ptr<Visitor> visitor = std::make_shared<TransformVisitor>(params);
	object->accept(visitor);
}

void TransformManager::scaleObject(const std::shared_ptr <BaseObject> &object, const double &kx, const double &ky,const double &kz) {
	Point scale(kx, ky,kz);
	TransformParams params;
	params.setScaleParams(scale);
	std::shared_ptr<Visitor> visitor = std::make_shared<TransformVisitor>(params);
	object->accept(visitor);
}

void TransformManager::rotateObject(const std::shared_ptr <BaseObject> &object, const double &ox, const double &oy,const double &oz) {
	Point rotate(ox, oy,oz);
	TransformParams params;
	params.setRotateParams(rotate);
	std::shared_ptr<Visitor> visitor = std::make_shared<TransformVisitor>(params);
	object->accept(visitor);
}

void TransformManager::transformObject(const std::shared_ptr<BaseObject> &object, const TransformParams &transformParams) {
	std::shared_ptr<Visitor> visitor = TransformVisitorFactory(transformParams).create();
	object->accept(visitor);
}



void TransformManager::setParams(const TransformParams transformParams) {
	_transformParams = transformParams;
}


