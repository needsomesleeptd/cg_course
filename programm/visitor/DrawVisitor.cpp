//
// Created by Андрей on 03.06.2023.
//

#include "DrawVisitor.h"
#include "glmWrapper.h"
#include "point.h"
#include "frameModel.h"
#include "camera.h"

DrawVisitor::DrawVisitor(std::shared_ptr<Camera> camera, std::shared_ptr<AbstractDrawer> drawer) : _camera(camera), _drawer(drawer)
{

}


void DrawVisitor::setCamera(std::shared_ptr<Camera> camera) {
	_camera = camera;
}

void DrawVisitor::setDrawer(std::shared_ptr<AbstractDrawer> drawer) {
	_drawer = drawer;
}


Vector4 pointToVector(const Point point) {
	return  Vector4(point.getX(), point.getY(), point.getZ(), 0);
}


Point DrawVisitor::getProection(const Point &point) {
	Point proection = point;
	proection.setX(proection.getX() + _camera->_cameraStructure->getCoordinates().getX());
	proection.setY(proection.getY() + _camera->_cameraStructure->getCoordinates().getY());
	return proection;
}



void DrawVisitor::visit(const FrameModel &model) {
	/*auto points = model._modelStructure->getPoints();
	auto edges = model._modelStructure->getEdges();
	auto center = model._modelStructure->getCenter();
	Matrix4 view = _camera->_cameraStructure->getView();
	Matrix4 projection = _camera->_cameraStructure->getProjection();
	Matrix4 matrix_changes = projection * (view * (Matrix4(1)));
	for (auto edge : edges) {
		Vector4 vectorFrom = matrix_changes  * pointToVector(points.at(edge.getStartIndex()));
		Vector4 vectorTo = matrix_changes  * pointToVector(points.at(edge.getEndIndex()));
		Point pointFrom = getProection(Point{vectorFrom[0],vectorFrom[1],vectorFrom[2]}).addCenter(center);
		Point pointTo = getProection({vectorTo[0], vectorTo[1], vectorTo[2]}).addCenter(center);
		_drawer->drawLine(pointFrom, pointTo);
	}*/

}

void DrawVisitor::visit(const Camera &camera) {}

void DrawVisitor::visit(const Composite &composite) {}

//Factory starts here


DrawVisitorFactory::DrawVisitorFactory(std::shared_ptr<Camera>& cameraArg, std::shared_ptr<AbstractDrawer>& drawerArg)
{
	drawer = drawerArg;
	camera = cameraArg;
}

std::unique_ptr<Visitor> DrawVisitorFactory::create()
{
	return std::make_unique<DrawVisitor>(camera, drawer);
}
