#include "modelStructure.h"

ModelStructureImp::ModelStructureImp(std::vector<Point> &points, std::vector<Edge> &edges) :
		_center{}, _points(points), _edges(edges) {}

ModelStructureImp::ModelStructureImp(std::vector<Point> &points, std::vector<Edge> &edges, Point center):
		_center(center), _points(points), _edges(edges) {}

const std::vector<Point> &ModelStructureImp::getPoints() const {
	return _points;
}

const std::vector<Edge> &ModelStructureImp::getEdges() const {
	return _edges;
}

const Point &ModelStructureImp::getCenter() const {
	return _center;
}

void ModelStructureImp::addPoint(const Point &point) {
	_points.push_back(point);
}

void ModelStructureImp::addEdge(const Edge &edge) {
	_edges.push_back(edge);
}

void ModelStructureImp::transform(const TransformParams &transformParams) {
	const Point moveParams = transformParams.getMoveParams();
	const Point scaleParams = transformParams.getScaleParams();
	const Point rotateParams = transformParams.getRotateParams();
	_center.move(moveParams.getX(), moveParams.getY(), moveParams.getZ());
	for (auto &point : _points) {
		point.scale(scaleParams.getX(),  scaleParams.getY(),  scaleParams.getZ());
		point.rotate(rotateParams.getX(), rotateParams.getY(), rotateParams.getZ());
	}
}
void ModelStructureImp::setCenter(const Point& center)
{
	_center = center;
}
