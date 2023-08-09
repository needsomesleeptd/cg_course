#ifndef LAB_03_MODELSTRUCTURE_H
#define LAB_03_MODELSTRUCTURE_H

#include <vector>
#include "point.h"
#include "transform.h"
#include "edge.h"

class ModelStructureImp {
public:
	ModelStructureImp() = default;
	ModelStructureImp(std::vector<Point> &points, std::vector<Edge> &edges);
	ModelStructureImp(std::vector<Point> &points, std::vector<Edge> &edges, Point center);
	~ModelStructureImp() = default;

	[[nodiscard]] const std::vector<Point> &getPoints() const;
	[[nodiscard]] const std::vector<Edge> &getEdges() const;
	[[nodiscard]] const Point &getCenter() const;

	void addPoint(const Point &Point);
	void addEdge(const Edge &Edge);
	void transform(const TransformParams &transformParams);

	void setCenter(const Point& center);
private:
	Point _center;
	std::vector<Point>  _points;
	std::vector<Edge> _edges;
};


#endif //LAB_03_MODELSTRUCTURE_H
