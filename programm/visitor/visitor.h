#ifndef LAB_03_VISITOR_H
#define LAB_03_VISITOR_H

#include <memory>

class FrameModel;
class Camera;
class Composite;
class LightSource;
class Sphere;


class Visitor {
public:
	Visitor() = default;
	virtual ~Visitor() = default;

	virtual void visit(const Camera &camera) = 0;
	virtual void visit(const FrameModel &model) = 0;
	virtual void visit(const Composite &composite) = 0;
	virtual void visit(const LightSource &lightSource) = 0;
	virtual void visit(const Sphere &sphere) = 0;
};

class VisitorFactory {
public:
	VisitorFactory() = default;
	virtual ~VisitorFactory() = default;

	virtual std::unique_ptr<Visitor> create() = 0;
};


#endif //LAB_03_VISITOR_H
