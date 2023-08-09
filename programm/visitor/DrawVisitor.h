//
// Created by Андрей on 03.06.2023.
//

#ifndef LAB_03_VISITOR_DRAWVISITOR_H_
#define LAB_03_VISITOR_DRAWVISITOR_H_

#include "visitor.h"
#include "point.h"
#include "baseCamera.h"

class AbstractDrawer;
class Camera;

class DrawVisitor : public Visitor
{
 public:
	DrawVisitor() = default;
	DrawVisitor(std::shared_ptr<Camera> camera, std::shared_ptr<AbstractDrawer> drawer);
	void visit(const FrameModel& model) override;
	void visit(const Camera& camera) override;
	void visit(const Composite& composite) override;
	void setCamera(std::shared_ptr<Camera> camera);
	void setDrawer(std::shared_ptr<AbstractDrawer> camera);
 protected:
	Point getProection(const Point& point);

 private:
	std::shared_ptr<Camera> _camera;
	std::shared_ptr<AbstractDrawer> _drawer;
};

class DrawVisitorFactory : public VisitorFactory
{
 public:
	DrawVisitorFactory(std::shared_ptr<Camera>& cameraArg, std::shared_ptr<AbstractDrawer>& drawerArg);
	~DrawVisitorFactory() override = default;

	std::unique_ptr<Visitor> create() override;

 private:
	std::shared_ptr<AbstractDrawer> drawer;
	std::shared_ptr<Camera> camera;
};

#endif //LAB_03_VISITOR_DRAWVISITOR_H_
