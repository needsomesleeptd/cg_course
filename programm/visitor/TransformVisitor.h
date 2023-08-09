//
// Created by Андрей on 03.06.2023.
//


#ifndef LAB_03_VISITOR_TRANSFORMVISITOR_H_
#define LAB_03_VISITOR_TRANSFORMVISITOR_H_

#include "object.h"
#include "visitor.h"
#include "transform.h"
#include "sphere.h"

class TransformVisitor : public Visitor
{
 public:
	TransformVisitor() = default;
	TransformVisitor(const TransformParams& transformParams);
	void visit(const FrameModel& model) override;
	void visit(const Camera& camera) override;
	void visit(const Composite& composite) override;
	void visit(const Sphere& sphere) override;
	void visit(const LightSource& lightSource) override;
	void setParams(const TransformParams& transformParams);


 private:
	TransformParams _transformParams;
};


class TransformVisitorFactory : public VisitorFactory
{
 public:
	TransformVisitorFactory(const TransformParams &transformParams);
	~TransformVisitorFactory() override = default;

	std::unique_ptr<Visitor> create() override;

 private:
	TransformParams _transformParams;
};






#endif //LAB_03_VISITOR_TRANSFORMVISITOR_H_
