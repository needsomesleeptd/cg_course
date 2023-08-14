//
// Created by Андрей on 03.06.2023.
//


#ifndef LAB_03_VISITOR_TRANSFORMVISITOR_H_
#define LAB_03_VISITOR_TRANSFORMVISITOR_H_

#include "visitor.h"
#include "LightSource.h"
#include "transform.h"

class TransformVisitor : public Visitor
{
 public:
	TransformVisitor() = default;
	TransformVisitor(const TransformParams& transformParams);

	void visit(const Camera& camera) override;
	void visit(const Composite& composite) override;
	void visit(const Sphere& sphere) override;
	void visit(LightSource& lightSorce) const override;
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
