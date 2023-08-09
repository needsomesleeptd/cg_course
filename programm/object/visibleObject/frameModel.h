#ifndef LAB_03_FRAMEMODEL_H
#define LAB_03_FRAMEMODEL_H

#include <memory>
#include <utility>
#include "point.h"
#include "edge.h"
#include "modelStructure.h"
#include "visitor.h"
#include "drawManager.h"
#include "transformManager.h"
#include "TransformVisitor.h"
#include "DrawVisitor.h"
#include "baseShape.h"

class FrameModel: public VisibleObject {
public:
	friend void DrawVisitor::visit(const FrameModel &model);
	friend void TransformVisitor::visit(const FrameModel &model);



	FrameModel(const FrameModel &model);
	explicit FrameModel(std::shared_ptr<BaseShape> modelStructure): _modelStructure(std::move(modelStructure)) {}
	~FrameModel() override = default;


	void accept(std::shared_ptr<Visitor> visitor) override;

protected:
	std::shared_ptr<BaseShape> _modelStructure;
};

#endif //LAB_03_FRAMEMODEL_H
