#include "frameModel.h"

FrameModel::FrameModel(const FrameModel &model) {
	_modelStructure = model._modelStructure;
}


void FrameModel::accept(std::shared_ptr<Visitor> visitor) {
	visitor->visit(*this);
}
