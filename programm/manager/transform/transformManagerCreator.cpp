#include "transformManagerCreator.h"

void TransformManagerCreator::createInstance() {
	static std::shared_ptr<TransformManager> manager(new TransformManager());
	_manager = manager;
}


std::shared_ptr<TransformManager> TransformManagerCreator::createManager()
{
	if (_manager == nullptr)
		createInstance();
	return _manager;
}
