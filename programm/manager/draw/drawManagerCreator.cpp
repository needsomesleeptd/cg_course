#include "drawManagerCreator.h"

std::shared_ptr<DrawManager> DrawManagerCreator::createManager() {
	if (_manager == nullptr)
		createInstance();
	return _manager;
}

void DrawManagerCreator::createInstance() {
	static std::shared_ptr<DrawManager> manager(new DrawManager());
	_manager = manager;
}
