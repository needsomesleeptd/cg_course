#include "loadManagerCreator.h"

void LoadManagerCreator::makeInstance()
{
	static std::shared_ptr<LoadManager> manager(new LoadManager());
	_manager = manager;
}

std::shared_ptr<LoadManager> LoadManagerCreator::createManager()
{
	if (_manager == nullptr)
		makeInstance();
	return _manager;
}

std::shared_ptr<LoadManager> LoadManagerCreator::createManager(const std::shared_ptr<BaseLoadDirector>& director)
{
	if (_manager == nullptr)
		makeInstance();
	_manager->setDirector(director);
	return _manager;
}
