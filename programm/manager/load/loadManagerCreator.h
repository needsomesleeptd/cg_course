#ifndef LAB_03_LOADMANAGERCREATOR_H
#define LAB_03_LOADMANAGERCREATOR_H

#include "loadManager.h"

class LoadManagerCreator {
public:
	std::shared_ptr<LoadManager> createManager();
	std::shared_ptr<LoadManager> createManager(const std::shared_ptr<BaseLoadDirector>& director);
private:
	void makeInstance();
	std::shared_ptr<LoadManager> _manager;
};


#endif //LAB_03_LOADMANAGERCREATOR_H
