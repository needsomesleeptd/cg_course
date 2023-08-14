#ifndef LAB_03_DRAWMANAGERCREATOR_H
#define LAB_03_DRAWMANAGERCREATOR_H

#include "drawManager.h"

class DrawManagerCreator {
public:
	std::shared_ptr<DrawManager> createManager();

private:
	void createInstance();
	std::shared_ptr<DrawManager> _manager;
};

#endif //LAB_03_DRAWMANAGERCREATOR_H
