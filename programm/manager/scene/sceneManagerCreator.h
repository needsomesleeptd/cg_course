#ifndef LAB_03_SCENEMANAGERCREATOR_H
#define LAB_03_SCENEMANAGERCREATOR_H

#include <memory>
#include "sceneManager.h"

class SceneManagerCreator {
public:
	std::shared_ptr<SceneManager> createManager();

private:
	void createInstance();
	std::shared_ptr<SceneManager> _manager;
};

#endif //LAB_03_SCENEMANAGERCREATOR_H
