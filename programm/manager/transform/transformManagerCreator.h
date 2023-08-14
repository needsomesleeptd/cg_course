#ifndef LAB_03_TRANSFORMMANAGERCREATOR_H
#define LAB_03_TRANSFORMMANAGERCREATOR_H

#include "transformManager.h"

class TransformManagerCreator {
public:
	std::shared_ptr<TransformManager> createManager();

private:
	void createInstance();
	std::shared_ptr<TransformManager> _manager;
};

#endif //LAB_03_TRANSFORMMANAGERCREATOR_H
