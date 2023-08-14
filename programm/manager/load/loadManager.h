#ifndef LAB_03_LOADMANAGER_H
#define LAB_03_LOADMANAGER_H

#include "baseManager.h"
#include "frameModel.h"
#include "baseLoadDirector.h"
#include "baseBuilder.h"

class LoadManager: BaseManager {
public:
	LoadManager() = default;
	LoadManager(const LoadManager &manager) = delete;
	LoadManager &operator=(const LoadManager &manager) = delete;
	~LoadManager() = default;

	virtual std::shared_ptr<BaseObject> loadModel(std::string &fileName);
	virtual std::shared_ptr<BaseObject> loadCamera(std::string &fileName);

	virtual void setDirector(std::shared_ptr<BaseLoadDirector> director);


private:
	std::shared_ptr<BaseLoadDirector> _director;
};

#endif //LAB_03_LOADMANAGER_H
