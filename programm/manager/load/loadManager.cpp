#include "loadManager.h"
#include "config.h"

std::shared_ptr<BaseObject> LoadManager::loadModel(std::string &fileName) {

	auto director = DirectorSolution().createCreator("model")->getDirector(fileName);
	setDirector(director);
	return director->create();
}

std::shared_ptr<BaseObject> LoadManager::loadCamera(std::string &fileName) {

	auto director = DirectorSolution().createCreator("camera")->getDirector(fileName);
	setDirector(director);
	return director->create();
}


void LoadManager::setDirector(std::shared_ptr<BaseLoadDirector> director) {
	_director = director;
}



