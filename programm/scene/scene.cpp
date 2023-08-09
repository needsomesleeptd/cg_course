#include <iterator>
#include "scene.h"
#include "composite.h"

Scene::Scene() : _models(new Composite) {};

std::vector<std::shared_ptr<BaseObject>> Scene::getModels() {
	return _models->_elements;
}

std::vector<std::shared_ptr<Camera>> Scene::getCameras() {
	return _cameras;
}

std::shared_ptr<Composite> Scene::getComposite() {
	return _models;
}

void Scene::addModel(const std::shared_ptr<BaseObject> &model) {
	_models->add(model);
}

void Scene::removeModel(const std::size_t index) {
	auto iter = _models->begin();
	std::advance(iter, index);
	_models->remove(iter);
}

void Scene::addCamera(const std::shared_ptr<Camera>& camera) {
	_cameras.push_back(camera);
}

void Scene::removeCamera(const std::size_t index) {
	auto iter = _cameras.begin();
	std::advance(iter, index);
	_cameras.erase(iter);
}

void Scene::setCamera(std::size_t index) {
	_currCameraIdx = index;
}

std::shared_ptr<Camera> Scene::getCamera() const {
	return _cameras.at(_currCameraIdx);
}
