#include "sceneManager.h"

SceneManager::SceneManager() {
	_scene = std::make_shared<Scene>();
}

void SceneManager::setScene(std::shared_ptr<Scene> scene) {
	_scene = std::move(scene);
}

void SceneManager::setCamera(const size_t &cameraIndex) {
	_camera = _scene->getCameras().at(cameraIndex);
}

std::shared_ptr<Scene> SceneManager::getScene() const {
	return _scene;
}

std::shared_ptr<Camera> SceneManager::getCamera() const {
	return _camera;
}
