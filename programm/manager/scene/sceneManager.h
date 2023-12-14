#ifndef LAB_03_SCENEMANAGER_H
#define LAB_03_SCENEMANAGER_H

#include "baseManager.h"
#include "scene.h"

class SceneManager : public BaseManager {
public:
	SceneManager();
	SceneManager(const SceneManager &scene) = delete;
	SceneManager &operator = (const SceneManager &scene)  = delete;
	~SceneManager() = default;

	void setScene(std::shared_ptr<Scene> scene);
	void setCamera(const std::size_t &cameraIndex);

	[[nodiscard]] std::shared_ptr<Scene> getScene() const;
	[[nodiscard]] std::shared_ptr<Camera> getCamera() const;

private:
	std::shared_ptr<Camera> _camera;
	std::shared_ptr<Scene>  _scene;
};


#endif //LAB_03_SCENEMANAGER_H
