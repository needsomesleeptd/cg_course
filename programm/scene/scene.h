#ifndef LAB_03_SCENE_H
#define LAB_03_SCENE_H

#include <vector>
#include "object.h"
#include "baseCamera.h"

class Scene
{
 public:
	Scene();
	~Scene() = default;

	std::vector<std::shared_ptr<BaseObject>> getModels();
	std::vector<std::shared_ptr<Camera>> getCameras();
	std::shared_ptr<Composite> getComposite();

	void addModel(const std::shared_ptr<BaseObject>& model);
	void addCamera(const std::shared_ptr<Camera>& camera);

	void removeModel(const std::size_t index);
	void removeCamera(const std::size_t index);

	void setCamera(std::size_t index);
	[[nodiscard]] std::shared_ptr<Camera> getCamera() const;

 protected:


	std::vector<std::shared_ptr<Camera>> _cameras;
	std::shared_ptr<Composite> _models;
	std::size_t _currCameraIdx;
};

#endif //LAB_03_SCENE_H
