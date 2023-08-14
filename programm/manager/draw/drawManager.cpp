#include "drawManager.h"
#include "camera.h"


void DrawManager::setCamera(std::shared_ptr<Camera> camera)
{
	_camera = camera;
}


void DrawManager::drawScene(std::shared_ptr<Scene> scene)
{
	//TODO::implement renderer
	//_drawer->clearScene();
	//std::shared_ptr<Visitor> visitor = DrawVisitorFactory(_camera, _drawer).create();
	/*auto objects = scene->getComposite();
	for (auto iterator = objects->begin(); iterator < objects->end(); iterator++)
	{
		auto object = *iterator;
		object->accept(visitor);
	}*/

}
void DrawManager::setRenderer(std::shared_ptr<BaseRenderer> renderer)
{
	_renderer = renderer;
}


