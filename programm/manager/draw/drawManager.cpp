#include "drawManager.h"
#include "camera.h"


void DrawManager::setCamera(std::shared_ptr<Camera> camera)
{
	_camera = camera;
}


void DrawManager::drawScene(std::shared_ptr<Scene> scene)
{

	//_drawer->clearScene();
	//std::shared_ptr<Visitor> visitor = DrawVisitorFactory(_camera, _drawer).create();
	//auto objects = scene->getComposite();
	_renderer->renderScene(scene);

}
void DrawManager::setRenderer(std::shared_ptr<BaseRenderer> renderer)
{
	_renderer = renderer;
}


