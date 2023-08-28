#ifndef LAB_03_DRAWMANAGER_H
#define LAB_03_DRAWMANAGER_H

#include "baseManager.h"
#include "point.h"
#include "composite.h"
#include "object.h"
#include "visitor.h"
#include "baseRenderer.h"
#include <QGraphicsScene>
class Camera;

class DrawManager: public BaseManager {
public:
	DrawManager() = default;
	DrawManager(const DrawManager &manager) = delete;
	DrawManager &operator = (const DrawManager &manager) = delete;
	~DrawManager() = default;

	void setCamera(std::shared_ptr<Camera> camera);
	void setRenderer(std::shared_ptr<BaseRenderer> renderer);
	void setDrawingScene(QGraphicsScene * drawingScene);
	void drawScene(std::shared_ptr<Scene> scene);





private:
	std::shared_ptr<Camera> _camera;
	std::shared_ptr<BaseRenderer> _renderer;
	QGraphicsScene* _drawingScene;

};


#endif //LAB_03_DRAWMANAGER_H
