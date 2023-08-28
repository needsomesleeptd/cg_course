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
	ImageAdapter* imageAdapter = _renderer->renderScene(scene);
	QPixmap pixmap;
	QImage drawingImage(600, 600,QImage::Format_RGB32);
	for (int i = 0; i < imageAdapter->_width; i++)
		for (int j = 0; j < imageAdapter->_height; j++)
		{
			ColorRGB color = imageAdapter->colorMatrix[i * imageAdapter->_width + j];
			QColor qcolor;
			qcolor.setRedF(color.R);
			qcolor.setGreenF(color.G);
			qcolor.setBlueF(color.B);
			drawingImage.setPixelColor(i,j,qcolor);
		}


	pixmap.convertFromImage(drawingImage);
	_drawingScene->addPixmap(pixmap);


}
void DrawManager::setRenderer(std::shared_ptr<BaseRenderer> renderer)
{
	_renderer = renderer;
}
void DrawManager::setDrawingScene(QGraphicsScene* drawingScene)
{
	_drawingScene = drawingScene;
}


