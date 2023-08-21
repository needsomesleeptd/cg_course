//
// Created by Андрей on 09.08.2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_RENDERER_RENDERER_H_
#define LAB_03_CG_COURSE_PROGRAMM_RENDERER_RENDERER_H_

#include "baseRenderer.h"
#include "camera.h"
#include "baseLightSource.h"
#include "scene.h"
#include <QGraphicsScene>


const float EPS = 1e-7;
const float maxRange = 1e9;
const int maxDepth = 12;
const ColorRGB backGround = ColorRGB(0,0,0);


class Renderer : public BaseRenderer
{
 public:
	explicit Renderer(QGraphicsScene *scene);
	Ray createRay(int x, int y, std::shared_ptr<Camera> currentCamera);
	__device__	ColorRGB renderPixel(int x, int y, std::shared_ptr<Scene> scene) override;
	__device__ void renderScene(std::shared_ptr<Scene> scene) override;
	__device__ void rayTrace(const Ray& tracedRay, ColorRGB& finalColor, std::shared_ptr<Scene> scene,int curDepth) override;
	__device__ void drawImage(std::shared_ptr<ImageAdapter> image);
 private:
	QGraphicsScene *_scene;
};

#endif //LAB_03_CG_COURSE_PROGRAMM_RENDERER_RENDERER_H_
