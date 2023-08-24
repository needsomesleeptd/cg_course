//
// Created by Андрей on 09.08.2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_RENDERER_RENDERER_H_
#define LAB_03_CG_COURSE_PROGRAMM_RENDERER_RENDERER_H_

#include "baseRenderer.h"
#include "camera.h"
#include "baseLightSource.h"
#include "scene.h"
#include "../scene/scene.h"
#include "../object/invisibleObject/camera.h"
#include "../color/color.h"
#include "../image_adapter/ImageAdapter.h"
#include <QGraphicsScene>

const float EPS = 1e-7;
const float maxRange = 1e9;
const int maxDepth = 12;
const ColorRGB backGround = ColorRGB(0, 0, 0);

class Renderer : public BaseRenderer
{
 public:
	explicit Renderer(QGraphicsScene* scene);
	__device__ Ray createRay(int x, int y, Camera* currentCamera);
	__device__    ColorRGB renderPixel(int x, int y, Scene* scene, Camera* camera) override;
	__host__  void renderScene(std::shared_ptr<Scene> scene) override;
	__device__ void rayTrace(const Ray& tracedRay, ColorRGB& finalColor, Scene* scene, int curDepth) override;
	void drawImage(ImageAdapter* image);
 private:
	QGraphicsScene* _scene;
};

__device__  void rayTrace(const Ray& tracedRay,
	ColorRGB& finalColor,
	Scene* scene,
	int curDepth, thrust::device_vector<BaseShape*> objects,
	BaseLightSource* lightSource);

__device__ ColorRGB renderPixel(int x,
	int y,
	Scene* scene,
	Camera* camera,
	thrust::device_vector<BaseObject*> objects,
	BaseLightSource* lightSource);

__global__ void renderSceneCuda(Scene* scene,
	Camera* camera,
	BaseRenderer* renderer,
	ImageAdapter* image,
	thrust::device_vector<BaseObject*> objects, BaseLightSource* lightSource);
#endif //LAB_03_CG_COURSE_PROGRAMM_RENDERER_RENDERER_H_
