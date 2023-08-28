//
// Created by Андрей on 09.08.2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_RENDERER_RENDERER_CUH_
#define LAB_03_CG_COURSE_PROGRAMM_RENDERER_RENDERER_CUH_

#include "baseRenderer.h"
#include "camera.h"
#include "baseLightSource.h"
#include "scene.h"
#include "../scene/scene.h"
#include "../object/invisibleObject/camera.h"
#include "../color/color.h"
#include "../image_adapter/ImageAdapter.h"
//#include "QGraphicsScene"
#include "CudaShape.h"

const float EPS = 1e-7;
const float maxRange = 1e9;
const int maxDepth = 12;
const ColorRGB backGround = ColorRGB(0, 0, 0);

class Renderer : public BaseRenderer
{
 public:
	//explicit Renderer(QGraphicsScene* cene);
	__device__ Ray createRay(int x, int y, Camera* currentCamera);
	__device__ ColorRGB renderPixel(int x, int y, Scene* scene, Camera* camera) override;
	ImageAdapter* renderScene(std::shared_ptr<Scene> scene) override;
	__device__ void rayTrace(const Ray& tracedRay, ColorRGB& finalColor, Scene* scene, int curDepth) override;
	void getImage(ImageAdapter* image);
 //private:
	//QGraphicsScene* _scene;

};

ColorRGB rayTrace(const Ray& tracedRay,
	ColorRGB& otherColor,
	Scene* scene,
	int curDepth, CudaArray<CudaShape*> objects,
	LightSource* lightSource);

__device__ ColorRGB renderPixel(int x,
	int y,
	Scene* scene,
	Camera* camera,
	CudaArray<CudaShape*> objects,
	LightSource* lightSource,
	ImageAdapter* image);

__global__ void renderSceneCuda(Scene* scene,
	Camera* camera,
	Renderer* renderer,
	CudaArray<CudaShape*> objects,
	LightSource* lightSource,
	ImageAdapter* image);
#endif //LAB_03_CG_COURSE_PROGRAMM_RENDERER_RENDERER_CUH_
