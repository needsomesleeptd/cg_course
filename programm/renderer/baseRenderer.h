//
// Created by Андрей on 14.08.2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_RENDERER_BASERENDERER_H_
#define LAB_03_CG_COURSE_PROGRAMM_RENDERER_BASERENDERER_H_
#include "ray.h"
#include "color.h"
#include "baseShape.h"
#include "scene.h"
#include "ImageAdapter.h"
class BaseRenderer
{
 public:
	__device__ virtual ColorRGB renderPixel(int x, int y, Scene* scene, Camera* camera) = 0;
	__host__ 	virtual void renderScene(std::shared_ptr<Scene> scene) = 0;
	virtual ~BaseRenderer() = default;
	__device__ virtual void rayTrace(const Ray& tracedRay, ColorRGB& finalColor, Scene* scene,int curDepth) = 0;
	 virtual void drawImage(ImageAdapter* image) = 0;
};
#endif //LAB_03_CG_COURSE_PROGRAMM_RENDERER_BASERENDERER_H_
