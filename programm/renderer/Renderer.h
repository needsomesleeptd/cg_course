//
// Created by Андрей on 09.08.2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_RENDERER_RENDERER_H_
#define LAB_03_CG_COURSE_PROGRAMM_RENDERER_RENDERER_H_

#include "baseRenderer.h"
#include "ImageAdapter.h"
#include "camera.h"
#include "baseLightSource.h"

const float EPS = 1e-7;
const float maxRange = 1e9;

class Renderer : BaseRenderer
{
 public:
	Ray createRay(int x,int y);
	ColorRGB renderPixel(int x, int y, const std::vector<std::shared_ptr<BaseShape>>& shapes) override;
	void renderScene(const std::vector<std::shared_ptr<BaseShape>>& shapes) override;
 private:
	std::shared_ptr<ImageAdapter> image;
	std::shared_ptr<Camera> currentCamera;
	std::shared_ptr<BaseLightSource> currentLightSource;
};

#endif //LAB_03_CG_COURSE_PROGRAMM_RENDERER_RENDERER_H_
