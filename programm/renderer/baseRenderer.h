//
// Created by Андрей on 14.08.2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_RENDERER_BASERENDERER_H_
#define LAB_03_CG_COURSE_PROGRAMM_RENDERER_BASERENDERER_H_
#include "ray.h"
#include "color.h"
#include "baseShape.h"
class BaseRenderer
{
 public:
	virtual ColorRGB renderPixel(int x,int y, Ray ray,const std::vector<BaseShape>& shapes) = 0;
	virtual ~BaseRenderer() = default;
};
#endif //LAB_03_CG_COURSE_PROGRAMM_RENDERER_BASERENDERER_H_
