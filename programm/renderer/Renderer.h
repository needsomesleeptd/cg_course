//
// Created by Андрей on 09.08.2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_RENDERER_RENDERER_H_
#define LAB_03_CG_COURSE_PROGRAMM_RENDERER_RENDERER_H_

#include "baseRenderer.h"

class Renderer : BaseRenderer
{
 private:
	ColorRGB renderPixel(int x,int y, Ray ray,const std::vector<BaseShape>& shapes) override;
};

#endif //LAB_03_CG_COURSE_PROGRAMM_RENDERER_RENDERER_H_
