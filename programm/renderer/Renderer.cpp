//
// Created by Андрей on 09.08.2023.
//

#include "Renderer.h"
ColorRGB Renderer::renderPixel(int x, int y, const std::vector<std::shared_ptr<BaseShape>>& shapes)
{
	Ray tracedRay = createRay(x,y);
	ColorRGB color;
	for (auto shape:shapes)
	{

		float t = 1e9;
		float intersection_t = shape->intersection(tracedRay);
		if (intersection_t > 0 || fabs(intersection_t) < EPS)
		{
			t = std::min(t, intersection_t);
			//color = shape;
		}
	}

}
Ray Renderer::createRay(int x, int y)
{
	VecD3 viewPoint = currentCamera->getViewPoint();
	VecD3 l =viewPoint - float(image->getWidth() / 2);
	VecD3 r = viewPoint + float(image->getWidth() / 2);

	VecD3 up = viewPoint - float(image->getHeight() / 2);
	VecD3 down = viewPoint + float(image->getHeight() / 2);



	VecD3 u_deformation = float(x + 0.5) * (r - l) / float(image->getWidth());
	VecD3 v_deformation = float(y + 0.5) * (up - down) / float(image->getHeight());
	VecD3 ray_origin = viewPoint + dot(u_deformation,viewPoint) + dot(v_deformation,viewPoint);
	return Ray(ray_origin,currentCamera->getViewDirection());
}
