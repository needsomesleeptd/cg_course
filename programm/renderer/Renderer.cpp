//
// Created by Андрей on 09.08.2023.
//

#include "Renderer.h"

ColorRGB Renderer::renderPixel(int x, int y, const std::vector<std::shared_ptr<BaseShape>>& shapes)
{
	Ray tracedRay = createRay(x, y);
	std::shared_ptr<BaseShape> closestShape;
	float t = maxRange;
	for (auto shape : shapes)
	{

		float intersection_t = shape->intersection(tracedRay);
		if (intersection_t > 0 || fabs(intersection_t) < EPS)
		{
			t = std::min(t, intersection_t);
			closestShape = shape;
		}
	}
	if (abs(t - maxRange) < EPS)
	{
		return ColorRGB(0, 0, 0); //Returning background color
	}
	VecD3 intersectionPoint = tracedRay.getPoint(t);
	VecD3 lightVector = normalise(currentLightSource->getPosition() - intersectionPoint);
	VecD3 shapeNormal = closestShape->getNormal(intersectionPoint);

	Material shapeMaterial = closestShape->getMaterial();
	if (shapeMaterial.k_d > 0) //TODO:: replace with get diffuse exc... when tested
	{
		float diffuseIntensivity =
			dot(lightVector, shapeNormal) * shapeMaterial.k_d * currentLightSource->getIntensivity();
		float ambientIntensivity = shapeMaterial.k_a * currentLightSource->getIntensivity();
		return shapeMaterial.color * (diffuseIntensivity + ambientIntensivity);
	}
	else
	{
		return ColorRGB(0, 0, 0); //Returning background color
	}

}
Ray Renderer::createRay(int x, int y)
{
	VecD3 viewPoint = currentCamera->getViewPoint();
	VecD3 l = viewPoint - float(image->getWidth() / 2);
	VecD3 r = viewPoint + float(image->getWidth() / 2);

	VecD3 up = viewPoint - float(image->getHeight() / 2);
	VecD3 down = viewPoint + float(image->getHeight() / 2);

	VecD3 u_deformation = float(x + 0.5) * (r - l) / float(image->getWidth());
	VecD3 v_deformation = float(y + 0.5) * (up - down) / float(image->getHeight());
	VecD3 ray_origin = viewPoint + dot(u_deformation, viewPoint) + dot(v_deformation, viewPoint);
	return Ray(ray_origin, currentCamera->getViewDirection());
}
void Renderer::renderScene(const std::vector<std::shared_ptr<BaseShape>>& shapes)
{
	for (int i = 0; i < image->getWidth(); i++)
	{
		for (int j = 0; j < image->getHeight(); j++)
		{
			ColorRGB pixelColor = renderPixel(i, j, shapes);
			image->setPixelColor(i, j, pixelColor);
		}
	}
}
