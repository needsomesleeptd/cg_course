//
// Created by Андрей on 09.08.2023.
//

#include "Renderer.h"
#include "scene.h"

#include <iostream>
ColorRGB Renderer::renderPixel(int x, int y, std::shared_ptr<Scene> scene)
{
	Ray tracedRay = createRay(x, y, scene->getCamera());
	std::shared_ptr<BaseShape> closestShape;
	float t = maxRange;
	for (auto shape : scene->getModels())
	{

		float intersection_t = shape->intersection(tracedRay);
		std::cout << intersection_t << std::endl;
		if (intersection_t > 0 || fabs(intersection_t) < EPS)
		{
			t = std::min(t, intersection_t);
			closestShape = std::dynamic_pointer_cast<BaseShape>(shape);
		}
	}
	if (abs(t - maxRange) < EPS)
	{
		return ColorRGB(0, 0, 0); //Returning background color
	}
	std::shared_ptr<BaseLightSource> currentLightSource = scene->getLightSource();
	VecD3 intersectionPoint = tracedRay.getPoint(t);
	VecD3 lightVector = normalise(currentLightSource->getPosition() + intersectionPoint);
	VecD3 shapeNormal = normalise(closestShape->getNormal(intersectionPoint));

	Material shapeMaterial = closestShape->getMaterial();
	float dotLight = dot(lightVector, shapeNormal);
	if (dotLight < 0) //TODO:: replace with get diffuse exc... when tested
	{
		float diffuseIntensivity = -1 * dotLight * shapeMaterial._k_d * currentLightSource->getIntensivity();
		float ambientIntensivity = shapeMaterial._k_a * currentLightSource->getIntensivity();
		return shapeMaterial._color * (diffuseIntensivity + ambientIntensivity);
	}
	else
	{
		return ColorRGB(0, 0, 0); //Returning background color
	}

}
Ray Renderer::createRay(int x, int y, std::shared_ptr<Camera> currentCamera)
{
	float imageHeight = _scene->height();
	float imageWidth = _scene->width();
	VecD3 viewPoint = currentCamera->getViewPoint();
	VecD3 l = viewPoint - float(imageWidth / 2);
	VecD3 r = viewPoint + float(imageWidth / 2);

	VecD3 up = viewPoint - float(imageHeight / 2);
	VecD3 down = viewPoint + float(imageHeight / 2);

	VecD3 u_deformation = float(x + 0.5) * (r - l) / float(imageWidth);
	VecD3 v_deformation = float(y + 0.5) * (up - down) / float(imageHeight);//TODO::fix ray origin
	VecD3
		ray_origin = viewPoint + dot(u_deformation, currentCamera->getUpVector()) + dot(v_deformation, VecD3(0, 0, 1));
	return Ray(ray_origin, -currentCamera->getViewDirection());
}
void Renderer::renderScene(std::shared_ptr<Scene> scene)
{
	std::shared_ptr<ImageAdapter> image = std::make_shared<ImageAdapter>(_scene->width(), _scene->height());
	auto objects = scene->getModels();//TODO::remove dynamic casting
	for (int i = 0; i < image->getWidth(); i++)
	{
		for (int j = 0; j < image->getHeight(); j++)
		{
			ColorRGB pixelColor = renderPixel(i, j, scene);
			image->setPixelColor(i, j, pixelColor);
		}
	}
	QPixmap pixmap;
	pixmap.convertFromImage(*image->getImage());
	_scene->addPixmap(pixmap);
}

Renderer::Renderer(QGraphicsScene* scene)
{
	_scene = scene;
}
