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
		return backGround; //Returning background color
	}
	std::shared_ptr<BaseLightSource> currentLightSource = scene->getLightSource();
	VecD3 intersectionPoint = tracedRay.getPoint(t);
	VecD3 lightVector = normalise(intersectionPoint - currentLightSource->getPosition());
	VecD3 shapeNormal = normalise(closestShape->getNormal(intersectionPoint));

	Material shapeMaterial = closestShape->getMaterial();
	float dotLight = dot(shapeNormal, lightVector);
	if (dotLight > 0) //TODO:: replace with get diffuse exc... when tested
	{
		float diffuseIntensivity = dotLight * shapeMaterial._k_d * currentLightSource->getIntensivity();
		float ambientIntensivity = shapeMaterial._k_a * currentLightSource->getIntensivity();
		return shapeMaterial._color * (diffuseIntensivity + ambientIntensivity);
	}
	else
	{
		return backGround; //Returning background color
	}
}
void rayTrace(const Ray& tracedRay, ColorRGB& finalColor, std::shared_ptr<Scene> scene,int curDepth)
{
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
		return; //Returning background color
	}
	std::shared_ptr<BaseLightSource> currentLightSource = scene->getLightSource();
	VecD3 intersectionPoint = tracedRay.getPoint(t);
	VecD3 lightVector = normalise(intersectionPoint - currentLightSource->getPosition());
	VecD3 shapeNormal = normalise(closestShape->getNormal(intersectionPoint));

	Material shapeMaterial = closestShape->getMaterial();
	float ambientIntensivity = shapeMaterial._k_a * currentLightSource->getIntensivity();
	finalColor = shapeMaterial._color * ambientIntensivity;
	float dotLight = dot(shapeNormal, lightVector);
	if (shapeMaterial._k_d > 0)
	{
		if (dotLight > 0)
		{
			float diffuseIntensivity = dotLight * shapeMaterial._k_d * currentLightSource->getIntensivity();
			float ambientIntensivity = shapeMaterial._k_a * currentLightSource->getIntensivity();
			finalColor = shapeMaterial._color * diffuseIntensivity + finalColor;
		}
	}
	if (shapeMaterial._k_s > 0)
	{
		if (curDepth < maxDepth)
		{
			Ray reflected = tracedRay.calculateReflected(shapeNormal, intersectionPoint);
			if (dotLight > 0)
			{

				finalColor = shapeMaterial._color * shapeMaterial._k_s + finalColor;
				rayTrace(reflected, finalColor, scene, curDepth + 1);
			}
		}
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
	/*VecD3 ray_origin = viewPoint + u_deformation * VecD3(1, 0, 0) + v_deformation * VecD3(0, 1, 0);
	return Ray(ray_origin, -currentCamera->getViewDirection());*/
	glm::vec2 coord = { (float)x / (float)imageWidth, (float)y / (float)imageWidth };
	coord = coord * 2.0f - 1.0f; // -1 -> 1

	glm::vec4 target = 1.0f * glm::vec4(coord.x, coord.y, 1, 1);
	glm::vec3
		rayDirection = glm::vec3(1.0f * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0)); // World space
	return Ray(viewPoint, rayDirection);
	/*VecD3 dir =VecD3(4*x,3*y,0) - viewPoint;
	return Ray(viewPoint,dir);*/
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
