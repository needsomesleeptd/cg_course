//
// Created by Андрей on 09.08.2023.
//

#include "Renderer.h"
#include "scene.h"

#include <iostream>
#include <QGraphicsItem>
#include <QPixmap>


#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


ColorRGB Renderer::renderPixel(int x, int y, std::shared_ptr<Scene> scene)
{

	Ray tracedRay = createRay(x, y, scene->getCamera());
	ColorRGB finalColor = backGround;
	rayTrace(tracedRay, finalColor, scene, 0);
	return finalColor;
}

bool hasIntersection(const Ray& tracedRay, std::shared_ptr<Scene> scene)
{
	for (auto shape : scene->getModels())
	{

		float intersection_t = shape->intersection(tracedRay);
		if (intersection_t > 0.0f)
			return false;
		if (intersection_t < EPS && intersection_t > 0) // due to self intersection
			return false;
		else if (intersection_t > 0)
			return true;
	}
	return false;
}

void Renderer::rayTrace(const Ray& tracedRay, ColorRGB& finalColor, std::shared_ptr<Scene> scene, int curDepth)
{
	std::shared_ptr<BaseShape> closestShape;
	float t = maxRange;
	for (auto shape : scene->getModels())
	{

		float intersection_t = shape->intersection(tracedRay);
		//std::cout << intersection_t << std::endl;
		if (intersection_t > 0 || abs(intersection_t) < EPS)
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

	Ray lightRay(intersectionPoint, lightVector);
	if (hasIntersection(lightRay, scene))
		return;

	VecD3 shapeNormal = normalise(closestShape->getNormal(intersectionPoint));

	Material shapeMaterial = closestShape->getMaterial();
	float ambientIntensivity = shapeMaterial._k_a * currentLightSource->getIntensivity();
	finalColor = shapeMaterial._color * ambientIntensivity + finalColor;
	float diffuseLight = dot(shapeNormal, lightVector);
	if (shapeMaterial._k_d > 0)
	{
		if (diffuseLight > 0)
		{
			//std::cout << " diffuseLight" << diffuseLight << std::endl;
			ColorRGB diffuseColorRay = currentLightSource->getColor() * diffuseLight * shapeMaterial._k_d;
			finalColor = shapeMaterial._color * diffuseColorRay + finalColor;

		}
	}
	if (shapeMaterial._k_s > 0)
	{

		Ray reflected = tracedRay.calculateReflected(shapeNormal, intersectionPoint);
		float specularDot = pow(dot(reflected.D, tracedRay.D),20);
		//std::cout << " diffuseLight" << diffuseLight << std::endl;
		if (specularDot > 0.0)
		{
			//float spec = powf( specularDot, 20 ) * shapeMaterial._k_s;
			finalColor = currentLightSource->getColor() * specularDot * shapeMaterial._k_s + finalColor;
		}
	}
	if (shapeMaterial._k_s > 0.0f)
	{
		VecD3 N = closestShape->getNormal(intersectionPoint);
		Ray reflected = tracedRay.calculateReflected(shapeNormal, intersectionPoint);
		if (curDepth < maxDepth)
		{
			ColorRGB rcol(0, 0, 0);
			rayTrace(reflected, rcol, scene, curDepth + 1);
			finalColor = rcol * shapeMaterial._k_s * closestShape->getMaterial()._color + finalColor;
		}
	}
	float rayLength = (intersectionPoint - tracedRay.E).length();
	finalColor = finalColor / (rayLength + k);
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

	GLuint tbo;


	//VecD3 u_deformation = float(x) * (r - l) / float(imageWidth);
	//VecD3 v_deformation = float(y) * (up - down) / float(imageHeight);//TODO::fix ray origin
	/*VecD3 ray_origin = viewPoint + u_deformation * VecD3(1, 0, 0) + v_deformation * VecD3(0, 1, 0);
	return Ray(ray_origin, -currentCamera->getViewDirection());*/
	glm::vec2 coord = { (float)x / (float)imageWidth, (float)y / (float)imageWidth };
	coord = coord * 2.0f - 1.0f; // -1 -> 1

	glm::vec4 target = currentCamera->getInverseProjectionMatrix() * glm::vec4(coord.x, coord.y, 1, 1);
	glm::vec3 rayDirection = glm::vec3(currentCamera->getInverseViewMatrix()
		* glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0)); // World space
	return Ray(viewPoint, rayDirection);

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
			//pixelColor.normalize();
			//std::cout << pixelColor.R <<" "<< pixelColor.G << " "<< pixelColor.B << std::endl;
			image->setPixelColor(i, j, pixelColor);
		}
	}
	QGraphicsItem* pixmapPtr;
	pixmapPtr = _scene->itemAt(0, 0, QTransform());
	if (!pixmapPtr)
	{
		QPixmap pixmap;
		pixmap.convertFromImage(*image->getImage());
		_scene->addPixmap(pixmap);
	}
	else
	{
		QGraphicsPixmapItem* canvasPixmap = qgraphicsitem_cast<QGraphicsPixmapItem*>(pixmapPtr);
		QPixmap pixmap;
		pixmap.convertFromImage(*image->getImage());
		canvasPixmap->setPixmap(pixmap);
		_scene->update();
	}
}

Renderer::Renderer(QGraphicsScene* scene)
{
	_scene = scene;
}
