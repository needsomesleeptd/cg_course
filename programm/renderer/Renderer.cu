//
// Created by Андрей on 09.08.2023.
//

#include "Renderer.h"
#include "scene.h"

#include "color.h"
#include "../scene/scene.h"
#include "../object/invisibleObject/camera.h"
#include "../image_adapter/ImageAdapter.h"
#include "../object/object.h"

#include "../object/invisibleObject/lightSource/baseLightSource.h"

__device__  void rayTrace(const Ray& tracedRay,
	ColorRGB& finalColor,
	Scene* scene,
	int curDepth, thrust::device_vector<BaseShape*> objects,
	BaseLightSource* lightSource)
{
	BaseShape* closestShape;
	float t = maxRange;
	/*for (thrust::device_vector<BaseShape*>::iterator shape = objects.begin(); shape != objects.end(); shape++)
	{

		float intersection_t = (**shape).intersection(tracedRay);
		//std::cout << intersection_t << std::endl;
		if (intersection_t > 0 || fabs(intersection_t) < EPS)
		{
			t = thrust::min(t, intersection_t);
			closestShape = *shape;
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
		float specularDot = dot(reflected.D, tracedRay.D);
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
			rayTrace(reflected, rcol, scene, curDepth + 1, objects, lightSource);
			finalColor = rcol * shapeMaterial._k_s * closestShape->getMaterial()._color + finalColor;
		}
	}*/
}

__device__  Ray createRay(int x, int y, Camera* currentCamera, ImageAdapter* image)
{
	float imageHeight = image->getHeight();
	float imageWidth = image->getWidth();
	VecD3 viewPoint = { 0, 0, -3 };
	/*VecD3 l = viewPoint - float(imageWidth / 2);
	VecD3 r = viewPoint + float(imageWidth / 2);

	VecD3 up = viewPoint - float(imageHeight / 2);
	VecD3 down = viewPoint + float(imageHeight / 2);*/

	//VecD3 u_deformation = float(x) * (r - l) / float(imageWidth);
	//VecD3 v_deformation = float(y) * (up - down) / float(imageHeight);//TODO::fix ray origin
	/*VecD3 ray_origin = viewPoint + u_deformation * VecD3(1, 0, 0) + v_deformation * VecD3(0, 1, 0);
	return Ray(ray_origin, -currentCamera->getViewDirection());*/
	glm::vec2 coord = { (float)x / (float)imageWidth, (float)y / (float)imageWidth };
	coord = coord * 2.0f - 1.0f; // -1 -> 1

	VecD4 target = VecD4(coord.x, coord.y, 1, 1);
	VecD3 rayDirection = normalise(VecD3(target) / target.w()); // World space //TODO::Check this
	return Ray(viewPoint, rayDirection);
	/*VecD3 dir =VecD3(4*x,3*y,0) - viewPoint;
	return Ray(viewPoint,dir);*/
}

__device__ ColorRGB renderPixel(int x,
	int y,
	Scene* scene,
	Camera* camera,
	thrust::device_vector<BaseShape*> objects,
	BaseLightSource* lightSource,
	ImageAdapter* image)
{
	Ray tracedRay = createRay(x, y, camera, image);
	ColorRGB finalColor;
//	rayTrace(tracedRay, finalColor, scene, 0, objects, lightSource);
	return finalColor;
}

__global__ void renderSceneCuda(Scene* scene,
	Camera* camera,
	BaseRenderer* renderer,
	ImageAdapter* image,
	thrust::device_vector<BaseObject*> objects, BaseLightSource* lightSource)
{

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	/*ColorRGB pixelColor = renderPixel(i, j, scene, camera, objects, lightSource);
	pixelColor.normalize();
	//std::cout << pixelColor.R <<" "<< pixelColor.G << " "<< pixelColor.B << std::endl;
	image->setPixelColor(i, j, pixelColor);*/
}

__host__ void Renderer::renderScene(std::shared_ptr<Scene> scene)
{
	int blockX = 10;
	int blockY = 10;
	int nx = 500;
	int ny = 500;
	std::shared_ptr<ImageAdapter> image = std::make_shared<ImageAdapter>(nx, ny);
	std::shared_ptr<Camera> camera = scene->getCamera();
	std::shared_ptr<BaseLightSource> lightSource = scene->getLightSource();
	thrust::device_vector < BaseObject * > deviceObjects;

	auto hostObjects = scene->getModels();
	/*for (int i = 0; i < hostObjects.size(); i++)
	{
		deviceObjects.push_back(hostObjects[i].get());
	}*/

	dim3 blocks(nx / blockX + 1, ny / blockY + 1);
	dim3 threads(blockX, blockY);
	/*
	Scene* scene,
	Camera* camera,
	BaseRenderer* renderer,
	ImageAdapter* image,
	thrust::device_vector<BaseObject*> objects, BaseLightSource* lightSource
	 */
	renderSceneCuda<<<blocks, threads>>>(scene.get(), camera.get(), this, image.get(), deviceObjects, lightSource.get());

	cpuErrorCheck(cudaGetLastError());
	cpuErrorCheck(cudaDeviceSynchronize());

}

Renderer::Renderer() = default;

__host__   void Renderer::getImage(ImageAdapter* image)
{
	;
}
__device__ Ray Renderer::createRay(int x, int y, Camera* currentCamera)
{
	//TODO:: make this
}
__device__ ColorRGB Renderer::renderPixel(int x, int y, Scene* scene, Camera* camera)
{
	//TODO:: make this
}
__device__ void Renderer::rayTrace(const Ray& tracedRay, ColorRGB& finalColor, Scene* scene, int curDepth)
{
	//TODO:: make this
}
