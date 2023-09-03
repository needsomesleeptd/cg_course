//
// Created by Андрей on 09.08.2023.
//

#include "Renderer.cuh"
#include "scene.h"

#include "color.h"
#include "../scene/scene.h"
#include "../object/invisibleObject/camera.h"
#include "../image_adapter/ImageAdapter.h"
#include "../object/object.h"

#include "LightSource.cuh"
#include "../utils/cudaUtils.h"
#include "CudaShape.cuh"
#include "../object/invisibleObject/lightSource/LightSource.cuh"
#include "../math_primitives/ray/ray.h"
#include "material.h"
#include "camera.h"

#include <cuda_profiler_api.h>

//const ColorRGB backGround = ColorRGB(0, 0, 0);

__device__  ColorRGB rayTrace(const Ray& tracedRay,
	ColorRGB& otherColor,
	int curDepth,
	CudaArray<CudaShape>* objects,
	LightSource* lightSource)
{
	CudaShape* closestShape;
	float t = maxRange;
	ColorRGB finalColor;

	for (int i = 0; i < objects->n; i++)
	{
		//printf("%d \n" ,i);
		float intersection_t = (objects->values[i]).intersection(tracedRay);
		//std::cout << intersection_t << std::endl;
		if (intersection_t > 0 || fabs(intersection_t) < EPS)
		{
			if (intersection_t < t)
			{
				t = intersection_t;
				closestShape = &objects->values[i];
			}
		}
	}
	//0x55b4c14f9288
	//printf("\nobjects = %p\n lightSource = %p\n\n",objects,lightSource);
	//printf("\nobject1 = %p\n object2 = %p\n\n",objects,lightSource);
	if (abs(t - maxRange) < EPS)
	{
		return  ColorRGB(0, 0, 0); //Returning background color
	}
	//printf("light source = %f",lightSource->getColor().R);
	//printf("function = %p\n",lightSource->getColor());
	//lightSource->getColor();
	LightSource* currentLightSource = lightSource;

	VecD3 intersectionPoint = tracedRay.getPoint(t);
	//printf("Position  == %f",currentLightSource->getPosition().x());
	VecD3 lightVector = glm::normalize(intersectionPoint - currentLightSource->getPosition());
	//lightSource->getColor();

	VecD3 shapeNormal = glm::normalize(closestShape->getNormal(intersectionPoint));

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

	return finalColor;
	/*if (shapeMaterial._k_s > 0.0f)
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
	float imageHeight = image->_height; //image->getHeight();
	float imageWidth = image->_width; //image->getWidth();
	VecD3 viewPoint = currentCamera->getViewPoint();
	VecD3 l = viewPoint - float(imageWidth / 2);
	VecD3 r = viewPoint + float(imageWidth / 2);

	VecD3 up = viewPoint - float(imageHeight / 2);
	VecD3 down = viewPoint + float(imageHeight / 2);

	//VecD3 u_deformation = float(x) * (r - l) / float(imageWidth);
	//VecD3 v_deformation = float(y) * (up - down) / float(imageHeight);//TODO::fix ray origin
	/*VecD3 ray_origin = viewPoint + u_deformation * VecD3(1, 0, 0) + v_deformation * VecD3(0, 1, 0);
	return Ray(ray_origin, -currentCamera->getViewDirection());*/
	glm::vec2 coord = { (float)x / (float)imageWidth, (float)y / (float)imageWidth };
	coord = coord * 2.0f - 1.0f; // -1 -> 1

	VecD4 target = currentCamera->getInverseProjectionMatrix() * VecD4(coord.x, coord.y, 1, 1);
	VecD3 rayDirection = VecD3(currentCamera->getInverseViewMatrix() * VecD4(glm::normalize(VecD3(target) / target.w), 0)); // World space
	return Ray(viewPoint, rayDirection);
	/*VecD3 dir =VecD3(4*x,3*y,0) - viewPoint;
	return Ray(viewPoint,dir);*/
}

__device__ ColorRGB renderPixel(int x,
	int y,
	Camera* camera,
	CudaArray<CudaShape>* objects,
	LightSource* lightSource,
	ImageAdapter* image)
{
	Ray tracedRay = createRay(x, y, camera, image);
	ColorRGB finalColor;
	finalColor = rayTrace(tracedRay, finalColor, 0, objects, lightSource);
	return finalColor;
}

__global__ void renderSceneCuda(Camera** camera,
	CudaArray<CudaShape>* objects,
	LightSource** lightSource,
	ImageAdapter* image)
{

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	//printf("height = %d width = %d\n",image->_height,image->_width);
	//printf("%d %d\n",i,j);
	if (i >= image->_width || j >= image->_height)
		return;
	ColorRGB pixelColor = renderPixel(i, j, *camera, objects, *lightSource, image);
	pixelColor.normalize();
	//std::cout << pixelColor.R <<" "<< pixelColor.G << " "<< pixelColor.B << std::endl;

	image->setPixelColor(i, j, pixelColor);
	//printf("%p",image->colorMatrix);
	//pixelColor.R = 1;
}


__global__ void createLightSource(
	LightSource** lightSource,VecD3 position,float intensivity)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		(*lightSource) = new LightSource(position,intensivity);
	}
}

__global__ void destroyLightSource(
	LightSource** lightSource)
{
	delete *lightSource;
}


__global__ void createCamera(
	Camera** camera,VecD3 coordinates,VecD3 direction)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		(*camera) = new Camera(coordinates,direction);
	}
}



__host__ ImageAdapter* Renderer::renderScene(std::shared_ptr<Scene> scene)
{
	int blockX = 5;
	int blockY = 5;
	int nx = 600;
	int ny = 600;
	ImageAdapter hostImage;
	hostImage._width = nx;
	hostImage._height = ny;
	ImageAdapter* deviceImage;

	cpuErrorCheck(cudaMalloc((void**)&(hostImage.colorMatrix), sizeof(ColorRGB) * nx * ny));
	cpuErrorCheck(cudaMalloc(&deviceImage, sizeof(ImageAdapter)));

	cpuErrorCheck(cudaMemcpy(deviceImage, &hostImage, sizeof(ImageAdapter), cudaMemcpyHostToDevice));


	std::shared_ptr<Camera> cameraHost = scene->getCamera();



	std::shared_ptr<LightSource> lightSourceHost = std::dynamic_pointer_cast<LightSource>(scene->getLightSource());


	//creating lightSource
	LightSource** lightSourceDevice;
	cpuErrorCheck(cudaMalloc((void **)&lightSourceDevice, sizeof(LightSource**)));
	createLightSource<<<1,1>>>(lightSourceDevice,VecD3(1,1,1),1.0f);
	cpuErrorCheck(cudaGetLastError());
	cpuErrorCheck(cudaDeviceSynchronize());

	//creating Camera
	Camera** cameraDevice;
	cpuErrorCheck(cudaMalloc((void **)&cameraDevice, sizeof(Camera**))); //TODO::make camera class transparent
	createCamera<<<1,1>>>(cameraDevice,cameraHost->_cameraStructure->getCoordinates(),cameraHost->_cameraStructure->getViewDirection());
	cpuErrorCheck(cudaGetLastError());
	cpuErrorCheck(cudaDeviceSynchronize());



	/*cpuErrorCheck(cudaMalloc((void**)&(lightSourceDevice), sizeof(LightSource)));
	cpuErrorCheck(cudaMemcpy(lightSourceDevice, lightSourceHost.get() ,sizeof(LightSource), cudaMemcpyHostToDevice));*/


	std::vector<std::shared_ptr<BaseObject>> hostObjects = scene->getModels();

	CudaArray<CudaShape> hostVector;
	hostVector.n = hostObjects.size();
	hostVector.values = (CudaShape*)malloc(sizeof(CudaShape) * hostVector.n);

	for (int i = 0; i < hostObjects.size(); i++)
	{
		std::shared_ptr<BaseShape> hostShape = std::dynamic_pointer_cast<BaseShape>(hostObjects[i]);

		/*void* deviceShape;
		switch (hostShape.getShapeType())
		{
			case CudaShapeType::sphere:
				cpuErrorCheck(cudaMalloc((void**)&(deviceShape), sizeof(Sphere)));
				cpuErrorCheck(cudaMemcpy(deviceShape, hostShape.get() sizeof(Sphere), cudaMemcpyHostToDevice));
			break;
		}
		CudaShape *cudaDeviceShape;
		cpuErrorCheck(cudaMalloc((void**)(CudaDevice), sizeof(CudaArray)));*/
		CudaShape hostCudaShape(CudaShapeType::sphere,hostShape.get());
		hostVector.values[i] = hostCudaShape;
	}
	CudaArray<CudaShape> transferArray;
	transferArray.n = hostObjects.size();
	cpuErrorCheck(cudaMalloc((void**)&(transferArray.values), sizeof(CudaShape) * transferArray.n));
	cpuErrorCheck(cudaMemcpy(transferArray.values, hostVector.values ,sizeof(CudaShape) * transferArray.n, cudaMemcpyHostToDevice));

	CudaArray<CudaShape>* deviceVector;
	cpuErrorCheck(cudaMalloc((void**)&(deviceVector), sizeof(CudaArray<CudaShape>)));
	cpuErrorCheck(cudaMemcpy(deviceVector, &transferArray ,sizeof(CudaArray<CudaShape>), cudaMemcpyHostToDevice));




	dim3 blocks(nx / blockX, ny / blockY);
	dim3 threads(blockX, blockY);
	renderSceneCuda<<<blocks, threads>>>(cameraDevice, deviceVector, lightSourceDevice, deviceImage);
	cpuErrorCheck(cudaGetLastError());
	cpuErrorCheck(cudaDeviceSynchronize());

	ImageAdapter* resultImage;
	resultImage = (ImageAdapter*)malloc(sizeof(ImageAdapter)); //Forced to allocate on heap because of destructor
	//TODO::Create normal destructor for image
	cudaMemcpy(resultImage, deviceImage, sizeof(ImageAdapter), cudaMemcpyDeviceToHost);
	void* deviceColorMap = resultImage->colorMatrix;
	resultImage->colorMatrix = (ColorRGB*)malloc(sizeof(ColorRGB) * nx * ny);
	cudaMemcpy(resultImage->colorMatrix, deviceColorMap, sizeof(ColorRGB) * nx * ny, cudaMemcpyDeviceToHost);



	cudaFree(hostImage.colorMatrix);
	cudaFree(deviceImage);
	cudaFree(transferArray.values);
	cudaFree(deviceVector);

	destroyLightSource<<<1, 1>>>(lightSourceDevice);
	cudaFree(lightSourceDevice);

	resultImage->_width = nx;
	resultImage->_height = ny;
	return resultImage;
}

/*Renderer::Renderer(QGraphicsScene* scene)
{
	_scene = scene;
}*/

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
};
__device__ void Renderer::rayTrace(const Ray& tracedRay, ColorRGB& finalColor, Scene* scene, int curDepth)
{
	//TODO:: make this
};
