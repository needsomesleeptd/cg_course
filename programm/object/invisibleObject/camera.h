#ifndef LAB_03_CAMERA_H
#define LAB_03_CAMERA_H

#include <QKeyEvent>
#include "cameraStructure.h"
#include "baseCamera.h"

class TransformVisitor;
class RayCastCanvas;
class Camera : public BaseCamera
{
 public:


	friend TransformVisitor;
	friend RayCastCanvas;



	Camera() = default;
	explicit Camera(const  VecD3& coordinates, const  VecD3& direction);
	explicit Camera(std::shared_ptr<CameraStructureImp> cameraStructure);
	~Camera() override = default;

	VecD3 getViewPoint() override;
	VecD3 getViewDirection() override;
	VecD3 getUpVector() override;
	MatD4  getInverseProjectionMatrix() override;
	MatD4  getInverseViewMatrix() override;


	void setImageParams(int height, int width);


	void accept(std::shared_ptr<Visitor> visitor) override;

 private:
	std::shared_ptr<CameraStructureImp> _cameraStructure;

};

class CameraFactory : public BaseCameraFactory
{
 public:
	CameraFactory(const  VecD3& position, const  VecD3& direction);
	virtual ~CameraFactory() = default;

	virtual std::shared_ptr<Camera> create() override;
 private:
	 VecD3 _position{ 0, 0, 0 };
	 VecD3 _direction{ 0, 0, 0 };
};

//};

#endif //LAB_03_CAMERA_H
