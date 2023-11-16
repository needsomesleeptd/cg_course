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

	//friend void TransformVisitor::visit(const Camera& camera);
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
	void update(float time);

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


//class Camera: public InvisibleObject {
//public:
//	Camera() = default;
//	explicit Camera(const Point &coordinates): _coordinates(coordinates) {};
//	~Camera() override = default;
//
//	void transform(const TransformParams &transformParams) override;
//	void accept(std::shared_ptr<Visitor> visitor) override;
//
//	[[nodiscard]] Point getCoordinates() const noexcept;
//	void setCoordinates(Point &coordinates);
//
//	[[nodiscard]] Matrix4 getViewDirection();
//	[[nodiscard]] Matrix4 getProjection() const;
//
//	friend Point DrawManager::getProection(const Point &point);
//	friend void TransformManager::visit(const Camera &model);
//
//
//
//protected:
//	void updateVectors();
//
//private:
//	Point _coordinates;
//
//	Vector3 _forward{0.0f, 0.0f, -1.0f};
//	Vector3 _up{0.0, 1.0, 0.0};
//	Vector3 _right{0.0, 0.0, 1.0};
//	Vector3 _worldUp{0.0, 1.0, 0.0};
//	float _yaw = -90;
//	float _pitch = 0;
//	float _aspect = 1.0f;
//	float _zNear = 0.1f;
//	float _zFar = 100.0f;
//
//	void rotate(float xOffset, float yOffset);
//	void move(const Point& moveParams);
////	CameraStructureImp _cameraStructure;
//};

#endif //LAB_03_CAMERA_H
