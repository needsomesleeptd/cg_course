#ifndef LAB_03_CAMERA_H
#define LAB_03_CAMERA_H

#include "TransformVisitor.h"
#include "object.h"
#include "drawManager.h"
#include "transformManager.h"
#include "glmWrapper.h"
#include "cameraStructure.h"
#include "TransformVisitor.h"
#include "DrawVisitor.h"
#include "baseCamera.h"

class Camera : public BaseCamera
{
 public:

	friend void TransformVisitor::visit(const Camera& camera);
	friend DrawVisitor; // need this to enable getting Proection out of protected


	Camera() = default;
	explicit Camera(const Point& coordinates, const Vector3& direction);
	explicit Camera(std::shared_ptr<CameraStructureImp> cameraStructure);
	~Camera() override = default;


	void accept(std::shared_ptr<Visitor> visitor) override;

 private:
	std::shared_ptr<CameraStructureImp> _cameraStructure;

};

class CameraFactory : public BaseCameraFactory
{
 public:
	CameraFactory(const Point& position, const Vector3& direction);
	virtual ~CameraFactory() = default;

	virtual std::shared_ptr<Camera> create() override;
 private:
	Point _position{ 0, 0, 0 };
	Vector3 _direction{ 0, 0, 0 };
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
//	[[nodiscard]] Matrix4 getView();
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
//	Vector3 _front{0.0f, 0.0f, -1.0f};
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
