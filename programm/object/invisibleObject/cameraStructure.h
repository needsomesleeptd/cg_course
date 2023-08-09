#ifndef LAB_03_CAMERASTRUCTURE_H
#define LAB_03_CAMERASTRUCTURE_H

#include "glmWrapper.h"
#include "transform.h"

class CameraStructureImp {
public:
	CameraStructureImp() = default;
	explicit CameraStructureImp(const Point &coordinates,const Vector3 &direction);
	~CameraStructureImp() = default;

	[[nodiscard]] const Point getCoordinates() const;
	void setCoordinates(Point &coordinates);
	void transform(const TransformParams &transformParams);

	[[nodiscard]] Matrix4 getView();
	[[nodiscard]] Matrix4 getProjection() const;
	void move(const Point& moveParams);
	Vector3 setDirection(const Vector3 &direction);

protected:
	void updateVectors();
	void rotate(float xOffset, float yOffset);

private:
	Point _coordinates{0.0, 0.0, 0.0};

	Vector3 _front{0.0f, 0.0f, -1.0f};

	Vector3 _up{0.0, 1.0, 0.0};

	Vector3 _right{0.0, 0.0, 1.0};

	Vector3 _worldUp{0.0, 1.0, 0.0};

	float _yaw = -90;

	float _pitch = 0;

	float _aspect = 1.0f;
//	float _aspect;
	float _zNear = 0.1f;

	float _zFar = 100.0f;



};


#endif //LAB_03_CAMERASTRUCTURE_H
