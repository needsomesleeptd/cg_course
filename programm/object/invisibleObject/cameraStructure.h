#ifndef LAB_03_CAMERASTRUCTURE_H
#define LAB_03_CAMERASTRUCTURE_H

#include "transform.h"

class CameraStructureImp {
public:
	CameraStructureImp() = default;
	explicit CameraStructureImp(const  VecD3& coordinates,const  VecD3 &direction);
	~CameraStructureImp() = default;

	[[nodiscard]]   VecD3 getCoordinates() const;
	[[nodiscard]]  VecD3 getView() const;
	[[nodiscard]]  VecD3 getUp() const;


	void setCoordinates(const  VecD3& coordinates);
	void transform(const TransformParams &transformParams);

	//[[nodiscard]] Matrix4 getView();
	//[[nodiscard]] Matrix4 getProjection() const;
	void move(const  VecD3& moveParams);
	VecD3 setDirection(const  VecD3 &direction);

protected:
	//void updateVectors();
	void rotate(float xOffset, float yOffset);

private:
	 VecD3 _coordinates{0.0, 0.0, 0.0};

	 VecD3 _front{0.0f, 0.0f, -1.0f};

	 VecD3 _up{0.0, 1.0, 0.0};

	 VecD3 _right{1.0, 0.0, 0.0};

	 VecD3 _worldUp{0.0, 1.0, 0.0};

	float _yaw = -90;

	float _pitch = 0;

	float _aspect = 1.0f;
//	float _aspect;
	float _zNear = 0.1f;

	float _zFar = 100.0f;



};


#endif //LAB_03_CAMERASTRUCTURE_H
