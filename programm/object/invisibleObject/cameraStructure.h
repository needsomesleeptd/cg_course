#ifndef LAB_03_CAMERASTRUCTURE_H
#define LAB_03_CAMERASTRUCTURE_H

#include "transform.h"

const int defaultViewPortHeight = 1151;
const int defaultViewPortWidth = 901;
const float nearClip = 0.1f;
const float farClip = 100.0f;

struct CameraStructureImp
{
 public:
	CameraStructureImp() = default;
	explicit CameraStructureImp(const VecD3& coordinates, const VecD3& direction);
	~CameraStructureImp() = default;

	[[nodiscard]]  VecD3 getCoordinates() const;
	[[nodiscard]]  VecD3 getViewDirection() const;
	[[nodiscard]]  VecD3 getUp() const;
	[[nodiscard]]  VecD3 getRight() const;
	[[nodiscard]]  MatD4 getInverseProjectionMatrix();
	[[nodiscard]]  MatD4 getInverseViewMatrix();



	void setCoordinates(const VecD3& coordinates);
	void transform(const TransformParams& transformParams);

	//[[nodiscard]] Matrix4 getViewDirection();
	//[[nodiscard]] Matrix4 getProjection() const;
	void move(const VecD3& moveParams);
	void setDirection(const VecD3& direction);
	void updateView();
	void updateProjection();
	void setViewPortParams(int height, int width);

	VecD3 _coordinates{ 0.0, 0.0, 0.0 };

	VecD3 _forward{ 0.0f, 0.0f, -1.0f };

	VecD3 _up{ 0.0f, 1.0f, 0.0f};

	VecD3 _right{ 1.0f, 0.0f, 0.0f };

	MatD4 _mView{ 1.0f };
	MatD4 _mInverseView {1.0f};

	MatD4 _mProjection{ 1.0f };
	MatD4 _mInverseProjection{1.0f};

	int _viewPortWidth = defaultViewPortHeight;
	int _viewPortHeight = defaultViewPortWidth;


	float _verticalFOV = 60.0f;

	float _nearCLip = 1.0f;

	float _farClip = 100.0f;



};

#endif //LAB_03_CAMERASTRUCTURE_H
