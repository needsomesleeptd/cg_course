#ifndef LAB_03_CAMERASTRUCTURE_H
#define LAB_03_CAMERASTRUCTURE_H

#include "transform.cuh"

const int defaultViewPortHeight = 500;
const int defaultViewPortWidth = 500;
const float nearClip = 0.1f;
const float farClip = 100.0f;

class CameraStructureImp
{
 public:
	CameraStructureImp() = default;
	__host__ __device__ explicit CameraStructureImp(const VecD3& coordinates, const VecD3& direction);
	~CameraStructureImp() = default;

	__host__ __device__  [[nodiscard]]  VecD3 getCoordinates() const;
	__host__ __device__  [[nodiscard]]  VecD3 getViewDirection() const;
	__host__ __device__  [[nodiscard]]  VecD3 getUp() const;
	__host__ __device__  [[nodiscard]]  VecD3 getRight() const;
	__host__ __device__  [[nodiscard]]  MatD4 getInverseProjectionMatrix();
	__host__ __device__  [[nodiscard]]  MatD4 getInverseViewMatrix();



	__host__ __device__ void setCoordinates(const VecD3& coordinates);
	__host__ __device__ void transform(const TransformParams& transformParams);

	//[[nodiscard]] Matrix4 getViewDirection();
	//[[nodiscard]] Matrix4 getProjection() const;
	__host__ __device__ void move(const VecD3& moveParams);
	__host__ __device__ void setDirection(const VecD3& direction);
	__host__ __device__ void updateView();
	__host__ __device__ void updateProjection();
	__host__ __device__ void setViewPortParams(int height, int width);

 private:
	VecD3 _coordinates{ 0.0, 0.0, 0.0 };

	VecD3 _forward{ 0.0f, 0.0f, -1.0f };

	VecD3 _up{ 0.0, 1.0, 0.0 };

	VecD3 _right{ 1.0, 0.0, 0.0 };

	MatD4 _mView{ 1.0f };
	MatD4 _mInverseView {1.0f};

	MatD4 _mProjection{ 1.0f };
	MatD4 _mInverseProjection{1.0f};

	int _viewPortWidth = defaultViewPortHeight;
	int _viewPortHeight = defaultViewPortWidth;


	float _verticalFOV = 45.0f;

	float _nearCLip = 0.1f;

	float _farClip = 100.0f;



};

#endif //LAB_03_CAMERASTRUCTURE_H
