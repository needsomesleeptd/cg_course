#ifndef LAB_03_TRANSFORM_H
#define LAB_03_TRANSFORM_H

#include "glmWrapper.h"

class TransformParams
{
 public:
	TransformParams() : _moveParams({ 0, 0, 0 }), _scaleParams({ 1, 1, 1 }), _rotateParams({ 0, 0, 0 })
	{

	};
	TransformParams(const VecD3& moveParams, const VecD3& scaleParams, const VecD3& rotateParams)
		: _moveParams(moveParams), _scaleParams(scaleParams), _rotateParams(rotateParams)
	{
	};
	TransformParams(const TransformParams& transform) = default;
	static VecD3 toRadians(const VecD3& rotation)
	{
		double PI = std::acos(-1);
		float rotation_x = rotation.x * PI / 180.0;
		float rotation_y = rotation.y * PI / 180.0;
		float rotation_z = rotation.z * PI / 180.0;
		return VecD3(rotation_x, rotation_y, rotation_z);
	}

	[[nodiscard]] static VecD3 rotatePoint(const VecD3& point, const VecD3& rotation) noexcept //rotation is in radians and centered
	{
		VecD3  resPoint = point;

		//x rotation
		float save_y = point.y;
		float cos_val = cos(rotation.x);
		float sin_val = sin(rotation.x);
		
		resPoint.y = (resPoint.y)  * cos_val + (resPoint.z) * sin_val;
		resPoint.z = (save_y) * -sin_val + (resPoint.z) * cos_val;

		//y rotation
		float save_x = resPoint.x;

		cos_val = cos(rotation.y);
		sin_val = sin(rotation.y);

		resPoint.x = (resPoint.x) * cos_val - (resPoint.z) * sin_val;
		resPoint.z = (save_x) * sin_val + (resPoint.z) * cos_val;

		//z rotation
		save_x = resPoint.x;

		cos_val = cos(rotation.z);
		sin_val = sin(rotation.z);

		resPoint.x = (resPoint.x) * cos_val + (resPoint.y) * sin_val;
		resPoint.y = (save_x) * -sin_val + (resPoint.y) * cos_val;

		return  resPoint;


	}

	~TransformParams() = default;

	[[nodiscard]]  VecD3 getMoveParams() const noexcept
	{
		return VecD3(_moveParams);
	};
	[[nodiscard]]  VecD3 getScaleParams() const noexcept
	{
		return VecD3(_scaleParams);
	};
	[[nodiscard]]  VecD3 getRotateParams() const noexcept
	{
		return VecD3(_rotateParams);
	};

	void setMoveParams(VecD3& moveParams)
	{
		_moveParams = moveParams;
	};
	void setScaleParams(VecD3& scaleParams)
	{
		_scaleParams = scaleParams;
	};
	void setRotateParams(VecD3& rotateParams)
	{
		_rotateParams = rotateParams;
	};

 private:
	VecD3 _moveParams;
	VecD3 _scaleParams;
	VecD3 _rotateParams;
};

#endif //LAB_03_TRANSFORM_H
