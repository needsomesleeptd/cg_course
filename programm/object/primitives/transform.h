#ifndef LAB_03_TRANSFORM_H
#define LAB_03_TRANSFORM_H


#include "vector.h"

class TransformParams
{
 public:
	TransformParams() :_moveParams({0,0,0}),_scaleParams({1,1,1}),_rotateParams({0,0,0})
	{

	};
	TransformParams(const  VecD3& moveParams, const  VecD3& scaleParams, const  VecD3& rotateParams)
		: _moveParams(moveParams), _scaleParams(scaleParams), _rotateParams(rotateParams)
	{
	};
	TransformParams(const TransformParams& transform) = default;
	~TransformParams() = default;

	[[nodiscard]]  VecD3 getMoveParams() const noexcept
	{
		return  VecD3(_moveParams);
	};
	[[nodiscard]]  VecD3 getScaleParams() const noexcept
	{
		return  VecD3(_scaleParams);
	};
	[[nodiscard]]  VecD3 getRotateParams() const noexcept
	{
		return  VecD3(_rotateParams);
	};

	void setMoveParams( VecD3& moveParams)
	{
		_moveParams = moveParams;
	};
	void setScaleParams( VecD3& scaleParams)
	{
		_scaleParams = scaleParams;
	};
	void setRotateParams( VecD3& rotateParams)
	{
		_rotateParams = rotateParams;
	};

 private:
	 VecD3 _moveParams;
	 VecD3 _scaleParams;
	 VecD3 _rotateParams;
};

#endif //LAB_03_TRANSFORM_H
