#ifndef DZ2_CG_COURSE_PROGRAMM_MATH_PRIMITIVES_VECTOR_VECTOR_H_
#define DZ2_CG_COURSE_PROGRAMM_MATH_PRIMITIVES_VECTOR_VECTOR_H_

#include <initializer_list>
#include <cuda_runtime.h>
#include <iostream>
#include "base_math_primitive.h"
#include "glmWrapper.h"


using MatD4 = Matrix4;
using VecD3 = Vector3;
using VecD4 = Vector4;
/*
class VecD4
{

 public:
	__host__ __device__ VecD4()
	{
	}
	__host__ __device__ VecD4(float e0, float e1, float e2, float e3)
	{
		e[0] = e0;
		e[1] = e1;
		e[2] = e2;
		e[3] = e3;
	}
	__host__ __device__ inline float x() const
	{
		return e[0];
	}
	__host__ __device__ inline float y() const
	{
		return e[1];
	}
	__host__ __device__ inline float z() const
	{
		return e[2];
	}

	__host__ __device__ inline float w() const
	{
		return e[3];
	}
	__host__ __device__ inline float r() const
	{
		return e[0];
	}
	__host__ __device__ inline float g() const
	{
		return e[1];
	}
	__host__ __device__ inline float b() const
	{
		return e[2];
	}

	__host__ __device__ inline float h() const
	{
		return e[3];
	}

	__host__ __device__ inline const VecD4 operator+() const
	{
		return *this;
	}
	__host__ __device__ inline  VecD4 operator-() const
	{
		return VecD4(-e[0], -e[1], -e[2], -e[3]);
	}
	__host__ __device__ inline float operator[](int i) const
	{
		return e[i];
	}
	__host__ __device__ inline float& operator[](int i)
	{
		return e[i];
	};

	__host__ __device__ inline VecD4& operator+=(const VecD4& v2);
	__host__ __device__ inline VecD4& operator-=(const VecD4& v2);
	__host__ __device__ inline VecD4& operator*=(const VecD4& v2);
	__host__ __device__ inline VecD4& operator/=(const VecD4& v2);
	__host__ __device__ inline VecD4& operator*=(const float t);
	__host__ __device__ inline VecD4& operator/=(const float t);

	__host__ __device__ inline float length() const
	{
		return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2] + e[3] * e[3]);
	}
	__host__ __device__ inline float squared_length() const
	{
		return e[0] * e[0] + e[1] * e[1] + e[2] * e[2] + e[3] * e[3];
	}
	__host__ __device__ inline void make_unit_vector();

	float e[4];
};

class VecD3
{

 public:
	__host__ __device__ VecD3()
	{
	}
	__host__ __device__ VecD3(float e0, float e1, float e2)
	{
		e[0] = e0;
		e[1] = e1;
		e[2] = e2;
	}

	__host__ __device__ VecD3(const VecD4& ref)
	{
		e[0] = ref.x();
		e[1] = ref.y();
		e[2] = ref.z();
	}

	__host__ __device__ inline float x() const
	{
		return e[0];
	}
	__host__ __device__ inline float y() const
	{
		return e[1];
	}
	__host__ __device__ inline float z() const
	{
		return e[2];
	}
	__host__ __device__ inline float r() const
	{
		return e[0];
	}
	__host__ __device__ inline float g() const
	{
		return e[1];
	}
	__host__ __device__ inline float b() const
	{
		return e[2];
	}

	__host__ __device__ inline const VecD3& operator+() const
	{
		return *this;
	}
	__host__ __device__ inline VecD3 operator-() const
	{
		return VecD3(-e[0], -e[1], -e[2]);
	}

	__host__ __device__ inline VecD3 operator-(float t)
	{
		return VecD3(e[0] - t, e[1] - t, e[2] - t);
	}

	__host__ __device__ inline VecD3 operator+(float t)
	{
		return VecD3(e[0] + t, e[1] + t, e[2] + t);
	}

	__host__ __device__ inline float operator[](int i) const
	{
		return e[i];
	}
	__host__ __device__ inline float& operator[](int i)
	{
		return e[i];
	};



	__host__ __device__ inline VecD3& operator+=(const VecD3& v2);
	__host__ __device__ inline VecD3& operator-=(const VecD3& v2);
	__host__ __device__ inline VecD3& operator*=(const VecD3& v2);
	__host__ __device__ inline VecD3& operator/=(const VecD3& v2);
	__host__ __device__ inline VecD3& operator*=(const float t);
	__host__ __device__ inline VecD3& operator/=(const float t);

	__host__ __device__ inline float length() const
	{
		return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
	}
	__host__ __device__ inline float squared_length() const
	{
		return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
	}
	__host__ __device__ inline void make_unit_vector();

	float e[3];
};

inline std::istream& operator>>(std::istream& is, VecD3& t)
{
	is >> t.e[0] >> t.e[1] >> t.e[2];
	return is;
}

inline std::ostream& operator<<(std::ostream& os, const VecD3& t)
{
	os << t.e[0] << " " << t.e[1] << " " << t.e[2];
	return os;
}

__host__ __device__ inline void VecD3::make_unit_vector()
{
	float k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
	e[0] *= k;
	e[1] *= k;
	e[2] *= k;
}

__host__ __device__ inline VecD3 operator+(const VecD3& v1, const VecD3& v2)
{
	return VecD3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline VecD3 operator-(const VecD3& v1, const VecD3& v2)
{
	return VecD3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline VecD3 operator*(const VecD3& v1, const VecD3& v2)
{
	return VecD3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline VecD3 operator/(const VecD3& v1, const VecD3& v2)
{
	return VecD3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline VecD3 operator*(float t, const VecD3& v)
{
	return VecD3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline VecD3 operator/(VecD3 v, float t)
{
	return VecD3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__host__ __device__ inline VecD3 operator*(const VecD3& v, float t)
{
	return VecD3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline float dot(const VecD3& v1, const VecD3& v2)
{
	return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ inline VecD3 cross(const VecD3& v1, const VecD3& v2)
{
	return VecD3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
		(-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
		(v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}

__host__ __device__ inline VecD3& VecD3::operator+=(const VecD3& v)
{
	e[0] += v.e[0];
	e[1] += v.e[1];
	e[2] += v.e[2];
	return *this;
}




__host__ __device__ inline VecD3& VecD3::operator*=(const VecD3& v)
{
	e[0] *= v.e[0];
	e[1] *= v.e[1];
	e[2] *= v.e[2];
	return *this;
}

__host__ __device__ inline VecD3& VecD3::operator/=(const VecD3& v)
{
	e[0] /= v.e[0];
	e[1] /= v.e[1];
	e[2] /= v.e[2];
	return *this;
}

__host__ __device__ inline VecD3& VecD3::operator-=(const VecD3& v)
{
	e[0] -= v.e[0];
	e[1] -= v.e[1];
	e[2] -= v.e[2];
	return *this;
}

__host__ __device__ inline VecD3& VecD3::operator*=(const float t)
{
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;
	return *this;
}

__host__ __device__ inline VecD3& VecD3::operator/=(const float t)
{
	float k = 1.0 / t;

	e[0] *= k;
	e[1] *= k;
	e[2] *= k;
	return *this;
}


__host__ __device__ inline VecD3 normalise(VecD3 v)
{
	return v / v.length();
}



inline std::istream& operator>>(std::istream& is, VecD4& t)
{
	is >> t.e[0] >> t.e[1] >> t.e[2] >> t.e[3];
	return is;
}

inline std::ostream& operator<<(std::ostream& os, const VecD4& t)
{
	os << t.e[0] << " " << t.e[1] << " " << t.e[2] << " " << t.e[3];
	return os;
}

__host__ __device__ inline void VecD4::make_unit_vector()
{
	float k = 1.0f / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2] + e[3] * e[3]);
	e[0] *= k;
	e[1] *= k;
	e[2] *= k;
	e[3] *= k;
}

__host__ __device__ inline VecD4 operator+(const VecD4& v1, const VecD4& v2)
{
	return VecD4(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2], v1.e[3] + v1.e[3]);
}

__host__ __device__ inline VecD4 operator-(const VecD4& v1, const VecD4& v2)
{
	return VecD4(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2],v1.e[3] - v1.e[3]);
}

__host__ __device__ inline VecD4 operator*(const VecD4& v1, const VecD4& v2)
{
	return VecD4(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2],v1.e[3] * v1.e[3]);
}

__host__ __device__ inline VecD4 operator/(const VecD4& v1, const VecD4& v2)
{
	return VecD4(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2],v1.e[3] / v1.e[3]);
}

__host__ __device__ inline VecD4 operator*(float t, const VecD4& v)
{
	return VecD4(t * v.e[0], t * v.e[1], t * v.e[2],t * v.e[3]);
}

__host__ __device__ inline VecD4 operator/(VecD4 v, float t)
{
	return VecD4(v.e[0] / t, v.e[1] / t, v.e[2] / t, v.e[3] / t);
}

__host__ __device__ inline VecD4 operator*(const VecD4& v, float t)
{
	return VecD4(t * v.e[0], t * v.e[1], t * v.e[2], t * v.e[3]);
}

__host__ __device__ inline float dot(const VecD4& v1, const VecD4& v2)
{
	return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2] + v1.e[3] * v2.e[3];
}


__host__ __device__ inline VecD4& VecD4::operator+=(const VecD4& v)
{
	e[0] += v.e[0];
	e[1] += v.e[1];
	e[2] += v.e[2];
	e[3] += v.e[3];
	return *this;
}

__host__ __device__ inline VecD4& VecD4::operator*=(const VecD4& v)
{
	e[0] *= v.e[0];
	e[1] *= v.e[1];
	e[2] *= v.e[2];
	e[3] *= v.e[3];
	return *this;
}

__host__ __device__ inline VecD4& VecD4::operator/=(const VecD4& v)
{
	e[0] /= v.e[0];
	e[1] /= v.e[1];
	e[2] /= v.e[2];
	e[3] /= v.e[3];
	return *this;
}

__host__ __device__ inline VecD4& VecD4::operator-=(const VecD4& v)
{
	e[0] -= v.e[0];
	e[1] -= v.e[1];
	e[2] -= v.e[2];
	return *this;
}

__host__ __device__ inline VecD4& VecD4::operator*=(const float t)
{
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;
	e[3] *= t;
	return *this;
}

__host__ __device__ inline VecD4& VecD4::operator/=(const float t)
{
	float k = 1.0 / t;

	e[0] *= k;
	e[1] *= k;
	e[2] *= k;
	e[3] *= k;
	return *this;
}

__host__ __device__ inline VecD4 normalise(VecD4 v)
{
	return v / v.length();
}

*/


#endif //DZ2_CG_COURSE_PROGRAMM_MATH_PRIMITIVES_VECTOR_VECTOR_H_