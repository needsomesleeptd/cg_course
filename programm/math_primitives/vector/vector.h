#ifndef DZ2_CG_COURSE_PROGRAMM_MATH_PRIMITIVES_VECTOR_VECTOR_H_
#define DZ2_CG_COURSE_PROGRAMM_MATH_PRIMITIVES_VECTOR_VECTOR_H_

#include <initializer_list>
#include "base_math_primitive.h"
#include "glmWrapper.h"
template<typename T, int n>
class Vector
{
 public:
	explicit Vector();
	explicit Vector(int size);
	explicit Vector(std::initializer_list<T> l);
	explicit Vector(const Vector& other);
	explicit Vector(Vector&& other);
	~Vector();

	T& operator[](const int i);
	T operator[](const int i) const;
	T operator^(Vector<T, n>& other) const;
	Vector<T,n>& operator=(const Vector &other);
	void operator+=(const Vector &other);
	void operator/=(const Vector &other);
	void operator-=(const Vector &other);

	int norm() const;

 private:
	int len;
	T& values;

};
template<typename T, int n>
Vector<T, n> operator*(const Vector<T, n>& lhs, const T& rhs)
{
	Vector<T, n> ret = lhs;
	for (int i = n; i--; ret[i] *= rhs);
	return ret;
}

template<typename T, int n>
Vector<T, n> operator/(const Vector<T, n>& lhs, const double& rhs)
{
	Vector<T, n> ret = lhs;
	for (int i = n; i--; ret[i] /= rhs);
	return ret;
}

//starting shortcuts
using VecI3 = Vector<int,3>;
using VecD4 = glm::vec4;

using MatD4 = glm::mat4;

using VecI4 = Vector<int,4>;
//using VecD4 = Vector<double,4>;

template<typename T, int n>
Vector<T,3> operator*(const Vector<T,3>& lhs, const Vector<T,3>& rhs)
{
	T x = lhs[1] * rhs[2] - lhs[2] * rhs[1];
	T y = -1 * (lhs[0] * rhs[2] - lhs[2] * rhs[0]);
	T z = lhs[0] * rhs[1] - lhs[1] * rhs[0];
	return Vector<T,3>({x,y,z});
}
#endif //DZ2_CG_COURSE_PROGRAMM_MATH_PRIMITIVES_VECTOR_VECTOR_H_