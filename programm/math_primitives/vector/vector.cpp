
#include "vector.h"
#include "vector_exceptions.h"
#include <time.h>
#include <cstring>
#include "cassert"

template<typename T, int n>
Vector<T, n>::Vector(int size) : len(size)
{
	if (n == 0)
	{
		time_t timer = time(nullptr);
		throw InvalidMallocException(__FILE__, __LINE__, "Vector<T>", ctime(&timer));
	}
	new T[size];
}

template<typename T, int n>
Vector<T, n>::Vector(std::initializer_list<T> l)
{
	len = l.size();
	if (n == 0)
	{
		time_t timer = time(nullptr);
		throw InvalidMallocException(__FILE__, __LINE__, "Vector<T>", ctime(&timer));
	}

	for (int i = 0; i < n; i++)
		values[i] = l[i];
}
template<typename T, int n>
T& Vector<T, n>::operator[](const int i)
{
	assert(i >= 0 && i < len); //TODO::Might need to insert exception instead of assert here;
	return values[i];
}
template<typename T, int n>
T Vector<T, n>::operator[](const int i) const
{
	assert(i >= 0 && i < len); //TODO::Might need to insert exception instead of assert here;
	return values[i];
}
template<typename T, int n>
T Vector<T, n>::operator^(Vector<T, n>& other) const
{
	if (len != other.len)
	{
		time_t timer = time(nullptr);
		throw InvalidMallocException(__FILE__, __LINE__, "Vector<T>", ctime(&timer));
	}
	T res = 0;
	for (int i = 0; i < other.len; i++)
		res += other[i] * this[i];
	return res;
}
template<typename T, int n>
Vector<T, n>::~Vector()
{
	delete values;
}
template<typename T, int n>
Vector<T, n>::Vector(const Vector& other)
{
	len = other.len;
	values = std::memmove(values,other.values,sizeof(T) * other.values); //TODO::Might need to choose memcpy
}
template<typename T, int n>
int Vector<T, n>::norm() const
{
	return *this ^ *this; //TODO::might overflow
}







