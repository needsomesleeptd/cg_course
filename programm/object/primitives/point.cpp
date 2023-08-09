#include "point.h"
#include <cmath>

double to_radians(const double &angle) {
	return angle * (M_PI / 180);
}


Point::Point(const double x, const double y, const double z) {
	_x = x;
	_y = y;
	_z = z;
}

Point::Point(const Point &&point) noexcept {
	_x = point.getX();
	_y = point.getY();
	_z = point.getZ();
	point.~Point();
}


Point &Point::operator=(Point &&point) noexcept {
	_x = point.getX();
	_y = point.getY();
	_z = point.getZ();
	point.~Point();
	return *this;
}


void Point::move(const double &dx, const double &dy, const double &dz) {
	_x += dx;
	_y += dy;
	_z += dz;
}

void Point::scale(const double &kx, const double &ky, const double &kz) {
	_x *= kx;
	_y *= ky;
	_z *= kz;
}

void Point::rotate(const double &ox, const double &oy, const double &oz) {
	rotateX(ox);
	rotateY(oy);
	rotateZ(oz);
}

void Point::rotateX(const double &ox) {
	const double cos_rotate = cos(to_radians(ox));
	const double sin_rotate = sin(to_radians(ox));

	const double init_y = _y;

	_y = _y * cos_rotate - _z * sin_rotate;
	_z = _z * cos_rotate + init_y * sin_rotate;
}

void Point::rotateY(const double &oy) {
	const double cos_rotate = cos(to_radians(oy));
	const double sin_rotate = sin(to_radians(oy));

	const double init_x = _x;

	_x = _x * cos_rotate - _z * sin_rotate;
	_z = _z * cos_rotate + init_x * sin_rotate;
}

void Point::rotateZ(const double &oz) {
	const double cos_rotate = cos(to_radians(oz));
	const double sin_rotate = sin(to_radians(oz));

	const double init_x = _x;

	_x = _x * cos_rotate - _y * sin_rotate;
	_y = _y * cos_rotate + init_x * sin_rotate;
}


double Point::getX() const {
	return _x;
}

double Point::getY() const {
	return _y;
}

double Point::getZ() const {
	return _z;
}


void Point::setX(double const &x) {
	_x = x;
}

void Point::setY(double const &y) {
	_y = y;
}

void Point::setZ(double const &z) {
	_z = z;
}


bool Point::operator==(const Point &point) const noexcept {
	return (_x == point.getX()) &&
		   (_y == point.getY()) &&
		   (_z == point.getZ());
}

bool Point::operator!=(const Point &point) const noexcept {
	return !(*this == point);
}

Point Point::operator+(const Point &point) {
	Point result(*this);
	result.setX(_x + point.getX());
	result.setY(_y + point.getY());
	result.setZ(_z + point.getZ());
	return result;
}

Point Point::operator-(const Point &second_point) {
	Point result(*this);
	result.setX(_x - second_point.getX());
	result.setY(_y - second_point.getY());
	result.setZ(_z - second_point.getZ());
	return result;
}


Point Point::operator-() const {
	Point result(*this);
	result.setX(-_x);
	result.setY(-_y);
	result.setZ(-_z);
	return result;
}

Point Point::operator+() const {
	Point result(*this);
	return result;
}

Point Point::addCenter(const Point &center) {
	return (*this) + center;
}
