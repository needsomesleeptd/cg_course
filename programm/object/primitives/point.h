#ifndef LAB_03_POINT_H
#define LAB_03_POINT_H

class Point {
public:
	Point() = default;
	Point(const double x, const double y, const double z);
	Point(const Point &point) = default;
	Point(const Point &&point) noexcept;
	~Point() = default;

	Point &operator=(const Point &point) = default;
	Point &operator=(Point &&point) noexcept;

	void move(const double &dx, const double &dy, const double &dz);
	void scale(const double &kx, const double &ky, const double &kz);
	void rotate(const double &ox, const double &oy, const double &oz);

	void rotateX(const double &ox);
	void rotateY(const double &oy);
	void rotateZ(const double &oz);

	double getX() const;
	double getY() const;
	double getZ() const;

	void setX(double const &src_x);
	void setY(double const &src_y);
	void setZ(double const &src_z);

	bool operator==(const Point &point) const noexcept;
	bool operator!=(const Point &point) const noexcept;

	Point operator+(const Point &point);
	Point operator-(const Point &point);

	Point operator-() const;
	Point operator+() const;

	Point addCenter(const Point &center);

private:
	double _x;
	double _y;
	double _z;
};

#endif //LAB_03_POINT_H
