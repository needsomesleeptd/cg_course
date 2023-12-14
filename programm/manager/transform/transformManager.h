#ifndef LAB_03_TRANSFORMMANAGER_H
#define LAB_03_TRANSFORMMANAGER_H

#include "baseManager.h"
#include "object.h"
#include "visitor.h"

class TransformManager: public BaseManager {
public:
	TransformManager() = default;
	TransformManager(const TransformManager &manager) = delete;
	TransformManager &operator = (const TransformManager &manager) = delete;
	~TransformManager() = default;

	void setParams(const TransformParams transformParams);

	static void moveObject(const std::shared_ptr<BaseObject> &object, const double &dx, const double &dy, const double &dz);
	static void scaleObject(const std::shared_ptr<BaseObject> &object, const double &kx, const double &ky, const double &kz);
	static void rotateObject(const std::shared_ptr<BaseObject> &object, const double &ox, const double &oy, const double &oz);
	void transformObject(const std::shared_ptr<BaseObject> &object, const TransformParams &transformParams);



private:
	TransformParams _transformParams;
};

#endif //LAB_03_TRANSFORMMANAGER_H
