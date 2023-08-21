#ifndef LAB_03_COMPOSITE_H
#define LAB_03_COMPOSITE_H

#include "object.h"
#include "visitor.h"
#include "scene.h"

class Composite: public BaseObject {
public:
	Composite() = default;
	explicit Composite(std::shared_ptr<BaseObject> &element);
	explicit Composite(const std::vector<std::shared_ptr<BaseObject>> &vector);

	friend std::vector<std::shared_ptr<BaseObject>> Scene::getModels();

	bool isVisible() override { return false; };
	bool isComposite() override { return true; };

	bool add(const std::shared_ptr<BaseObject> &element) override;
	bool remove(const Iterator &iter) override;


	__device__ void accept(std::shared_ptr<Visitor> visitor) override;

	Iterator begin() override;
	Iterator end() override;

protected:
	std::vector<std::shared_ptr<BaseObject>> _elements;
};

#endif //LAB_03_COMPOSITE_H
