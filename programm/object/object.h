#ifndef LAB_03_OBJECT_H
#define LAB_03_OBJECT_H

#include <memory>
#include <vector>
#include "visitor.h"
#include "transform.h"

class BaseObject;
using Iterator = std::vector<std::shared_ptr<BaseObject>>::iterator;

class BaseObject {
public:
	BaseObject() = default;
	virtual ~BaseObject() = default;

	virtual bool isVisible() { return false; };
	virtual bool isComposite() { return false; };

	virtual void accept(std::shared_ptr<Visitor> visitor) = 0;


	virtual bool add(const std::shared_ptr<BaseObject> &) { return false; };
	virtual bool remove(const Iterator &) { return false; };

	virtual Iterator begin() { return Iterator(); };
	virtual Iterator end() { return Iterator(); };


};

class VisibleObject : public BaseObject {
public:
	VisibleObject() = default;
	~VisibleObject() = default;
	bool isVisible() override
	{
		return true;
	}
};

class InvisibleObject : public BaseObject
{
 public:
	InvisibleObject() = default;
	~InvisibleObject() = default;
	bool isVisible() override
	{
		return false;
	}
};

#endif //LAB_03_OBJECT_H
