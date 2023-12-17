#include "composite.h"

Composite::Composite(std::shared_ptr<BaseObject> &element) {
	_elements.push_back(element);
}

Composite::Composite(const std::vector<std::shared_ptr<BaseObject>> &vector) {
	_elements = vector;
}

bool Composite::add(const std::shared_ptr<BaseObject> &element) {
	_elements.push_back(element);
	return true;
}

bool Composite::remove(const Iterator &iter) {
	_elements.erase(iter);
	return true;
}



Iterator Composite::begin() {
	return _elements.begin();
}

Iterator Composite::end() {
	return _elements.end();
}

void Composite::accept(std::shared_ptr<Visitor> visitor) {
	for (const auto &element : _elements) {
		element->accept(visitor);
	}
}
