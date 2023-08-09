#include "edge.h"

Edge::Edge(const std::size_t &startIndex, const std::size_t &endIndex) {
	setStartIndex(startIndex);
	setEndIndex(endIndex);
}

std::size_t Edge::getStartIndex() const {
	return _startIndex;
}

std::size_t Edge::getEndIndex() const {
	return _endIndex;
}

void Edge::setStartIndex(const std::size_t &startIndex) {
	_startIndex = startIndex;
}

void Edge::setEndIndex(const std::size_t &endIndex) {
	_endIndex = endIndex;
}
