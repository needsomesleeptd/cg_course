#ifndef LAB_03_EDGE_H
#define LAB_03_EDGE_H

#include <cstddef>

class Edge {
public:
	Edge() = default;
	Edge(const std::size_t &startIndex, const std::size_t &endIndex);
	Edge(const Edge &edge) = default;
	~Edge() = default;

	std::size_t getStartIndex() const;
	std::size_t getEndIndex() const;

	void setStartIndex(const std::size_t &startIndex);
	void setEndIndex(const std::size_t &endIndex);

private:
	std::size_t _startIndex;
	std::size_t _endIndex;
};


#endif //LAB_03_EDGE_H
