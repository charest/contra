#ifndef CONTRA_GRAPH_HPP
#define CONTRA_GRAPH_HPP

#include <vector>

namespace contra {

struct graph_t {
  std::vector<int> indices;
  std::vector<int> offsets;
  std::vector<int> roots;

  int size(int i) const { return offsets[i+1] - offsets[i]; }
  int operator()(int i, int c) const { return indices[offsets[i]+c]; }
};

graph_t graph(const std::vector<int> & parent);

}

#endif

