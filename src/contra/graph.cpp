#include "graph.hpp"

#include <numeric>

namespace contra {

graph_t graph(const std::vector<int> & parent)
{
  auto n = parent.size();

  // count the edges and the roots
  std::vector<int> counts(n,0);
  size_t nroots = 0;
  
  for (int i=0; i<n; ++i) {
    auto p = parent[i];
    if (p >= 0) counts[p]++;
  }

  // create the storage for the graph
  graph_t g;
  g.roots.reserve(nroots);
  g.offsets.resize(n+1);
  g.offsets[0] = 0;
  std::partial_sum(counts.begin(), counts.end(), std::next(g.offsets.begin()));
  g.indices.resize(g.offsets.back());

  std::fill(counts.begin(), counts.end(), 0);
  
  for (int i=0; i<n; ++i) {
    auto p = parent[i];
    if (p >= 0) {
      auto pos = g.offsets[p] + counts[p];
      g.indices[pos] = i;
      counts[p]++;
    }
    else {
      g.roots.push_back(i);
    }
  }

  return g;
}

} // namespace

