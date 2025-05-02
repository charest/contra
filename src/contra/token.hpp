#ifndef CONTRA_TOKEN_HPP
#define CONTRA_TOKEN_HPP

#include <iostream>
#include <map>
#include <string>
#include <unordered_set>
#include <unordered_map>

namespace contra {
  
#define TOKEN_NOT_FOUND -1000


//==============================================================================
// Helper class to return search result
//==============================================================================
struct token_map_t {
  std::unordered_map<std::string, int> str_to_enum;

  void add(int tok, const std::string & str)
  {
    str_to_enum[str] = tok;
  }
  
  void add(int tok)
  {
    str_to_enum[std::string(1,tok)] = tok;
  }

  int find(const std::string & str) const
  {
    auto it = str_to_enum.find(str);
    if (it != str_to_enum.end()) return it->second;
    return TOKEN_NOT_FOUND;
  }
};

} // namespace

#endif // CONTRA_TOKEN_HPP
