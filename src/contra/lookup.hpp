#ifndef CONTRA_LOOKUP_HPP
#define CONTRA_LOOKUP_HPP

#include <string>
#include <map>

namespace contra {
  
//==============================================================================
// Insert result
//==============================================================================
template<typename T>
class InsertResult {
  T* Pointer_ = nullptr;
  bool IsInserted_ = false;
public:
  InsertResult(T* Pointer, bool IsInerted)
    : Pointer_(Pointer), IsInserted_(IsInerted) {}
  auto get() const { return Pointer_; }
  auto isInserted() const { return IsInserted_; }
};

//==============================================================================
// Find result
//==============================================================================
template<typename T>
class FindResult {
  T* Pointer_ = nullptr;
  bool IsFound_ = false;
public:
  FindResult() = default;
  FindResult(T* Pointer, bool IsFound)
    : Pointer_(Pointer), IsFound_(IsFound) {}
  auto get() const { return Pointer_; }
  auto isFound() const { return IsFound_; }
  operator bool() { return IsFound_; }
};

//==============================================================================
// The symbol table
//==============================================================================
template<typename T>
class LookupTable {

  std::map<std::string, T> LookupTable_;

public:

  using FindResult = FindResult<T>;
  using InsertResult = InsertResult<T>;

  auto & operator[](const std::string & Name) {
    return LookupTable_[Name]; 
  }

  InsertResult insert(const std::string & Name, T && Lookup) {
    // search first
    auto it = find(Name);
    if (it) return {it.get(), false};
    // otherwise insert
    auto res = LookupTable_.emplace(Name, std::move(Lookup));
    return {&res.first->second, res.second};
  }

  FindResult find(const std::string & Name) {
    auto it = LookupTable_.find(Name);
    if (it == LookupTable_.end())  {
      return {nullptr, false};
    }
    return {&it->second, true};
  }

  void erase(const std::string & Name) {
    auto it = LookupTable_.find(Name);
    if (it != LookupTable_.end()) 
      LookupTable_.erase(it);
  }
    
  bool has(const std::string & Name) {
    auto it = LookupTable_.find(Name);
    return (it != LookupTable_.end());
  }

};

} // namespace contra


#endif // CONTRA_LOOKUP_HPP
