#include "range.hpp"
#include "llvm_includes.hpp"

#include "contra/symbols.hpp"
#include "utils/llvm_utils.hpp"

#include <cstdlib>
#include <iostream>

extern "C" {

/// simple range type
struct range_t {
  int_t start = 0;
  int_t end = 0;
  void * data = nullptr;
};

} // extern

namespace librt {

using namespace contra;
using namespace llvm;
using namespace utils;

Type* Range::RangeType = nullptr;
const std::string Range::Name = "range";

//==============================================================================
// Create the dopevector type 
//==============================================================================
Type * createRangeType(LLVMContext & TheContext)
{
  auto RangeType = StructType::create( TheContext, "dopevector_t" );
  auto VoidPointerType = llvmType<void*>(TheContext);
  auto IntType = llvmType<int_t>(TheContext);

  std::vector<Type*> members{ IntType, IntType, VoidPointerType }; 
  RangeType->setBody( members );

  return RangeType;
}

//==============================================================================
// Sets up whatever is needed for allocate
//==============================================================================
void Range::setup(LLVMContext & TheContext)
{ 
  if (!RangeType)
    RangeType = createRangeType(TheContext);
}

}
