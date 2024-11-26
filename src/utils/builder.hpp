#ifndef UTILS_BUILDER_HPP
#define UTILS_BUILDER_HPP

#include "config.hpp"

#include "llvm_utils.hpp"

#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/IRBuilder.h"

namespace utils {

class BuilderHelper {
  using AllocaInst = llvm::AllocaInst;
  using CallInst = llvm::CallInst;
  using FunctionCallee = llvm::FunctionCallee;
  using Function = llvm::Function;
  using Instruction = llvm::Instruction;
  using Module = llvm::Module;
  using Type = llvm::Type;
  using Value = llvm::Value;
  using ArrayType = llvm::ArrayType;

  std::unique_ptr<llvm::LLVMContext> Context_;
  std::unique_ptr<llvm::IRBuilder<>> Builder_;
  
  llvm::LLVMContext * TheContext_ = nullptr;

public:
  BuilderHelper() : 
    Context_(std::make_unique<llvm::LLVMContext>()),
    TheContext_(Context_.get()),
    Builder_(std::make_unique<llvm::IRBuilder<>>(*TheContext_)) 
  {}
  
  void setContext(llvm::LLVMContext * ContextPtr)
  {
    TheContext_ = ContextPtr;
    Builder_ = std::make_unique<llvm::IRBuilder<>>(*TheContext_);
    Context_.reset();
  }

  auto & getBuilder() { return *Builder_; }
  auto & getContext() { return *TheContext_; }

  Value* createCast(Value* FromVal, Type* ToType); 
  Value* createBitCast(Value* FromVal, Type* ToType);
  Value* createAddrSpaceCast(Value* FromVal, Type* ToType);

  Value* getAsValue(Value*);
  AllocaInst* getAsAlloca(Value*);

  Value* getElementPointer(Type*, Value*, unsigned);
  Value* getElementPointer(Type*, Value*, unsigned, unsigned);
  Value* getElementPointer(AllocaInst*, unsigned i, unsigned j);
  Value* getElementPointer(Type*, Value*, const std::vector<unsigned> &);

  Value* extractValue(Value*, unsigned);
  Value* extractValue(ArrayType*, Value*, unsigned);
  void insertValue(Value*, Value*, unsigned);

  Value* offsetPointer(Type*, Value*, Value*);

  Type* getAllocatedType(Value*);

  Value* getTypeSize(Type*, Type*);
  Value* getTypeSize(Value*, Type*);
  std::size_t getTypeSizeInBits(const llvm::Module & TheModule, Type* Ty);
 
  template<typename T>
  Value* getTypeSize(Type* ElementType)
  { return getTypeSize(ElementType, llvmType<T>(*TheContext_)); }

  AllocaInst* createEntryBlockAlloca(Type*, const llvm::Twine & = "");
  AllocaInst* createEntryBlockAlloca(Function*, Type*, const llvm::Twine & = "");

  Value* load(AllocaInst*, const std::string & ="");
[[deprecated]]
  Value* load(Value*, const std::string & ="");

  void increment(Type*, Value*, Value*, const std::string & = "");
  void increment(AllocaInst*, Value*, const std::string & = "");
  
  template<
    typename T,
    typename = std::enable_if_t< !std::is_pointer<T>::value >
    >
  void increment(
    Type* OffsetT,
    Value* OffsetPtr,
    T Offset,
    const std::string & Name = "")
  {
    auto OffsetV = llvmValue(*TheContext_, OffsetT, Offset);
    increment(OffsetT, OffsetPtr, OffsetV, Name);
  }

  template<
    typename T,
    typename = std::enable_if_t< !std::is_pointer<T>::value >
    >
  void increment(
    AllocaInst* OffsetA,
    T Offset,
    const std::string & Name = "")
  {
    auto OffsetT = OffsetA->getAllocatedType();
    auto OffsetV = llvmValue(*TheContext_, OffsetT, Offset);
    increment(OffsetT, OffsetA, OffsetV, Name);
  }
  
  Instruction* createMalloc(Type*, Value*, const std::string & ="");

  void createFree(Value*);

  FunctionCallee createFunction(
      Module &,
      const std::string &,
      Type*,
      const std::vector<Type*> & = {});

  CallInst* callFunction(
      Module &,
      const std::string &,
      Type*,
      const std::vector<Value*> & = {},
      const std::string & Str = "");

  CallInst* memCopy(Value* Dest, Value* Src, Value* Size);
  CallInst* memSet(Value* Dest, Value* Src, unsigned);

  Value* createMinimum(llvm::Module&, Value*, Value*, const std::string & = "");
  Value* createMaximum(llvm::Module&, Value*, Value*, const std::string & = "");
};

} // namespace

#endif // UTILS_BUILDER_HPP
