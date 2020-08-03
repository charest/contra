#ifndef CONTRA_SERIALIZER_HPP
#define CONTRA_SERIALIZER_HPP

#include "utils/builder.hpp"

#include <string>

namespace contra {

//==============================================================================
// Serializer
//==============================================================================
class Serializer {
  
protected:

  utils::BuilderHelper & TheHelper_;
  
  llvm::IRBuilder<> & Builder_;
  llvm::LLVMContext & TheContext_;

  llvm::Type* SizeType_ = nullptr;

public:

  Serializer(utils::BuilderHelper & TheHelper);

  llvm::Value* offsetPointer(llvm::Value*, llvm::Value*);

  virtual llvm::Value* getSize(llvm::Module&, llvm::Value*, llvm::Type*);
  virtual llvm::Value* serialize(
      llvm::Module&, 
      llvm::Value*,
      llvm::Value*,
      llvm::Value*);
  virtual llvm::Value* deserialize(
      llvm::Module&,
      llvm::AllocaInst*,
      llvm::Value*,
      llvm::Value*);

  virtual ~Serializer() = default;
};

//==============================================================================
// Array serializer
//==============================================================================
class ArraySerializer : public Serializer {

  llvm::Type* PtrType_ = nullptr;
  llvm::Type* LengthType_ = nullptr;

public:
  
  ArraySerializer(
      utils::BuilderHelper & TheHelper,
      llvm::StructType* ArrayType);

  llvm::Value* getSize(llvm::Module&, llvm::Value*, llvm::Type*) override;
  llvm::Value* serialize(
      llvm::Module&,
      llvm::Value*,
      llvm::Value*,
      llvm::Value*) override;
  llvm::Value* deserialize(
      llvm::Module&,
      llvm::AllocaInst*,
      llvm::Value*,
      llvm::Value*) override;
};

} // namespace

#endif // CONTRA_SERIALIZER_HPP
