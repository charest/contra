#ifndef CONTRA_LEGION_HPP
#define CONTRA_LEGION_HPP

#include "tasking.hpp"

namespace llvm {
class AllocaInst;
}

namespace contra {

class LegionTasker : public AbstractTasker {
protected:
  
  llvm::Type* VoidPtrType_ = nullptr;
  llvm::Type* VoidType_ = nullptr;
  llvm::Type* ByteType_ = nullptr;
  llvm::Type* BoolType_ = nullptr;
  llvm::Type* Int32Type_ = nullptr;
  llvm::Type* SizeType_ = nullptr;
  llvm::Type* ProcIdType_ = nullptr;
  llvm::Type* RealmIdType_ = nullptr;
  llvm::Type* NumRegionsType_ = nullptr;
  llvm::Type* TaskIdType_ = nullptr;
  llvm::Type* TaskVariantIdType_ = nullptr;
  llvm::Type* MapperIdType_ = nullptr;
  llvm::Type* MappingTagIdType_ = nullptr;
  
  llvm::StructType* TaskType_ = nullptr;
  llvm::StructType* RegionType_ = nullptr;
  llvm::StructType* ContextType_ = nullptr;
  llvm::StructType* RuntimeType_ = nullptr;
  llvm::StructType* ExecSetType_ = nullptr;
  llvm::StructType* LayoutSetType_ = nullptr;
  llvm::StructType* PredicateType_ = nullptr;
  llvm::StructType* LauncherType_ = nullptr;
  llvm::StructType* FutureType_ = nullptr;
  llvm::StructType* TaskConfigType_ = nullptr;
  llvm::StructType* TaskArgsType_ = nullptr;
  
  llvm::AllocaInst* ContextAlloca_ = nullptr;
  llvm::AllocaInst* RuntimeAlloca_ = nullptr;

public:
 
  LegionTasker(llvm::IRBuilder<> & TheBuilder, llvm::LLVMContext & TheContext);

  virtual PreambleResult taskPreamble(llvm::Module &, const std::string &,
      llvm::Function*) override;
  virtual void taskPostamble(llvm::Module &, llvm::Value*) override;
  
  virtual void preregisterTask(llvm::Module &, const std::string &, const TaskInfo &) override;
  virtual void postregisterTask(llvm::Module &, const std::string &, const TaskInfo &) override;
  
  virtual void setTopLevelTask(llvm::Module &, int) override;
  
  virtual llvm::Value* startRuntime(llvm::Module &, int, char **) override;
  
  virtual llvm::Value* launch(llvm::Module &, const std::string &, int,
      const std::vector<llvm::Value*> &, const std::vector<llvm::Value*> &) override;
  
  virtual llvm::Value* getFuture(llvm::Module &, llvm::Value*, llvm::Type*, llvm::Value*) override;

  virtual ~LegionTasker() = default;

protected:

  void reset() {
    ContextAlloca_ = nullptr;
    RuntimeAlloca_ = nullptr;
  }

  llvm::StructType* createOpaqueType(const std::string &, llvm::LLVMContext &);
  llvm::StructType* createTaskConfigOptionsType(const std::string &, llvm::LLVMContext &);
  llvm::StructType* createTaskArgumentsType(const std::string &, llvm::LLVMContext &);

};

} // namepsace

#endif // LIBRT_LEGION_HPP
