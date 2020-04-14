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
  llvm::Type* FutureIdType_ = nullptr;
  llvm::Type* CoordType_ = nullptr;
  llvm::Type* Point1dType_ = nullptr;
  
  llvm::StructType* TaskType_ = nullptr;
  llvm::StructType* RegionType_ = nullptr;
  llvm::StructType* ContextType_ = nullptr;
  llvm::StructType* RuntimeType_ = nullptr;
  llvm::StructType* ExecSetType_ = nullptr;
  llvm::StructType* LayoutSetType_ = nullptr;
  llvm::StructType* PredicateType_ = nullptr;
  llvm::StructType* TaskLauncherType_ = nullptr;
  llvm::StructType* IndexLauncherType_ = nullptr;
  llvm::StructType* FutureType_ = nullptr;
  llvm::StructType* TaskConfigType_ = nullptr;
  llvm::StructType* TaskArgsType_ = nullptr;
  llvm::StructType* DomainPointType_ = nullptr;
  llvm::StructType* Rect1dType_ = nullptr;
  llvm::StructType* DomainRectType_ = nullptr;    
  llvm::StructType* ArgMapType_ = nullptr;
  llvm::StructType* FutureMapType_ = nullptr;

  struct TaskEntry {
    llvm::AllocaInst* ContextAlloca = nullptr;
    llvm::AllocaInst* RuntimeAlloca = nullptr;
  };

  std::forward_list<TaskEntry> TaskAllocas_;

public:
 
  LegionTasker(llvm::IRBuilder<> & TheBuilder, llvm::LLVMContext & TheContext);

  virtual PreambleResult taskPreamble(llvm::Module &, const std::string &,
      llvm::Function*) override;

  virtual PreambleResult taskPreamble(llvm::Module &, const std::string &,
      const std::vector<std::string> &, const std::vector<llvm::Type*> &,
      bool IsIndex=false) override;

  virtual void taskPostamble(llvm::Module &, llvm::Value*) override;
  
  virtual void preregisterTask(llvm::Module &, const std::string &, const TaskInfo &) override;
  virtual void postregisterTask(llvm::Module &, const std::string &, const TaskInfo &) override;
  
  virtual void setTopLevelTask(llvm::Module &, int) override;
  
  virtual llvm::Value* startRuntime(llvm::Module &, int, char **) override;
  
  virtual llvm::Value* launch(llvm::Module &, const std::string &, int,
      const std::vector<llvm::Value*> &, const std::vector<llvm::Value*> &) override;
  virtual llvm::Value* launch(llvm::Module &, const std::string &, int,
      const std::vector<llvm::Value*> &, const std::vector<llvm::Value*> &,
      llvm::Value*, llvm::Value*) override;
  
  virtual bool isFuture(llvm::Value*) const override;
  virtual llvm::Value* createFuture(llvm::Module &,llvm::Function*, const std::string &) override;
  virtual llvm::Value* loadFuture(llvm::Module &, llvm::Value*, llvm::Type*, llvm::Value*) override;
  virtual void destroyFuture(llvm::Module &, llvm::Value*) override;

  virtual ~LegionTasker() = default;

protected:

  auto & getCurrentTask() { return TaskAllocas_.front(); }
  const auto & getCurrentTask() const { return TaskAllocas_.front(); }

  auto & startTask() { 
    TaskAllocas_.push_front({});
    return getCurrentTask();
  }
  void finishTask() { TaskAllocas_.pop_front(); }

  llvm::StructType* createOpaqueType(const std::string &, llvm::LLVMContext &);
  llvm::StructType* createTaskConfigOptionsType(const std::string &, llvm::LLVMContext &);
  llvm::StructType* createTaskArgumentsType(const std::string &, llvm::LLVMContext &);
  llvm::StructType* createDomainPointType(const std::string &, llvm::LLVMContext &);
  llvm::StructType* createRect1dType(const std::string &, llvm::LLVMContext &);
  llvm::StructType* createDomainRectType(const std::string &, llvm::LLVMContext &);

  llvm::AllocaInst* createPredicateTrue(llvm::Module &);
  llvm::AllocaInst* createGlobalArguments(
      llvm::Module &,
      const std::vector<llvm::Value*> &,
      const std::vector<llvm::Value*> &,
      std::vector<unsigned> &);
  void createGlobalFutures(
    llvm::Module &,
    llvm::Value*,
    const std::vector<llvm::Value*> &,
    const std::vector<llvm::Value*> &,
    const std::vector<unsigned> &,
    bool IsIndex);
  llvm::AllocaInst* createOpaqueType(llvm::Module&, llvm::StructType*, const std::string &,
      const std::string & = "");
  void destroyOpaqueType(llvm::Module&, llvm::Value*, const std::string &,
      const std::string & = "");
  void destroyGlobalArguments(llvm::Module&, llvm::AllocaInst*);
  void createRegistrationArguments(llvm::Module&, llvm::AllocaInst*&,
      llvm::AllocaInst*&, llvm::AllocaInst*&);

};

} // namepsace

#endif // LIBRT_LEGION_HPP
