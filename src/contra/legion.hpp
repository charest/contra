#ifndef CONTRA_LEGION_HPP
#define CONTRA_LEGION_HPP

#include "tasking.hpp"
#include "librt/dllexport.h"

#include "legion/legion_c.h"

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
  llvm::Type* CharType_ = nullptr;
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
  llvm::Type* IndexSpaceIdType_ = nullptr;
  llvm::Type* IndexTreeIdType_ = nullptr;
  llvm::Type* TypeTagType_ = nullptr;
  llvm::Type* FieldSpaceIdType_ = nullptr;
  llvm::Type* FieldIdType_ = nullptr;
  llvm::Type* RegionTreeIdType_ = nullptr;
  llvm::Type* IndexPartitionIdType_ = nullptr;
  
  llvm::ArrayType* Point1dType_ = nullptr;
  
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
  llvm::StructType* IndexSpaceType_ = nullptr;
  llvm::StructType* FieldSpaceType_ = nullptr;
  llvm::StructType* FieldAllocatorType_ = nullptr;
  llvm::StructType* LogicalRegionType_ = nullptr;
  llvm::StructType* IndexPartitionType_ = nullptr;
  llvm::StructType* LogicalPartitionType_ = nullptr;
  llvm::StructType* AccessorArrayType_ = nullptr;
  llvm::StructType* ByteOffsetType_ = nullptr;

  llvm::StructType* IndexSpaceDataType_ = nullptr;
  llvm::StructType* FieldDataType_ = nullptr;
  llvm::StructType* AccessorDataType_ = nullptr;
  llvm::StructType* PartitionDataType_ = nullptr;

  struct TaskEntry {
    llvm::AllocaInst* ContextAlloca = nullptr;
    llvm::AllocaInst* RuntimeAlloca = nullptr;
    llvm::AllocaInst* PartInfoAlloca = nullptr;
  };

  std::forward_list<TaskEntry> TaskAllocas_;

  enum class ArgType : char {
    None = 0,
    Future,
    Field
  };

public:
 
  LegionTasker(
      llvm::IRBuilder<> & TheBuilder,
      llvm::LLVMContext & TheContext);

  virtual PreambleResult taskPreamble(
      llvm::Module &,
      const std::string &,
      llvm::Function*) override;

  virtual PreambleResult taskPreamble(
      llvm::Module &,
      const std::string &,
      const std::vector<std::string> &,
      const std::vector<llvm::Type*> &,
      bool,
      const std::map<std::string, VariableType> & = {}) override;

  virtual void taskPostamble(
      llvm::Module &,
      llvm::Value*) override;
  
  virtual void preregisterTask(
      llvm::Module &,
      const std::string &,
      const TaskInfo &) override;
  virtual void postregisterTask(
      llvm::Module &,
      const std::string &,
      const TaskInfo &) override;
  
  virtual void setTopLevelTask(llvm::Module &, int) override;
  
  virtual llvm::Value* startRuntime(
      llvm::Module &,
      int,
      char **) override;
  
  virtual llvm::Value* launch(
      llvm::Module &,
      const std::string &,
      int,
      const std::vector<llvm::Value*> &) override;
  virtual llvm::Value* launch(
      llvm::Module &,
      const std::string &,
      int,
      const std::vector<llvm::Value*> &,
      const std::vector<llvm::Value*> &,
      llvm::Value*,
      bool) override;
  
  virtual llvm::Type* getFutureType() const
  { return FutureType_; }

  virtual bool isFuture(llvm::Value*) const override;
  virtual llvm::AllocaInst* createFuture(
      llvm::Module &,
      llvm::Function*,
      const std::string &) override;
  virtual llvm::Value* loadFuture(
      llvm::Module &,
      llvm::Value*,
      llvm::Type*) override;
  virtual void destroyFuture(
    llvm::Module &,
    llvm::Value*) override;
  virtual void toFuture(
      llvm::Module &,
      llvm::Value*,
      llvm::Value*) override;
  virtual void copyFuture(
      llvm::Module &,
      llvm::Value*,
      llvm::Value*) override;
  
  virtual llvm::Type* getFieldType() const override
  { return FieldDataType_; }

  virtual bool isField(llvm::Value*) const override;
  virtual llvm::AllocaInst* createField(
      llvm::Module &,
      llvm::Function*,
      const std::string &,
      llvm::Type*,
      llvm::Value*,
      llvm::Value*) override;
  virtual void createField(
      llvm::Module &,
      llvm::Value*, 
      const std::string &,
      llvm::Type*,
      llvm::Value*,
      llvm::Value*) override;
  virtual void destroyField(llvm::Module &, llvm::Value*) override;
  

  virtual bool isRange(llvm::Type*) const override;
  virtual bool isRange(llvm::Value*) const override;
  virtual llvm::AllocaInst* createRange(
      llvm::Module &,
      llvm::Function*,
      const std::string &,
      llvm::Value*,
      llvm::Value*,
      llvm::Value*) override;
  virtual llvm::AllocaInst* createRange(
      llvm::Module &,
      llvm::Function*,
      llvm::Value*,
      const std::string &) override;
  virtual llvm::AllocaInst* createRange(
      llvm::Module &,
      llvm::Function*,
      llvm::Type*,
      llvm::Value*,
      const std::string &) override;
  virtual void destroyRange(llvm::Module &, llvm::Value*) override;
  virtual llvm::Value* getRangeSize(
      llvm::Module &,
      llvm::Value*) override;
  virtual llvm::Value* getRangeStart(
      llvm::Module &,
      llvm::Value*) override;
  virtual llvm::Value* getRangeEnd(
      llvm::Module &,
      llvm::Value*) override;
  virtual llvm::Value* loadRangeValue(
      llvm::Module &,
      llvm::Type*,
      llvm::Value*,
      llvm::Value*) override;
  
  virtual llvm::Type* getRangeType() const
  { return IndexSpaceDataType_; }

  virtual bool isAccessor(llvm::Type*) const override;
  virtual bool isAccessor(llvm::Value*) const override;
  virtual void storeAccessor(
      llvm::Module &,
      llvm::Value*,
      llvm::Value*,
      llvm::Value*) const override;
  virtual llvm::Value* loadAccessor(
      llvm::Module &,
      llvm::Type*,
      llvm::Value*,
      llvm::Value*) const override;
  virtual void destroyAccessor(llvm::Module &, llvm::Value*) override;

  virtual llvm::Type* getAccessorType() const
  { return AccessorDataType_; }
  
  virtual llvm::AllocaInst* partition(
      llvm::Module &,
      llvm::Function*,
      llvm::Value*,
      llvm::Value*) override;
  virtual llvm::AllocaInst* partition(
      llvm::Module &,
      llvm::Function*,
      llvm::Value*,
      llvm::Type*,
      llvm::Value*,
      bool) override;
  
  virtual bool isPartition(llvm::Type*) const;
  virtual bool isPartition(llvm::Value*) const;
  
  virtual llvm::Type* getPartitionType() const
  { return IndexPartitionType_; }
  
  virtual llvm::Type* getPointType() const override
  { return Point1dType_; }

  virtual llvm::Value* makePoint(std::intmax_t) const override;
  virtual llvm::Value* makePoint(llvm::Value*) const override;

  virtual ~LegionTasker() = default;

protected:

  auto & getCurrentTask() { return TaskAllocas_.front(); }
  const auto & getCurrentTask() const { return TaskAllocas_.front(); }

  auto & startTask() { 
    TaskAllocas_.push_front({});
    return getCurrentTask();
  }
  void finishTask() { TaskAllocas_.pop_front(); }

  auto isInsideTask() { return !TaskAllocas_.empty(); }

  llvm::StructType* createOpaqueType(const std::string &, llvm::LLVMContext &);
  llvm::StructType* createTaskConfigOptionsType(llvm::LLVMContext &);
  llvm::StructType* createTaskArgumentsType(llvm::LLVMContext &);
  llvm::StructType* createDomainPointType(llvm::LLVMContext &);
  llvm::StructType* createRect1dType(llvm::LLVMContext &);
  llvm::StructType* createDomainRectType(llvm::LLVMContext &);
  llvm::StructType* createIndexSpaceType(llvm::LLVMContext &);
  llvm::StructType* createFieldSpaceType(llvm::LLVMContext &);
  llvm::StructType* createLogicalRegionType(llvm::LLVMContext &);
  llvm::StructType* createIndexPartitionType(llvm::LLVMContext &);
  llvm::StructType* createLogicalPartitionType(llvm::LLVMContext &);
  llvm::StructType* createByteOffsetType(llvm::LLVMContext &);

  llvm::StructType* createIndexSpaceDataType(llvm::LLVMContext &);
  llvm::StructType* createFieldDataType(llvm::LLVMContext &);
  llvm::StructType* createAccessorDataType(llvm::LLVMContext &);
  llvm::StructType* createPartitionDataType(llvm::LLVMContext &);

  llvm::AllocaInst* createPredicateTrue(llvm::Module &);
  
  llvm::AllocaInst* createGlobalArguments(
      llvm::Module &,
      const std::vector<llvm::Value*> &);
  
  void createGlobalFutures(
    llvm::Module &,
    llvm::Value*,
    const std::vector<llvm::Value*> &,
    bool IsIndex);
  
  void createFieldArguments(
    llvm::Module &,
    llvm::Value*,
    const std::vector<llvm::Value*> &,
    const std::vector<llvm::Value*> & = {},
    llvm::Value* = nullptr,
    llvm::Value* = nullptr);
  
  llvm::AllocaInst* createOpaqueType(
      llvm::Module&,
      llvm::StructType*,
      const std::string &,
      const std::string & = "");

  void destroyOpaqueType(
      llvm::Module&,
      llvm::Value*,
      const std::string &,
      const std::string & = "");

  void destroyGlobalArguments(llvm::Module&, llvm::AllocaInst*);

  void createRegistrationArguments(
      llvm::Module&,
      llvm::AllocaInst*&,
      llvm::AllocaInst*&,
      llvm::AllocaInst*&);

  llvm::AllocaInst* createPartitionInfo(llvm::Module&);
  void pushPartitionInfo(llvm::Module&, llvm::AllocaInst*);
  void popPartitionInfo(llvm::Module&, llvm::AllocaInst*);
  void destroyPartitionInfo(llvm::Module&);
};

} // namepsace

#endif // LIBRT_LEGION_HPP
