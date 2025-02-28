#ifndef CONTRA_LEGION_HPP
#define CONTRA_LEGION_HPP

#ifdef HAVE_LEGION

#include "tasking.hpp"
#include "reductions.hpp"

namespace contra {

////////////////////////////////////////////////////////////////////////////////
// Legion Reduction
////////////////////////////////////////////////////////////////////////////////
class LegionReduceInfo : public AbstractReduceInfo {
  int Id_ = -1;

  llvm::FunctionType* ApplyT_ = nullptr;
  llvm::FunctionType* FoldT_ = nullptr;
  llvm::FunctionType* InitT_ = nullptr;
  
  std::string ApplyN_;
  std::string FoldN_;
  std::string InitN_;

  std::size_t DataSize_ = 0;

public:

  LegionReduceInfo(
      int Id,
      llvm::Function* Apply,
      llvm::Function* Fold,
      llvm::Function* Init,
      std::size_t DataSize)
    : 
      Id_(Id),
      ApplyT_(Apply->getFunctionType()),
      FoldT_(Fold->getFunctionType()),
      InitT_(Init->getFunctionType()),
      ApplyN_(Apply->getName()),
      FoldN_(Fold->getName()),
      InitN_(Init->getName()),
      DataSize_(DataSize)
  {}

  auto getId() const { return Id_; }
  
  auto getApplyType() const { return ApplyT_; }
  const auto & getApplyName() const { return ApplyN_; }

  auto getFoldType() const { return FoldT_; }
  const auto & getFoldName() const { return FoldN_; }

  auto getInitType() const { return InitT_; }
  const auto & getInitName() const { return InitN_; }

  auto getDataSize() const { return DataSize_; }

};


////////////////////////////////////////////////////////////////////////////////
// Legion Tasker
////////////////////////////////////////////////////////////////////////////////
class LegionTasker : public AbstractTasker {
protected:
  
  llvm::Type* CharType_ = nullptr;
  llvm::Type* OffType_ = nullptr;
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
    llvm::AllocaInst* TimerAlloca = nullptr;
  };

  std::forward_list<TaskEntry> TaskAllocas_;

  enum class ArgType : char {
    None = 0,
    Future,
    Field
  };

public:
 
  LegionTasker(utils::BuilderHelper & TheHelper);

  virtual PreambleResult taskPreamble(
      llvm::Module &,
      const std::string &,
      llvm::Function*) override;

  virtual PreambleResult taskPreamble(
      llvm::Module &,
      const std::string &,
      const std::vector<std::string> &,
      const std::vector<llvm::Type*> &,
      llvm::Type*) override;

  virtual void taskPostamble(
      llvm::Module &,
      llvm::Value*,
      bool) override;
  
  virtual void preregisterTask(
      llvm::Module &,
      const std::string &,
      const TaskInfo &) override;
  virtual void postregisterTask(
      llvm::Module &,
      const std::string &,
      const TaskInfo &) override;
  
  virtual void setTopLevelTask(
      llvm::Module &,
      const TaskInfo &) override;
  
  virtual void startRuntime(llvm::Module &) override;
  
  virtual llvm::Value* launch(
      llvm::Module &,
      const TaskInfo &,
      const std::vector<llvm::Value*> &) override;
  virtual llvm::Value* launch(
      llvm::Module &,
      const TaskInfo &,
      std::vector<llvm::Value*>,
      const std::vector<llvm::Value*> &,
      llvm::Value*,
      const AbstractReduceInfo *) override;
  
  virtual llvm::Type* getFutureType(llvm::Type*) const override
  { return FutureType_; }

  virtual bool isFuture(llvm::Value*) const override;
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
  
  virtual llvm::Type* getFieldType(llvm::Type*) const override
  { return FieldDataType_; }

  virtual bool isField(llvm::Value*) const override;
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
      const std::string &,
      llvm::Value*,
      llvm::Value*,
      llvm::Value*) override;
  virtual void destroyRange(llvm::Module &, llvm::Value*) override;
  virtual llvm::Value* getRangeSize(llvm::Value*) override;
  virtual llvm::Value* getRangeStart(llvm::Value*) override;
  virtual llvm::Value* getRangeEnd(llvm::Value*) override;
  virtual llvm::Value* getRangeEndPlusOne(llvm::Value*) override;
  virtual llvm::Value* getRangeStep(llvm::Value*) override;
  virtual llvm::Value* loadRangeValue(
      llvm::Value*,
      llvm::Value*) override;
  
  virtual llvm::Type* getRangeType(llvm::Type*) const override
  { return IndexSpaceDataType_; }

  virtual bool isAccessor(llvm::Type*) const override;
  virtual bool isAccessor(llvm::Value*) const override;
  virtual void storeAccessor(
      llvm::Module &,
      llvm::Value*,
      llvm::Value*,
      llvm::Value*) override;
  virtual llvm::Value* loadAccessor(
      llvm::Module &,
      llvm::Type*,
      llvm::Value*,
      llvm::Value*) override;
  virtual void destroyAccessor(llvm::Module &, llvm::Value*) override;

  virtual llvm::Type* getAccessorType() const override
  { return AccessorDataType_; }
  
  virtual llvm::AllocaInst* createPartition(
      llvm::Module &,
      llvm::Value*,
      llvm::Value*,
      llvm::Value*) override;
  virtual llvm::AllocaInst* createPartition(
      llvm::Module &,
      llvm::Value*,
      llvm::Value*) override;
  
  virtual bool isPartition(llvm::Type*) const override;
  virtual bool isPartition(llvm::Value*) const override;
  
  virtual void destroyPartition(llvm::Module &, llvm::Value*) override;
  
  virtual llvm::Type* getPartitionType(llvm::Type*) const override
  { return IndexPartitionType_; }

  virtual std::unique_ptr<AbstractReduceInfo> createReductionOp(
      llvm::Module &,
      const std::string &,
      const std::vector<llvm::Type*> &,
      const std::vector<ReductionType> &) override;

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
    llvm::AllocaInst*,
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
      llvm::Type*,
      llvm::Value*,
      const std::string &,
      const std::string & = "");
  void destroyOpaqueType(
      llvm::Module&,
      llvm::AllocaInst*,
      const std::string &,
      const std::string & = "");
  
  PreambleResult taskPreamble(
      llvm::Module &,
      const std::string &,
      const std::vector<std::string> &,
      const std::vector<llvm::Type*> &,
      bool);

  void destroyGlobalArguments(llvm::AllocaInst*);

  void createRegistrationArguments(
      llvm::Module&,
      const TaskInfo & Task,
      llvm::AllocaInst*&,
      llvm::AllocaInst*&,
      llvm::AllocaInst*&);

  llvm::AllocaInst* createPartitionInfo(llvm::Module&);
  void pushPartitionInfo(llvm::Module&, llvm::AllocaInst*);
  void popPartitionInfo(llvm::Module&, llvm::AllocaInst*);
  void destroyPartitionInfo(llvm::Module&);
  
  void registerReductionOp(
    llvm::Module &TheModule,
    const AbstractReduceInfo * ReduceOp);

  llvm::Function* createReductionFunction(
      llvm::Module &TheModule,
      const std::string &,
      const std::string &,
      const std::vector<std::size_t> &,
      const std::vector<llvm::Type*> &,
      const std::vector<ReductionType> &);
  
};

} // namepsace

#endif // HAVE_LEGION
#endif // LIBRT_LEGION_HPP
