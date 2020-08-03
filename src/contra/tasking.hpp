#ifndef CONTRA_TASKING_HPP
#define CONTRA_TASKING_HPP

#include "serializer.hpp"
#include "reduceinfo.hpp"
#include "reductions.hpp"
#include "taskinfo.hpp"

#include "utils/builder.hpp"

#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace contra {

class VariableType;

//==============================================================================
// Main tasking interface
//==============================================================================
class AbstractTasker {

  unsigned TaskIdCounter_ = 0;
  unsigned ReduceIdCounter_ = 1;
  bool IsStarted_ = false;
  
  std::map<std::string, TaskInfo> TaskTable_;
  
protected:

  utils::BuilderHelper & TheHelper_;

  llvm::IRBuilder<> & Builder_;
  llvm::LLVMContext & TheContext_;

  Serializer DefaultSerializer_;
  std::map<llvm::Type*, std::unique_ptr<Serializer>> Serializer_;


  llvm::Type* VoidType_ = nullptr;
  llvm::Type* VoidPtrType_ = nullptr;
  llvm::Type* ByteType_ = nullptr;
  llvm::Type* BoolType_ = nullptr;
  llvm::Type* Int32Type_ = nullptr;
  llvm::Type* IntType_ = nullptr;
  llvm::Type* SizeType_ = nullptr;
  llvm::Type* RealType_ = nullptr;
  
  llvm::StructType* DefaultIndexSpaceType_ = nullptr;

public:
  
  AbstractTasker(utils::BuilderHelper & TheHelper);
  
  virtual ~AbstractTasker() = default;

  struct PreambleResult {
    llvm::Function* TheFunction;
    std::vector<llvm::AllocaInst*> ArgAllocas;
    llvm::AllocaInst* Index;
  };

  //----------------------------------------------------------------------------
  // abstraact interface
  virtual PreambleResult taskPreamble(
      llvm::Module &,
      const std::string &,
      llvm::Function*);

  virtual PreambleResult taskPreamble(
      llvm::Module &,
      const std::string &,
      const std::vector<std::string> &,
      const std::vector<llvm::Type*> &,
      llvm::Type*) = 0;

  virtual void taskPostamble(
      llvm::Module &,
      llvm::Value*,
      bool);

  virtual void preregisterTask(
      llvm::Module &,
      const std::string &,
      const TaskInfo &) {};

  virtual void postregisterTask(
      llvm::Module &,
      const std::string &,
      const TaskInfo &) {};
  
  virtual void setTopLevelTask(llvm::Module &, const TaskInfo &) = 0;
  virtual void startRuntime(llvm::Module &) = 0;
  virtual void stopRuntime(llvm::Module &) {}
  
  virtual llvm::Value* launch(
      llvm::Module &,
      const TaskInfo &,
      const std::vector<llvm::Value*> & = {});

  virtual llvm::Value* launch(
      llvm::Module &,
      const TaskInfo &,
      std::vector<llvm::Value*>,
      const std::vector<llvm::Value*> &,
      llvm::Value*,
      const AbstractReduceInfo * = nullptr) = 0;

  virtual llvm::Type* getFutureType(llvm::Type* Ty) const { return Ty; };
  virtual bool isFuture(llvm::Value*) const { return false; };

  virtual llvm::Value* loadFuture(
      llvm::Module &,
      llvm::Value* Val,
      llvm::Type*) {return Val;};
  virtual void destroyFuture(llvm::Module &, llvm::Value*) {};
  virtual void toFuture(llvm::Module &, llvm::Value*, llvm::Value*) {};
  virtual void copyFuture(llvm::Module &, llvm::Value*, llvm::Value*) {};

  virtual llvm::Type* getFieldType(llvm::Type*) const = 0;
  virtual bool isField(llvm::Value*) const = 0;
  virtual void createField(
      llvm::Module &,
      llvm::Value*, 
      const std::string &,
      llvm::Type*,
      llvm::Value*,
      llvm::Value*) = 0;
  virtual void destroyField(llvm::Module &, llvm::Value*) = 0;
  
  virtual bool isRange(llvm::Type*) const;
  virtual bool isRange(llvm::Value*) const;
  virtual llvm::AllocaInst* createRange(
      llvm::Module &,
      const std::string &,
      llvm::Value*,
      llvm::Value*,
      llvm::Value*);
  virtual void destroyRange(llvm::Module &, llvm::Value*) {}
  virtual llvm::Value* getRangeSize(llvm::Value*);
  virtual llvm::Value* getRangeStart(llvm::Value*);
  virtual llvm::Value* getRangeEnd(llvm::Value*);
  virtual llvm::Value* getRangeEndPlusOne(llvm::Value*);
  virtual llvm::Value* getRangeStep(llvm::Value*);
  virtual llvm::Value* loadRangeValue(
      llvm::Value*,
      llvm::Value*);
  virtual llvm::Type* getRangeType(llvm::Type*) const
  { return DefaultIndexSpaceType_; }

  virtual bool isAccessor(llvm::Type*) const = 0;
  virtual bool isAccessor(llvm::Value*) const = 0;
  virtual llvm::Type* getAccessorType() const = 0;
  virtual void storeAccessor(
      llvm::Module &,
      llvm::Value*,
      llvm::Value*,
      llvm::Value* = nullptr) const = 0;
  virtual llvm::Value* loadAccessor(
      llvm::Module &,
      llvm::Type*,
      llvm::Value*,
      llvm::Value* = nullptr) const = 0;
  virtual void destroyAccessor(llvm::Module &, llvm::Value*) = 0;
  
  virtual llvm::AllocaInst* createPartition(
      llvm::Module &,
      llvm::Value*,
      llvm::Value*,
      llvm::Value*) = 0;
  virtual llvm::AllocaInst* createPartition(
      llvm::Module &,
      llvm::Value*,
      llvm::Value*) = 0;
  
  virtual llvm::Type* getPartitionType(llvm::Type*) const = 0;
  virtual bool isPartition(llvm::Type*) const = 0;
  virtual bool isPartition(llvm::Value*) const = 0;
  
  virtual void destroyPartition(llvm::Module &, llvm::Value*) = 0;
  
  virtual std::unique_ptr<AbstractReduceInfo> createReductionOp(
      llvm::Module &,
      const std::string &,
      const std::vector<llvm::Type*> &,
      const std::vector<ReductionType> &) = 0;
  
  //----------------------------------------------------------------------------
  // Common public members
  
  template<typename T, typename...Args>
  void registerSerializer(llvm::Type* Ty, Args&&... As)
  {
    Serializer_.emplace(
        Ty,
        std::make_unique<T>(std::forward<Args>(As)...) );
  }

  // registration
  void preregisterTasks(llvm::Module &);
  void postregisterTasks(llvm::Module &);

  // startup interface
  void start(llvm::Module & TheModule);
  
  bool isStarted() const { return IsStarted_; }
  void setStarted() { IsStarted_ = true; }
  
  // Task table interface
  TaskInfo & insertTask(const std::string & Name, llvm::Function* F);

  bool isTask(const std::string & Name) const
  { return TaskTable_.count(Name); }

  TaskInfo popTask(const std::string & Name);

  const TaskInfo & getTask(const std::string & Name) const;

  // future interface
  void destroyFutures(llvm::Module &, const std::vector<llvm::Value*> &);

  // field interface
  void destroyFields(llvm::Module &, const std::vector<llvm::Value*> &);

  // range interface
  void destroyRanges(llvm::Module &, const std::vector<llvm::Value*> &);
  
  // accessor interface
  void destroyAccessors(llvm::Module &, const std::vector<llvm::Value*> &);

  // partition interface
  void destroyPartitions(llvm::Module &, const std::vector<llvm::Value*> &);


protected:
  
  //----------------------------------------------------------------------------
  // Private members
  
  auto makeTaskId() { return TaskIdCounter_++; }
  auto makeReductionId() { return ReduceIdCounter_++; }
  
  llvm::StructType* createDefaultIndexSpaceType();

  // helpers
  llvm::Type* reduceStruct(llvm::StructType *, const llvm::Module &) const;
  llvm::Type* reduceArray(llvm::ArrayType *, const llvm::Module &) const;
  llvm::Value* sanitize(llvm::Value*, const llvm::Module &) const;
  void sanitize(std::vector<llvm::Value*> & Vs, const llvm::Module &) const;
  llvm::Value* load(llvm::Value *, const llvm::Module &, std::string) const;
  void store(llvm::Value*, llvm::Value *) const;

  // Serializer
  llvm::Value* getSerializedSize(llvm::Module&, llvm::Value*, llvm::Type*);
  llvm::Value* serialize(
      llvm::Module&,
      llvm::Value*,
      llvm::Value*,
      llvm::Value* = nullptr);
  llvm::Value* deserialize(
      llvm::Module&,
      llvm::AllocaInst*,
      llvm::Value*,
      llvm::Value* = nullptr);

  // reductions
  llvm::Constant* initReduce(llvm::Type*, ReductionType);
  llvm::Value* applyReduce(llvm::Module&, llvm::Value*, llvm::Value*, ReductionType);
  llvm::Value* foldReduce(llvm::Module&, llvm::Value*, llvm::Value*, ReductionType);

};

} // namespace

#endif // CONTRA_TASKING_HPP
