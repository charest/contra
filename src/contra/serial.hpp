#ifndef CONTRA_SERIAL_HPP
#define CONTRA_SERIAL_HPP

#include "config.hpp"

#include "tasking.hpp"

namespace contra {

class SerialTasker : public AbstractTasker {

  llvm::StructType* FieldDataType_ = nullptr;

  const TaskInfo * TopLevelTask_ = nullptr;

public:
 
  SerialTasker(utils::BuilderHelper & TheHelper);

  virtual void startRuntime(
      llvm::Module &,
      int,
      char **) override;
  
  virtual PreambleResult taskPreamble(
      llvm::Module &,
      const std::string &,
      const std::vector<std::string> &,
      const std::vector<llvm::Type*> &) override;
  
  using AbstractTasker::launch;
  
  virtual llvm::Value* launch(
      llvm::Module &,
      const TaskInfo &,
      std::vector<llvm::Value*>,
      const std::vector<llvm::Value*> &,
      llvm::Value*,
      bool,
      int) override;
  
  virtual void setTopLevelTask(
      llvm::Module &,
      const TaskInfo & TaskI) override
  { TopLevelTask_ = &TaskI; }
  
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

protected:
  llvm::StructType* createFieldDataType();
};

} // namepsace

#endif // CONTRA_SERIAL_HPP
