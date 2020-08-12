#ifndef CONTRA_UTILS_COMMUNICATOR_INTERFACE_HPP
#define CONTRA_UTILS_COMMUNICATOR_INTERFACE_HPP

#include "utils/builder.hpp"

namespace contra {

////////////////////////////////////////////////////////////////////////////////
/// Main virutal interface
////////////////////////////////////////////////////////////////////////////////
class Communicator
{
public:
  virtual ~Communicator() {}
  virtual Communicator& createCommunicator() = 0;
  virtual void init(int * argc, char ** argv[]) = 0;
  virtual void finalize() = 0;
  virtual void markTask(llvm::Module&) = 0;
  virtual void unmarkTask(llvm::Module&) = 0;
  virtual void pushRootGuard(llvm::Module&) = 0;
  virtual void popRootGuard(llvm::Module&) = 0;

  void setup(utils::BuilderHelper & Helper);

protected:
 utils::BuilderHelper * TheHelper_ = nullptr; 
 llvm::LLVMContext * TheContext_ = nullptr;
 llvm::IRBuilder<> * TheBuilder_ = nullptr;

 llvm::Type * VoidType_ = nullptr;
 llvm::Type * Int1Type_ = nullptr;
 llvm::Type * Int8Type_ = nullptr;
};

////////////////////////////////////////////////////////////////////////////////
/// The empty communicator
////////////////////////////////////////////////////////////////////////////////
class NoCommunicator : public Communicator
{
public:

  static NoCommunicator & getInstance() {
    static NoCommunicator instance;
    return instance;
  }
  
  NoCommunicator(NoCommunicator const&) = delete;
  void operator=(NoCommunicator const&)   = delete;

  Communicator& createCommunicator() override {
    return NoCommunicator::getInstance();
  }

  void init(int * argc, char ** argv[]) override {}
  void finalize() override {}
  void markTask(llvm::Module&) override {}
  void unmarkTask(llvm::Module&) override {}
  virtual void pushRootGuard(llvm::Module&) override {};
  virtual void popRootGuard(llvm::Module&) override {};

private:
 
  NoCommunicator() = default;
};

////////////////////////////////////////////////////////////////////////////////
/// Static builder class
////////////////////////////////////////////////////////////////////////////////
class CommunicatorBuilder
{
  static Communicator * comm;

public:
  static Communicator& getInstance()
  {
    static Communicator& instance = comm ?
      comm->createCommunicator() : NoCommunicator::getInstance();
    return instance;
  }

  static void set(Communicator& mycomm)
  { comm = &mycomm; }
  
};

////////////////////////////////////////////////////////////////////////////////
/// Select the serial communicator
////////////////////////////////////////////////////////////////////////////////
template<typename T>
void commSelect()
{
  auto & Comm = T::getInstance();
  CommunicatorBuilder::set(Comm);
}

/// Initialize the comminicator
void commInit(int * argc, char ** argv[]);

/// Shut down the communicator
void commFinalize();

/// Get the communicator
Communicator & commGetInstance();


} // namespace


#endif // CONTRA_UTILS_COMMUNICATOR_INTERFACE_HPP
