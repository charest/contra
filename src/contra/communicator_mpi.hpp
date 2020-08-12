#ifndef CONTRA_UTILS_COMMUNICATOR_MPI_HPP
#define CONTRA_UTILS_COMMUNICATOR_MPI_HPP

#include "communicator.hpp"
#include "mpi_rt.hpp"

#include <forward_list>
#include <iostream>

namespace contra {

////////////////////////////////////////////////////////////////////////////////
/// The mpi communicator
////////////////////////////////////////////////////////////////////////////////
class MPICommunicator : public Communicator
{
public:

  static MPICommunicator & getInstance() {
    static MPICommunicator instance;
    return instance;
  }
  
  MPICommunicator(MPICommunicator const&) = delete;
  void operator=(MPICommunicator const&)   = delete;

  Communicator& createCommunicator() override {
    return MPICommunicator::getInstance();
  }

  void init(int * argc, char ** argv[]) override
  { contra_mpi_init(argc, argv); }

  void finalize() override 
  { contra_mpi_finalize(); }

  void markTask(llvm::Module & M) override;
  void unmarkTask(llvm::Module & M) override;
  void pushRootGuard(llvm::Module&) override;
  void popRootGuard(llvm::Module&) override;

private:
 
  MPICommunicator() = default;

  struct RootGuard {
    llvm::BasicBlock * MergeBlock = nullptr;
  };

  std::forward_list<RootGuard> RootGuards_;

};

}

#endif // CONTRA_UTILS_COMMUNICATOR_MPI_HPP
