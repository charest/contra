#ifndef CONTRA_UTILS_COMMUNICATOR_MPI_HPP
#define CONTRA_UTILS_COMMUNICATOR_MPI_HPP

#include "communicator.hpp"

#include <mpi.h>

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
  {
    auto err = MPI_Init(argc, argv);
    checkError(err);

    err = MPI_Comm_rank(MPI_COMM_WORLD, &Rank_);
    checkError(err);

    err = MPI_Comm_rank(MPI_COMM_WORLD, &Size_);
    checkError(err);
  }

  void finalize() override {
    auto err = MPI_Finalize();
    checkError(err);
  }

private:
 
  MPICommunicator() = default;

  bool isRoot() const { return Rank_ == 0; }

  void checkError(int  errcode) {
    if (errcode) {
      if (isRoot()) {
        char * str;
        int len;
        MPI_Error_string(errcode, str, &len);
        std::cerr << "MPI Error failed with error code " << errcode << std::endl;
        std::cerr << str << std::endl;
      }
      MPI_Abort(MPI_COMM_WORLD, errcode);
    }
  }


  int Rank_ = 0;
  int Size_ = 0;

};

}

#endif // CONTRA_UTILS_COMMUNICATOR_MPI_HPP
