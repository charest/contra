#include "mpi_rt.hpp"

#include <iostream>

using namespace contra;

mpi_runtime_t MpiRuntime;

//==============================================================================
// Check errors
//==============================================================================
void mpi_runtime_t::check(int  errcode) {
  if (errcode) {
    if (isRoot()) {
      char * str = nullptr;
      int len = 0;
      MPI_Error_string(errcode, str, &len);
      std::cerr << "MPI Error failed with error code " << errcode << std::endl;
      std::cerr << str << std::endl;
    }
    MPI_Abort(MPI_COMM_WORLD, errcode);
  }
}
  

extern "C" {
  
//==============================================================================
/// startup runtime
//==============================================================================
void contra_mpi_init(int * argc, char *** argv)
{ 
  auto err = MPI_Init(argc, argv);
  MpiRuntime.check(err);

  err = MPI_Comm_rank(MPI_COMM_WORLD, &MpiRuntime.rank);
  MpiRuntime.check(err);

  err = MPI_Comm_rank(MPI_COMM_WORLD, &MpiRuntime.size);
  MpiRuntime.check(err);
}

//==============================================================================
// shutdown runtime
//==============================================================================
void contra_mpi_finalize()
{
  auto err = MPI_Finalize();
  MpiRuntime.check(err);
}

//==============================================================================
/// mark we are in a task
//==============================================================================
void contra_mpi_mark_task()
{ MpiRuntime.TaskCounter++; }

//==============================================================================
/// unmark we are in a task
//==============================================================================
void contra_mpi_unmark_task()
{ MpiRuntime.TaskCounter--; }

//==============================================================================
/// Test if we need to guard and if we are root
//==============================================================================
bool contra_mpi_test_root()
{ return MpiRuntime.TaskCounter > 0 && MpiRuntime.isRoot(); }


} // extern
