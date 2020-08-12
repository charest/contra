#include "communicator_mpi.hpp"

using namespace llvm;
using namespace utils;

namespace contra {

//==============================================================================
// Mark a task
//==============================================================================
void MPICommunicator::markTask(Module & M)
{
  TheHelper_->callFunction(
      M,
      "contra_mpi_mark_task",
      VoidType_);
}

//==============================================================================
// Unmark a task
//==============================================================================
void MPICommunicator::unmarkTask(Module & M)
{
  TheHelper_->callFunction(
      M,
      "contra_mpi_unmark_task",
      VoidType_);
}

//==============================================================================
// Create a root guard
//==============================================================================
void MPICommunicator::pushRootGuard(Module & M)
{
  auto TestV = TheHelper_->callFunction(
      M,
      "contra_mpi_test_root",
      Int8Type_);
  auto CondV = TheHelper_->createCast(TestV, Int1Type_);

  // Create blocks for the then and else cases.  Insert the 'then' block at the
  // end of the function.
  auto TheFunction = TheBuilder_->GetInsertBlock()->getParent();
  auto ThenBB = BasicBlock::Create(*TheContext_, "then", TheFunction);
  auto MergeBB = BasicBlock::Create(*TheContext_, "ifcont");
  TheBuilder_->CreateCondBr(CondV, ThenBB, MergeBB);
  TheBuilder_->SetInsertPoint(ThenBB);

  RootGuards_.push_front({MergeBB});

}

//==============================================================================
// Pop the root guard
//==============================================================================
void MPICommunicator::popRootGuard(Module&)
{
  auto MergeBB = RootGuards_.front().MergeBlock;

  auto TheFunction = TheBuilder_->GetInsertBlock()->getParent();
  TheBuilder_->CreateBr(MergeBB);
  TheFunction->getBasicBlockList().push_back(MergeBB);
  TheBuilder_->SetInsertPoint(MergeBB);
  RootGuards_.pop_front();
}

} // namespace
