#include <cassert>
#include <cstdio>

#include "legion_c.h"

enum TaskID {
  TOP_LEVEL_TASK_ID,
  FIBONACCI_TASK_ID,
  SUM_TASK_ID
};

void top_level_task(
    const void *data,
    size_t datalen,
    const void * userdata,
    size_t userlen,
    realm_id_t proc_id)
{
  legion_task_t task;
  const legion_physical_region_t *regions = nullptr;
  uint32_t num_regions = 0;
  legion_context_t ctx;
  legion_runtime_t runtime;

  legion_task_preamble(
    data,
    datalen,
    proc_id,
    &task,
    &regions,
    &num_regions,
    &ctx,
    &runtime);

  int num_fibonacci = 7; // Default value
  printf("Computing the first %d Fibonacci numbers...\n", num_fibonacci);

  legion_future_t* fib_results = new legion_future_t[num_fibonacci];
  
  for (int i = 0; i < num_fibonacci; i++) {
    legion_task_argument_t task_args;
    task_args.args = &i;
    task_args.arglen = sizeof(i);
    legion_task_launcher_t launcher =
      legion_task_launcher_create(FIBONACCI_TASK_ID, task_args, legion_predicate_true(), 0, 0);
    fib_results[i] = legion_task_launcher_execute(runtime, ctx, launcher);
  }
  
  for (int i = 0; i < num_fibonacci; i++) {
    assert( legion_future_get_untyped_size(fib_results[i]) == sizeof(int) );
    int result = *(int*)legion_future_get_untyped_pointer(fib_results[i]);
    printf("Fibonacci(%d) = %d\n", i, result);
  }

  delete[] fib_results;

  void* retval = nullptr;
  size_t retsize = 0;
  legion_task_postamble(
    runtime,
    ctx,
    retval,
    retsize);
}
  
void fibonacci_task(
    const void *data,
    size_t datalen,
    const void * userdata,
    size_t userlen,
    realm_id_t proc_id)
{
  legion_task_t task;
  const legion_physical_region_t *regions = nullptr;
  uint32_t num_regions = 0;
  legion_context_t ctx;
  legion_runtime_t runtime;

  legion_task_preamble(
    data,
    datalen,
    proc_id,
    &task,
    &regions,
    &num_regions,
    &ctx,
    &runtime);

  size_t arglen = legion_task_get_arglen(task);
  assert(arglen == sizeof(int));

  int res;

  int fib_num = *(int*)legion_task_get_args(task); 
  if (fib_num == 0) {
    res = 0;
  }
  else if (fib_num == 1) {
    res = 1;
  }
  else {

    // Launch fib-1
    int fib1 = fib_num-1;
    legion_task_argument_t task_args_1;
    task_args_1.args = &fib1;
    task_args_1.arglen = sizeof(fib1);
    legion_task_launcher_t t1 =
      legion_task_launcher_create(FIBONACCI_TASK_ID, task_args_1, legion_predicate_true(), 0, 0);
    legion_future_t f1 = legion_task_launcher_execute(runtime, ctx, t1);

    // Launch fib-2
    int fib2 = fib_num-2;
    legion_task_argument_t task_args_2;
    task_args_2.args = &fib2;
    task_args_2.arglen = sizeof(fib2);
    legion_task_launcher_t t2 =
      legion_task_launcher_create(FIBONACCI_TASK_ID, task_args_2, legion_predicate_true(), 0, 0);
    legion_future_t f2 = legion_task_launcher_execute(runtime, ctx, t2);

    legion_task_argument_t sum_args;
    sum_args.args = nullptr;
    sum_args.arglen = 0;
    legion_task_launcher_t tsum =
      legion_task_launcher_create(SUM_TASK_ID, sum_args, legion_predicate_true(), 0, 0);

    legion_task_launcher_add_future(tsum, f1);
    legion_task_launcher_add_future(tsum, f2);
    legion_future_t fsum = legion_task_launcher_execute(runtime, ctx, tsum);

    assert( legion_future_get_untyped_size(fsum) == sizeof(int) );
    res = *(int*)legion_future_get_untyped_pointer(fsum);

  }

  void* retval = (void*)(&res);
  size_t retsize = sizeof(res);
  legion_task_postamble(
    runtime,
    ctx,
    retval,
    retsize);
}

void sum_task(
    const void *data,
    size_t datalen,
    const void * userdata,
    size_t userlen,
    realm_id_t proc_id)
{
  legion_task_t task;
  const legion_physical_region_t *regions = nullptr;
  uint32_t num_regions = 0;
  legion_context_t ctx;
  legion_runtime_t runtime;

  legion_task_preamble(
    data,
    datalen,
    proc_id,
    &task,
    &regions,
    &num_regions,
    &ctx,
    &runtime);

  unsigned nfutures = legion_task_get_futures_size(task);
  assert(nfutures == 2);

  legion_future_t f1 = legion_task_get_future(task, 0);
  legion_future_t f2 = legion_task_get_future(task, 1);
    
  assert( legion_future_get_untyped_size(f1) == sizeof(int) );
  assert( legion_future_get_untyped_size(f2) == sizeof(int) );

  int r1 = *(int*)legion_future_get_untyped_pointer(f1);
  int r2 = *(int*)legion_future_get_untyped_pointer(f2);

  int sum = r1 + r2;

  void* retval = (void*)(&sum);
  size_t retsize = sizeof(sum);
  legion_task_postamble(
    runtime,
    ctx,
    retval,
    retsize);
}
  

int main(int argc, char **argv)
{
  legion_runtime_set_top_level_task_id(TOP_LEVEL_TASK_ID);

  legion_execution_constraint_set_t execution_constraints[3];
  legion_task_layout_constraint_set_t layout_constraints[3];
  
  {
    execution_constraints[0] = legion_execution_constraint_set_create();
    layout_constraints[0] = legion_task_layout_constraint_set_create();

    legion_execution_constraint_set_add_processor_constraint(
      execution_constraints[0], LOC_PROC);

    legion_task_config_options_t options{
      .leaf=false,
      .inner=false,
      .idempotent=false,
      .replicable=false};

    legion_task_id_t variant_id =
      legion_runtime_preregister_task_variant_fnptr(
        TOP_LEVEL_TASK_ID,
        AUTO_GENERATE_ID,
        "top_level task",
        "top_level variant",
        execution_constraints[0],
        layout_constraints[0],
        options,
        top_level_task,
        nullptr,
        0);
  }
  
  {
    execution_constraints[1] = legion_execution_constraint_set_create();
    layout_constraints[1] = legion_task_layout_constraint_set_create();

    legion_execution_constraint_set_add_processor_constraint(
      execution_constraints[1], LOC_PROC);

    legion_task_config_options_t options{
      .leaf=false,
      .inner=false,
      .idempotent=false,
      .replicable=false};

    legion_task_id_t variant_id =
      legion_runtime_preregister_task_variant_fnptr(
        FIBONACCI_TASK_ID,
        AUTO_GENERATE_ID,
        "fibonacci task",
        "fibonacci variant",
        execution_constraints[1],
        layout_constraints[1],
        options,
        fibonacci_task,
        nullptr,
        0);
  }
  
  {
    execution_constraints[2] = legion_execution_constraint_set_create();
    layout_constraints[2] = legion_task_layout_constraint_set_create();

    legion_execution_constraint_set_add_processor_constraint(
      execution_constraints[2], LOC_PROC);

    legion_task_config_options_t options{
      .leaf=true,
      .inner=false,
      .idempotent=false,
      .replicable=false};

    legion_task_id_t variant_id =
      legion_runtime_preregister_task_variant_fnptr(
        SUM_TASK_ID,
        AUTO_GENERATE_ID,
        "sum task",
        "sum variant",
        execution_constraints[2],
        layout_constraints[2],
        options,
        sum_task,
        nullptr,
        0);
    }
  
  int res = legion_runtime_start(argc, argv, false);

  for (int i=0; i<3; i++) {
    legion_execution_constraint_set_destroy(execution_constraints[i]);
    legion_task_layout_constraint_set_destroy(layout_constraints[i]);
  }

  return res;
} 
