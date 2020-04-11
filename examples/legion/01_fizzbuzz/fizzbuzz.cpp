#include <cassert>
#include <cstdio>

#include "legion_c.h"

enum TaskID {
  TOP_LEVEL_TASK_ID,
  FIZZBUZZ_TASK_ID
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

  for (int i = 1; i <= 100; i++) {
    legion_task_argument_t task_args;
    task_args.args = &i;
    task_args.arglen = sizeof(i);
    legion_task_launcher_t launcher =
      legion_task_launcher_create(FIZZBUZZ_TASK_ID, task_args, legion_predicate_true(), 0, 0);
    legion_task_launcher_execute(runtime, ctx, launcher);
  }
  

  void* retval = nullptr;
  size_t retsize = 0;
  legion_task_postamble(
    runtime,
    ctx,
    retval,
    retsize);
}
  
void fizzbuzz_task(
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

  int number = *(int*)legion_task_get_args(task); 
  if (number % 15 == 0)
    printf("fizzbuzz %i\n", number);
  else if (number % 5 == 0)
    printf("buzz %i\n", number);
  else if (number % 3 == 0)
    printf("fizz %i\n", number);
  else
    printf("%i\n", number);
  
  void* retval = nullptr;
  size_t retsize = 0;
  legion_task_postamble(
    runtime,
    ctx,
    retval,
    retsize);
}


int main(int argc, char **argv)
{
  legion_runtime_set_top_level_task_id(TOP_LEVEL_TASK_ID);

  legion_execution_constraint_set_t execution_constraints[2];
  legion_task_layout_constraint_set_t layout_constraints[2];
  
  {
    execution_constraints[0] = legion_execution_constraint_set_create();
    layout_constraints[0] = legion_task_layout_constraint_set_create();

    legion_execution_constraint_set_add_processor_constraint(
      execution_constraints[0], LOC_PROC);

    legion_task_config_options_t options{
      /*.leaf=*/ false,
      /*.inner=*/ false,
      /*.idempotent=*/ false,
      /*.replicable=*/ false};

    //legion_task_id_t variant_id =
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
      /*.leaf=*/ false,
      /*.inner=*/ false,
      /*.idempotent=*/ false,
      /*.replicable=*/ false};

    //legion_task_id_t variant_id =
      legion_runtime_preregister_task_variant_fnptr(
        FIZZBUZZ_TASK_ID,
        AUTO_GENERATE_ID,
        "fibonacci task",
        "fibonacci variant",
        execution_constraints[1],
        layout_constraints[1],
        options,
        fizzbuzz_task,
        nullptr,
        0);
  }
  
  int res = legion_runtime_start(argc, argv, false);

  for (int i=0; i<2; i++) {
    legion_execution_constraint_set_destroy(execution_constraints[i]);
    legion_task_layout_constraint_set_destroy(layout_constraints[i]);
  }

  return res;
} 
