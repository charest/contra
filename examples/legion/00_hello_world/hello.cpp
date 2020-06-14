#include <cstdio>

#include "legion_c.h"

enum TaskID {
  HELLO_WORLD_ID,
};

void hello_world_task(
    const void *data,
    size_t datalen,
    const void * /*userdata*/,
    size_t /*userlen*/,
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

  printf("Hello World!\n");

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
  legion_runtime_set_top_level_task_id(HELLO_WORLD_ID);
    
  auto execution_constraints = legion_execution_constraint_set_create();
  auto layout_constraints = legion_task_layout_constraint_set_create();

  legion_execution_constraint_set_add_processor_constraint(
    execution_constraints, LOC_PROC);

  legion_task_config_options_t options{
    /*.leaf=*/ false,
    /*.inner=*/ false,
    /*.idempotent=*/ false,
    /*.replicable=*/ false};

  /*auto variant_id =*/ legion_runtime_preregister_task_variant_fnptr(
    HELLO_WORLD_ID,
    AUTO_GENERATE_ID,
    "hello_world task",
    "hello_world variant",
    execution_constraints,
    layout_constraints,
    options,
    hello_world_task,
    nullptr,
    0);
  
  auto res = legion_runtime_start(argc, argv, false);

  legion_execution_constraint_set_destroy(execution_constraints);
  legion_task_layout_constraint_set_destroy(layout_constraints);

  return res;
} 
