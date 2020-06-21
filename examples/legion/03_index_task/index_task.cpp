#include <cassert>
#include <cstdio>

#include "legion_c.h"

enum TaskID {
  TOP_LEVEL_TASK_ID,
  INDEX_SPACE_TASK_ID
};

void index_space_task(
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

  legion_domain_point_t dp = legion_task_get_index_point(task);

  assert(dp.dim == 1);

  size_t local_arglen = legion_task_get_local_arglen(task);
  assert(local_arglen == sizeof(int));
  int local_arg = *(int *)legion_task_get_local_args(task);
  
	size_t global_arglen = legion_task_get_arglen(task);
  assert(global_arglen == sizeof(int));
  int global_arg = *(int *)legion_task_get_args(task);

  printf("Hello world from task %lld, with local arg %d, and global arg %d!\n",
		dp.point_data[0], local_arg, global_arg);

  int res = 2*local_arg;

  void* retval = (void*)(&res);
  size_t retsize = sizeof(res);
  legion_task_postamble(
    runtime,
    ctx,
    retval,
    retsize);
}



void top_level_task(
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

  int num_points = 10;
  printf("Running hello world redux for %d points...\n", num_points);
  
  legion_point_1d_t lo, hi;
  legion_rect_1d_t launch_bound;
  lo.x[0] = 0;
  hi.x[0] = num_points-1;
  launch_bound.lo = lo;
  launch_bound.hi = hi;
  legion_domain_t domain = legion_domain_from_rect_1d(launch_bound);

  int i = 0;
  legion_argument_map_t arg_map = legion_argument_map_create();
  for (i = 0; i < num_points; i++) {
    legion_task_argument_t local_task_args;
    int input = i + num_points;
    local_task_args.args = &input;
    local_task_args.arglen = sizeof(input);
    legion_point_1d_t tmp_p;
    tmp_p.x[0] = i;
    legion_domain_point_t dp = legion_domain_point_from_point_1d(tmp_p);
    legion_argument_map_set_point(arg_map, dp, local_task_args, true);
  }

  legion_task_argument_t global_task_args;
  global_task_args.args = &i;
  global_task_args.arglen = sizeof(i);

  legion_index_launcher_t index_launcher =
    legion_index_launcher_create(INDEX_SPACE_TASK_ID, domain, global_task_args,
    arg_map, legion_predicate_true(), false, 0, 0);
  legion_future_map_t fm = legion_index_launcher_execute(runtime, ctx, index_launcher);
  legion_future_map_wait_all_results(fm);


  bool all_passed = true;
  for (int i = 0; i < num_points; i++) {
    int expected = 2*(i+10);
    legion_point_1d_t tmp_p;
    tmp_p.x[0] = i;
    legion_domain_point_t dp = legion_domain_point_from_point_1d(tmp_p);
		legion_future_t fut = legion_future_map_get_future(fm, dp);
    int received = *(int*)legion_future_get_untyped_pointer(fut);
    if (expected != received) {
      printf("Check failed for point %d: %d != %d\n", i, expected, received);
      all_passed = false;
    }
  }
  if (all_passed)
    printf("All checks passed!\n");

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
        INDEX_SPACE_TASK_ID,
        AUTO_GENERATE_ID,
        "fibonacci task",
        "fibonacci variant",
        execution_constraints[1],
        layout_constraints[1],
        options,
        index_space_task,
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
