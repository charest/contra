#include <cassert>
#include <cstdio>
#include <cstring>

#include "legion_c.h"
enum FieldId {
  FIELD_ID
};

enum TaskID {
  TOP_LEVEL_TASK_ID,
  INDEX_SPACE_TASK_ID,
  FIELD_TASK_ID
};

////////////////////////////////////////////////////////////////////////////////
void field_task(
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

  legion_domain_point_t dp = legion_task_get_index_point(task);

  assert(dp.dim == 1);

  size_t local_arglen = legion_task_get_local_arglen(task);
  assert(local_arglen == 0);
  
	size_t global_arglen = legion_task_get_arglen(task);
  assert(global_arglen == 0);

  assert(num_regions == 1);
  legion_accessor_array_1d_t acc = 
    legion_physical_region_get_field_accessor_array_1d( 
      regions[0], FIELD_ID);
  
  legion_logical_region_t logical_region = 
    legion_physical_region_get_logical_region(regions[0]);
  
  legion_domain_t domain = 
    legion_index_space_get_domain(runtime, logical_region.index_space);
  legion_rect_1d_t rect = legion_domain_get_bounds_1d(domain);
  
  legion_rect_1d_t subrect;
  legion_byte_offset_t offsets;
  void * data_ptr = legion_accessor_array_1d_raw_rect_ptr(
      acc, rect, &subrect, &offsets);

  double field_data;
  memcpy(&field_data, data_ptr, sizeof(double));

  printf("Hello world from task %lld, with field %f!\n",
		dp.point_data[0], field_data);
  
  void* retval = 0;
  size_t retsize = 0;
  legion_task_postamble(
    runtime,
    ctx,
    retval,
    retsize);
}


////////////////////////////////////////////////////////////////////////////////
void index_space_task(
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

  legion_domain_point_t dp = legion_task_get_index_point(task);

  assert(dp.dim == 1);

  size_t local_arglen = legion_task_get_local_arglen(task);
  assert(local_arglen == sizeof(int));
  int local_arg = *(int *)legion_task_get_local_args(task);
  
	size_t global_arglen = legion_task_get_arglen(task);
  assert(global_arglen == sizeof(int));
  int global_arg = *(int *)legion_task_get_args(task);

  assert(num_regions == 1);
  legion_accessor_array_1d_t acc = 
    legion_physical_region_get_field_accessor_array_1d( 
      regions[0], FIELD_ID);

  legion_logical_region_t logical_region = 
    legion_physical_region_get_logical_region(regions[0]);
  
  legion_domain_t domain = 
    legion_index_space_get_domain(runtime, logical_region.index_space);
  legion_rect_1d_t rect = legion_domain_get_bounds_1d(domain);
  
  legion_rect_1d_t subrect;
  legion_byte_offset_t offsets;
  void * data_ptr = legion_accessor_array_1d_raw_rect_ptr(
      acc, rect, &subrect, &offsets);

  double field_data;
  memcpy(&field_data, data_ptr, sizeof(double));

  printf("Hello world from task %lld, with local arg %d, and global arg %d and field %f!\n",
		dp.point_data[0], local_arg, global_arg, field_data);
  
  field_data = dp.point_data[0];
  memcpy(data_ptr, &field_data, sizeof(double));

  legion_accessor_array_1d_destroy(acc);

  int res = 2*local_arg;

  void* retval = (void*)(&res);
  size_t retsize = sizeof(res);
  legion_task_postamble(
    runtime,
    ctx,
    retval,
    retsize);
}



////////////////////////////////////////////////////////////////////////////////
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

  int num_points = 10;
  printf("Running hello world redux for %d points...\n", num_points);

  legion_index_space_t index_space = 
    legion_index_space_create(runtime, ctx, num_points);

  legion_field_space_t field_space = legion_field_space_create(runtime, ctx);

  legion_field_allocator_t field_allocator =
    legion_field_allocator_create(runtime, ctx, field_space);
  legion_field_id_t field_id = legion_field_allocator_allocate_field(
    field_allocator, sizeof(double), FIELD_ID );
  legion_field_allocator_destroy(field_allocator);

  legion_logical_region_t logical_region =
    legion_logical_region_create(runtime, ctx, index_space, field_space, false);

  legion_index_space_t color_space = 
    legion_index_space_create(runtime, ctx, num_points);

  legion_index_partition_t part =
    legion_index_partition_create_equal(runtime, ctx, index_space, color_space,
      /* granularity */ 1, /*color*/ AUTO_GENERATE_ID );

  legion_logical_partition_t logical_part = 
    legion_logical_partition_create(runtime, ctx, logical_region, part);
  legion_domain_t domain = legion_domain_from_index_space(runtime, color_space);

  double zero = 0.;
  size_t data_size = sizeof(double);
  legion_runtime_fill_field(runtime, ctx, logical_region,
    logical_region, field_id, &zero, data_size, legion_predicate_true());

  //============================================================================
  {
    int i = 0;
    legion_argument_map_t arg_map = legion_argument_map_create();
    for (i = 0; i < num_points; i++) {
      legion_task_argument_t local_task_args;
      int input = i + 10;
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

    unsigned idx = legion_index_launcher_add_region_requirement_logical_partition(
      index_launcher, logical_part,
      /* legion_projection_id_t */ 0,
      READ_WRITE, EXCLUSIVE,
      logical_region,
      /* legion_mapping_tag_id_t */ 0,
      /* bool verified */ false);

    legion_index_launcher_add_field(index_launcher, idx, field_id, /* bool inst */ true);

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
      legion_future_destroy(fut);
    }
    if (all_passed)
      printf("All checks passed!\n");

    legion_future_map_destroy(fm);
    legion_index_launcher_destroy(index_launcher);
    legion_argument_map_destroy(arg_map);
  }
  //============================================================================
  {
    legion_task_argument_t global_task_args;
    global_task_args.args = nullptr;
    global_task_args.arglen = 0;
    
    legion_argument_map_t arg_map = legion_argument_map_create();

    legion_index_launcher_t index_launcher =
      legion_index_launcher_create(FIELD_TASK_ID, domain, global_task_args,
      arg_map, legion_predicate_true(), false, 0, 0);

    unsigned idx = legion_index_launcher_add_region_requirement_logical_partition(
      index_launcher, logical_part,
      /* legion_projection_id_t */ 0,
      READ_WRITE, EXCLUSIVE,
      logical_region,
      /* legion_mapping_tag_id_t */ 0,
      /* bool verified */ false);

    legion_index_launcher_add_field(index_launcher, idx, field_id, /* bool inst */ true);
    
    legion_future_map_t fm = legion_index_launcher_execute(runtime, ctx, index_launcher);
    legion_future_map_wait_all_results(fm);
    
    legion_future_map_destroy(fm);
    legion_index_launcher_destroy(index_launcher);
    legion_argument_map_destroy(arg_map);
  }
  //============================================================================



  legion_logical_partition_destroy(runtime, ctx, logical_part);
  legion_index_partition_destroy(runtime, ctx, part);
  legion_logical_region_destroy(runtime, ctx, logical_region);
  legion_field_space_destroy(runtime, ctx, field_space);
  legion_index_space_destroy(runtime, ctx, index_space);

  void* retval = nullptr;
  size_t retsize = 0;
  legion_task_postamble(
    runtime,
    ctx,
    retval,
    retsize);
}
 
////////////////////////////////////////////////////////////////////////////////
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
      /*.leaf=*/ true,
      /*.inner=*/ false,
      /*.idempotent=*/ false,
      /*.replicable=*/ false};

    //legion_task_id_t variant_id =
      legion_runtime_preregister_task_variant_fnptr(
        INDEX_SPACE_TASK_ID,
        AUTO_GENERATE_ID,
        "index_space task",
        "index_space variant",
        execution_constraints[1],
        layout_constraints[1],
        options,
        index_space_task,
        nullptr,
        0);
  }
  
  {
    execution_constraints[2] = legion_execution_constraint_set_create();
    layout_constraints[2] = legion_task_layout_constraint_set_create();

    legion_execution_constraint_set_add_processor_constraint(
      execution_constraints[2], LOC_PROC);

    legion_task_config_options_t options{
      /*.leaf=*/ true,
      /*.inner=*/ false,
      /*.idempotent=*/ false,
      /*.replicable=*/ false};

    //legion_task_id_t variant_id =
      legion_runtime_preregister_task_variant_fnptr(
        FIELD_TASK_ID,
        AUTO_GENERATE_ID,
        "field task",
        "field variant",
        execution_constraints[2],
        layout_constraints[2],
        options,
        field_task,
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
