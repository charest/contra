#include "legion_rt.hpp"

extern "C" {
  
//==============================================================================
// Get/create memory for an index partition
//==============================================================================
std::pair<legion_index_partition_t*, bool>
  contra_legion_partitions_t::getOrCreateIndexPartition(
    legion_runtime_t *rt,
    legion_context_t *ctx,
    legion_index_space_id_t id,
    bool owner)
{
  // found
  for (const auto & Scope : IndexPartitions) {
    auto it = Scope.find(id);
    if (it!=Scope.end()) {
      return std::make_pair(it->second.get(), true);
    }
  }
  // not found
  IndexPartitionDeleter Deleter(rt, ctx, owner);
  auto res = IndexPartitions.front().emplace(
      id,
      IndexPartitionVal(new legion_index_partition_t, Deleter) );
  return std::make_pair(res.first->second.get(), false);
}
  
//==============================================================================
// Get/create memory for a logical partition
//==============================================================================
std::pair<legion_logical_partition_t*, bool>
  contra_legion_partitions_t::getOrCreateLogicalPartition(
    legion_runtime_t *rt,
    legion_context_t *ctx,
    legion_field_id_t fid,
    legion_index_partition_id_t pid)
{
  // found
  auto it = LogicalPartitions.find(std::make_pair(fid,pid));
  if (it!=LogicalPartitions.end()) {
    return std::make_pair(it->second.get(), true);
  }
  // not found
  LogicalPartitionDeleter Deleter(rt, ctx);
  auto res = LogicalPartitions.emplace(
      std::make_pair(fid, pid),
      LogicalPartitionVal(new legion_logical_partition_t, Deleter) );
  return std::make_pair(res.first->second.get(), false);
}
  
  
//==============================================================================
// Get a logical partition that was already created
//==============================================================================
legion_logical_partition_t*
  contra_legion_partitions_t::getLogicalPartition(
    legion_field_id_t fid,
    legion_index_partition_id_t pid)
{ 
  auto it = LogicalPartitions.find(std::make_pair(fid,pid));
  if (it!=LogicalPartitions.end()) return it->second.get();
  return nullptr;
}


//==============================================================================
// Additional startup operations
//==============================================================================
void contra_legion_startup() {}

//==============================================================================
/// index space creation
//==============================================================================
int_t contra_legion_sum_array(dopevector_t * arr)
{
  int_t sum = 0;
  auto ptr = static_cast<const int_t*>(arr->data);
  for (int_t i=0; i<arr->size; ++i) sum += ptr[i];
  return sum;
}

//==============================================================================
/// index space creation
//==============================================================================
void contra_legion_index_space_partition(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    contra_legion_index_space_t * cs,
    contra_legion_index_space_t * is,
    legion_index_partition_t * part)
{
  *part = legion_index_partition_create_equal(
      *runtime,
      *ctx,
      is->index_space,
      cs->index_space,
      /* granularity */ 1,
      /*color*/ AUTO_GENERATE_ID );
}

//==============================================================================
/// index space creation
//==============================================================================
void contra_legion_index_space_partition_from_size(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    int_t size,
    contra_legion_index_space_t * is,
    legion_index_partition_t * part)
{
  legion_index_space_t color_space = legion_index_space_create(*runtime, *ctx, size);

  *part = legion_index_partition_create_equal(
      *runtime,
      *ctx,
      is->index_space,
      color_space,
      /* granularity */ 1,
      /*color*/ AUTO_GENERATE_ID );

  legion_index_space_destroy(*runtime, *ctx, color_space);
}

//==============================================================================
/// index space creation
//==============================================================================
void contra_legion_index_space_partition_from_array(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    dopevector_t *arr,
    contra_legion_index_space_t * is,
    legion_index_partition_t * part,
    bool do_report)
{
  auto ptr = static_cast<const int_t*>(arr->data);
  int_t expanded_size = 0;
  int_t color_size = arr->size;
  for (int_t i=0; i<color_size; ++i) expanded_size += ptr[i];
    
  // create coloring
  legion_coloring_t coloring = legion_coloring_create();

  int_t offset{0};
  for (int_t i=0; i<color_size; ++i) {
    legion_ptr_t lo{ offset };
    legion_ptr_t hi{ offset + ptr[i] - 1 };
    legion_coloring_add_range(coloring, i, lo, hi);
    offset += ptr[i];
  }

  //------------------------------------
  // if the sizes are differeint 
  auto index_size = is->end - is->start;
  if (expanded_size != index_size) {

    if (do_report) {
      legion_runtime_print_once(*runtime, *ctx, stdout,
          "Index spaces partitioned by arrays, without a 'where' clause, MUST "
          "match the size of the original index space\n");
      abort();
    }
    
    legion_index_space_t expanded_space =
      legion_index_space_create(*runtime, *ctx, expanded_size);
    
    *part = legion_index_partition_create_coloring(
        *runtime,
        *ctx,
        expanded_space,
        coloring,
        true,
        /*part color*/ AUTO_GENERATE_ID );


    // clean up
    legion_index_space_destroy(*runtime, *ctx, expanded_space);

  }
  //------------------------------------
  // Naive partitioning
  else {
  
    *part = legion_index_partition_create_coloring(
        *runtime,
        *ctx,
        is->index_space,
        coloring,
        true,
        /*part color*/ AUTO_GENERATE_ID );

  }
  //------------------------------------
    
  // destroy coloring
  legion_coloring_destroy(coloring);
}

//==============================================================================
/// index space creation
//==============================================================================
void contra_legion_index_space_partition_from_field(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    contra_legion_field_t *field,
    contra_legion_index_space_t * is,
    legion_index_partition_t * index_part,
    contra_legion_partitions_t ** parts,
    legion_index_partition_t * part)
{
  auto logical_part = (*parts)->getLogicalPartition(field->field_id, index_part->id);

  legion_index_space_t color_space = legion_index_partition_get_color_space(
      *runtime,
      *index_part);

  // partition with results
  *part = legion_index_partition_create_by_image(
      *runtime,
      *ctx,
      is->index_space,
      *logical_part,
      field->logical_region,
      field->field_id,
      color_space,
      /* part_kind */ COMPUTE_KIND,
      /* color */ AUTO_GENERATE_ID,
      /* mapper_id */ 0,
      /* mapping_tag_id */ 0);
}


//==============================================================================
/// index space partitioning
//==============================================================================
void contra_legion_index_space_create(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    const char *name,
    int_t start,
    int_t end,
    contra_legion_index_space_t * is)
{
  is->start = start;
  is->end = end+1;
  is->step = 1;

  int_t size = end - start + 1;

  is->index_space = legion_index_space_create(*runtime, *ctx, size);
  legion_index_space_attach_name(*runtime, is->index_space, name, false);  
}

//==============================================================================
/// index space partitioning
//==============================================================================
void contra_legion_index_space_create_from_index_partition(
    legion_runtime_t * runtime,
    legion_context_t *,
    legion_task_t * task,
    legion_index_partition_t * part,
    contra_legion_index_space_t * is)
{

  legion_domain_point_t color = legion_task_get_index_point(*task);

  is->index_space = 
    legion_index_partition_get_index_subspace_domain_point(
        *runtime,
        *part,
        color);
  
  legion_domain_t domain =
    legion_index_space_get_domain(*runtime, is->index_space);
  
  legion_rect_1d_t rect = legion_domain_get_rect_1d(domain);

  is->start = rect.lo.x[0];
  is->end = rect.hi.x[0] + 1;
  is->step = 1;
}

//==============================================================================
/// register an index partition with an index sapce
//==============================================================================
void contra_legion_register_index_partition(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    contra_legion_index_space_t * is,
    legion_index_partition_t * index_part,
    contra_legion_partitions_t ** parts)
{
  auto res = (*parts)->getOrCreateIndexPartition(runtime, ctx, is->index_space.id, false);
  if (!res.second) *res.first = *index_part;
}

//==============================================================================
/// index spce destruction
//==============================================================================
void contra_legion_index_space_destroy(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    contra_legion_index_space_t * is)
{
  legion_index_space_destroy(*runtime, *ctx, is->index_space);
}

//==============================================================================
/// index domain creation
//==============================================================================
void contra_legion_domain_create(
    legion_runtime_t * runtime,
    contra_legion_index_space_t * is,
    legion_domain_t * domain)
{
  *domain = legion_domain_from_index_space(*runtime, is->index_space);
}


//==============================================================================
/// field creation
//==============================================================================
void contra_legion_field_create(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    const char *name,
    size_t data_size,
    void* init,
    contra_legion_index_space_t * is,
    contra_legion_field_t * fld)
{
  fld->index_space = is->index_space;

  fld->field_space = legion_field_space_create(*runtime, *ctx);
  legion_field_space_attach_name(*runtime, fld->field_space, name, false);  

  fld->field_allocator = legion_field_allocator_create(*runtime, *ctx, fld->field_space);
  fld->field_id = legion_field_allocator_allocate_field(
    fld->field_allocator, data_size, AUTO_GENERATE_ID );
  legion_field_id_attach_name(*runtime, fld->field_space, fld->field_id, name, false);  

  fld->logical_region =
    legion_logical_region_create(*runtime, *ctx, is->index_space,
        fld->field_space, false);
  
  if (init && data_size>0) {
    legion_runtime_fill_field(*runtime, *ctx, fld->logical_region,
      fld->logical_region, fld->field_id, init, data_size, legion_predicate_true());
  }
}

//==============================================================================
/// field destruction
//==============================================================================
void contra_legion_field_destroy(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    contra_legion_field_t * field)
{
  legion_logical_region_destroy(*runtime, *ctx, field->logical_region);
  //legion_field_allocator_free_field(field->field_allocator, field->field_id);
  legion_field_allocator_destroy(field->field_allocator);
  legion_field_space_destroy(*runtime, *ctx, field->field_space);
}

//==============================================================================
/// field data
//==============================================================================
void contra_legion_pack_field_data(
    contra_legion_field_t * f,
    unsigned regidx,
    void * data)
{
  auto fid = static_cast<uint64_t>(f->field_id);
  auto rid = static_cast<uint32_t>(regidx);
  *static_cast<uint64_t*>(data) = ((fid<<32) | (rid));
}

void contra_legion_unpack_field_data(
    const void *data,
    uint32_t*fid,
    uint32_t*rid)
{
  uint64_t data_int = *static_cast<const uint64_t*>(data);
  uint64_t mask = std::numeric_limits<uint32_t>::max();
  *rid = static_cast<uint32_t>(data_int & mask);
  *fid = static_cast<uint32_t>(data_int >> 32);
}

//==============================================================================
/// field addition
//==============================================================================
void contra_legion_index_add_region_requirement(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    legion_index_launcher_t * launcher,
    contra_legion_index_space_t * cs,
    contra_legion_partitions_t ** parts,
    legion_index_partition_t * specified_part,
    contra_legion_field_t * field)
{
  legion_index_partition_t * index_part = nullptr;

  if ( specified_part ) {
    index_part = specified_part;
  }
  else {
  
    // index partition 
    auto res = (*parts)->getOrCreateIndexPartition(runtime, ctx, field->index_space.id);
    index_part = res.first;

    if (!res.second) {
      *index_part = legion_index_partition_create_equal(
          *runtime,
          *ctx,
          field->index_space,
          cs->index_space,
          /* granularity */ 1,
          /*color*/ AUTO_GENERATE_ID );
    }

  } // specified_part

  // region partition
  auto res = (*parts)->getOrCreateLogicalPartition(
      runtime,
      ctx,
      field->field_id,
      index_part->id);
  auto logical_part = res.first;

  if (!res.second) {
    *logical_part = legion_logical_partition_create(
        *runtime,
        *ctx,
        field->logical_region,
        *index_part);
  }

  legion_privilege_mode_t priviledge = READ_WRITE;
  if (!legion_index_partition_is_disjoint(*runtime, *index_part))
    priviledge = READ_ONLY;

  unsigned idx = legion_index_launcher_add_region_requirement_logical_partition(
    *launcher, *logical_part,
    /* legion_projection_id_t */ 0,
    priviledge, EXCLUSIVE,
    field->logical_region,
    /* legion_mapping_tag_id_t */ 0,
    /* bool verified */ false);

  legion_index_launcher_add_field(*launcher, idx, field->field_id, /* bool inst */ true);
}

//==============================================================================
/// field addition
//==============================================================================
void contra_legion_task_add_region_requirement(
    legion_task_launcher_t * launcher,
    contra_legion_field_t * field )
{
  unsigned idx = legion_task_launcher_add_region_requirement_logical_region(
    *launcher, field->logical_region,
    READ_WRITE, EXCLUSIVE,
    field->logical_region,
    /* legion_mapping_tag_id_t */ 0,
    /* bool verified */ false);

  legion_task_launcher_add_field(*launcher, idx, field->field_id, /* bool inst */ true);
}

//==============================================================================
/// partition create
//==============================================================================
void contra_legion_partitions_create(
    contra_legion_partitions_t ** parts)
{
  if (!*parts) *parts = new contra_legion_partitions_t;
}


//==============================================================================
/// partition push
//==============================================================================
void contra_legion_partitions_push(
    contra_legion_partitions_t ** parts)
{
  (*parts)->push();
}

//==============================================================================
/// partition pop
//==============================================================================
void contra_legion_partitions_pop(
    contra_legion_partitions_t ** parts)
{
  (*parts)->pop();
}

//==============================================================================
/// partition destruction
//==============================================================================
void contra_legion_partitions_destroy(
    legion_runtime_t *,
    legion_context_t *,
    contra_legion_partitions_t ** parts)
{
  //for (auto & part : (*parts)->LogicalPartitions) part.second.reset();
  delete *parts;
  *parts = nullptr;
}

//==============================================================================
/// get field accessor
//==============================================================================
void contra_legion_get_accessor(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    legion_physical_region_t **regionptr,
    uint32_t* /*num_regions*/,
    uint32_t* region_id,
    uint32_t* field_id,
    contra_legion_accessor_t* acc)
{
  acc->field_id = *field_id;

  acc->physical_region = (*regionptr)[*region_id];

  acc->accessor =
    legion_physical_region_get_field_accessor_array_1d( 
      acc->physical_region, *field_id);
  
  acc->logical_region = 
    legion_physical_region_get_logical_region(acc->physical_region);

  acc->domain = 
    legion_index_space_get_domain(*runtime, acc->logical_region.index_space);
  
  acc->rect = legion_domain_get_bounds_1d(acc->domain);

  acc->data = legion_accessor_array_1d_raw_rect_ptr(
      acc->accessor, acc->rect, &acc->subrect, &acc->offsets);

  acc->data_size = legion_field_id_get_size(
      *runtime,
      *ctx,
      acc->logical_region.field_space,
      acc->field_id);
}


//==============================================================================
/// Accessor write
//==============================================================================
void contra_legion_accessor_write(
    contra_legion_accessor_t * acc,
    const void * data,
    int_t index)
{
  byte_t * offset = static_cast<byte_t*>(acc->data) + acc->data_size*index;
  memcpy(offset, data, acc->data_size);
}

//==============================================================================
/// Accessor read
//==============================================================================
void contra_legion_accessor_read(
    contra_legion_accessor_t * acc,
    void * data,
    int_t index)
{
  const byte_t * offset = static_cast<const byte_t*>(acc->data) + acc->data_size*index;
  memcpy(data, offset, acc->data_size);
}

//==============================================================================
/// accessor destruction
//==============================================================================
void contra_legion_accessor_destroy(
    legion_runtime_t *,
    legion_context_t *,
    contra_legion_accessor_t * acc)
{
  legion_accessor_array_1d_destroy(acc->accessor);
}

//==============================================================================
/// Partition destruction
//==============================================================================
void contra_legion_partition_destroy(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    legion_index_partition_t * part)
{
  legion_index_partition_destroy(*runtime, *ctx, *part);
}

//==============================================================================
/// get the timer
//==============================================================================

real_t get_wall_time(void) {
  return timer();
}

void contra_legion_timer_start(real_t *)
{
  //*time = get_wall_time() * 1e3;
}

void contra_legion_timer_stop(real_t *)
{
  //std::cout << get_wall_time()*1e3 - *time << std::endl;
}

//==============================================================================
/// Create a reduction
//==============================================================================

void contra_legion_create_reduction(
    legion_reduction_op_id_t redop,
    contra::apply_t apply_ptr,
    contra::fold_t fold_ptr,
    contra::init_t init_ptr,
    std::size_t data_size)
{
  auto RedOp = new contra::ReductionOp(data_size, apply_ptr, fold_ptr, init_ptr);
  Legion::Runtime::register_reduction_op(redop, RedOp, NULL, NULL, false);
}


}
