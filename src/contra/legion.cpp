#include "config.hpp"

#include "codegen.hpp"
#include "errors.hpp"
#include "legion.hpp"
#include "utils/llvm_utils.hpp"
#include "librt/dopevector.hpp"

#include <vector>

////////////////////////////////////////////////////////////////////////////////
// Legion runtime
////////////////////////////////////////////////////////////////////////////////


namespace contra {

}

extern "C" {

struct contra_legion_index_space_t {
  int_t start;
  int_t end;
  legion_index_space_t index_space;
};

struct contra_legion_field_t {
  legion_index_space_t index_space;
  legion_field_space_t field_space;
  legion_field_allocator_t field_allocator;
  legion_field_id_t field_id;
  legion_logical_region_t logical_region;
};

struct contra_legion_accessor_t {
  legion_field_id_t field_id;
  legion_physical_region_t physical_region;
  legion_logical_region_t logical_region;
  legion_domain_t domain; 
  legion_rect_1d_t rect;
  legion_rect_1d_t subrect;
  legion_byte_offset_t offsets;
  legion_accessor_array_1d_t accessor;
  void * data;
};

struct contra_legion_partitions_t {
  std::map< legion_index_space_id_t, std::unique_ptr<legion_index_partition_t> > IndexPartitions;
  
  using partition_pair = std::pair<legion_field_id_t, legion_index_partition_id_t>;
  std::map< partition_pair, std::unique_ptr<legion_logical_partition_t> > LogicalPartitions;

  std::map< legion_index_space_id_t, legion_index_space_id_t > IndexSpaces;

  legion_index_space_id_t getIndexSpace(legion_index_space_id_t id)
  {
    auto parent = id;
    auto it = IndexSpaces.find(id);
    if (it != IndexSpaces.end()) parent = it->second;
    return parent;
  }
};

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
    contra_legion_partitions_t ** parts,
    legion_index_partition_t * part)
{
  if (!*parts) *parts = new contra_legion_partitions_t;

  auto & index_part =
    (*parts)->IndexPartitions[is->index_space.id];
  
  if (!index_part) {
    index_part = std::make_unique<legion_index_partition_t>(
      legion_index_partition_create_equal(
          *runtime,
          *ctx,
          is->index_space,
          cs->index_space,
          /* granularity */ 1,
          /*color*/ AUTO_GENERATE_ID ));
  }

  *part = *index_part;
}

//==============================================================================
/// index space creation
//==============================================================================
void contra_legion_index_space_partition_from_size(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    int_t size,
    contra_legion_index_space_t * is,
    contra_legion_partitions_t ** parts,
    legion_index_partition_t * part)
{
  if (!*parts) *parts = new contra_legion_partitions_t;

  legion_index_space_t color_space = legion_index_space_create(*runtime, *ctx, size);

  auto & index_part = (*parts)->IndexPartitions[is->index_space.id];
  
  index_part = std::make_unique<legion_index_partition_t>(
    legion_index_partition_create_equal(
        *runtime,
        *ctx,
        is->index_space,
        color_space,
        /* granularity */ 1,
        /*color*/ AUTO_GENERATE_ID ));

  *part = *index_part;
}

//==============================================================================
/// index space creation
//==============================================================================
void contra_legion_index_space_partition_from_array(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    dopevector_t *arr,
    contra_legion_index_space_t * is,
    contra_legion_partitions_t ** parts,
    legion_index_partition_t * part,
    bool do_report)
{
  if (!*parts) *parts = new contra_legion_partitions_t;

  auto ptr = static_cast<const int_t*>(arr->data);
  int_t expanded_size = 0;
  int_t color_size = arr->size;
  for (int_t i=0; i<color_size; ++i) expanded_size += ptr[i];

  legion_index_partition_t * index_part = nullptr;

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
    
    // create coloring
    legion_coloring_t expanded_coloring = legion_coloring_create();

    int_t offset{0};
    for (int_t i=0; i<color_size; ++i) {
      legion_ptr_t lo{ offset };
      legion_ptr_t hi{ offset + ptr[i] - 1 };
      legion_coloring_add_range(expanded_coloring, i, lo, hi);
      offset += ptr[i];
    }

    legion_index_space_t expanded_space =
      legion_index_space_create(*runtime, *ctx, expanded_size);
    
    auto res = (*parts)->IndexPartitions.emplace(
        is->index_space.id,
        std::make_unique<legion_index_partition_t>(
          legion_index_partition_create_coloring(
            *runtime,
            *ctx,
            expanded_space,
            expanded_coloring,
            true,
            /*part color*/ AUTO_GENERATE_ID )));
    if (!res.second) {
      legion_runtime_print_once(*runtime, *ctx, stdout,
          "You are clobbering a previous index partition!\n");
      abort();
    }
    index_part = res.first->second.get();

    (*parts)->IndexSpaces[expanded_space.id] = is->index_space.id;

#if 0
  
    // create field for projection
    legion_field_space_t field_space = legion_field_space_create(*runtime, *ctx);
    legion_field_allocator_t field_allocator = legion_field_allocator_create(
        *runtime,
        *ctx,
        field_space);
    legion_field_id_t field_id = legion_field_allocator_allocate_field(
        field_allocator,
        sizeof(legion_point_1d_t),
        AUTO_GENERATE_ID );

    // setup regions
    legion_logical_region_t expanded_region = legion_logical_region_create(
        *runtime,
        *ctx,
        expanded_space,
        field_space,
        false);
      
    legion_logical_partition_t expanded_logical_part = legion_logical_partition_create(
          *runtime,
          *ctx,
          expanded_region,
          expanded_index_part);
  
    // create args
    legion_argument_map_t arg_map = legion_argument_map_create();
    for (int_t i=0; i<color_size; i++) {
      legion_task_argument_t local_task_args;
      int_t input = 0;
      local_task_args.args = &input;
      local_task_args.arglen = sizeof(input);
      legion_point_1d_t tmp_p;
      tmp_p.x[0] = i;
      legion_domain_point_t dp = legion_domain_point_from_point_1d(tmp_p);
      legion_argument_map_set_point(arg_map, dp, local_task_args, true);
    }
  
    legion_task_argument_t global_task_args;
    global_task_args.args = &field_id;
    global_task_args.arglen = sizeof(field_id);
    
    // launch inded task
    legion_index_space_t color_space = legion_index_space_create(*runtime, *ctx, color_size);
    legion_domain_t launch_domain = legion_domain_from_index_space(*runtime, color_space);

    legion_index_launcher_t index_launcher = legion_index_launcher_create(
        INDEX_SPACE_TASK_ID,
        launch_domain,
        global_task_args,
        arg_map,
        legion_predicate_true(),
        false,
        0,
        0);

    unsigned idx = legion_index_launcher_add_region_requirement_logical_partition(
      index_launcher,
      expanded_logical_part,
      /* legion_projection_id_t */ 0,
      WRITE_DISCARD,
      EXCLUSIVE,
      expanded_region,
      /* legion_mapping_tag_id_t */ 0,
      /* bool verified */ false);

    legion_index_launcher_add_field(index_launcher, idx, field_id, /* bool inst */ true);

    legion_future_map_t fm = legion_index_launcher_execute(*runtime, *ctx, index_launcher);
    legion_future_map_wait_all_results(fm);
    
    // partition with results
    index_part = std::make_unique<legion_index_partition_t>(
      legion_index_partition_create_by_image(
        *runtime,
        *ctx,
        is->index_space,
        expanded_logical_part,
        expanded_region,
        field_id,
        color_space,
        /* part_kind */ COMPUTE_KIND,
        /* color */ AUTO_GENERATE_ID,
        /* mapper_id */ 0,
        /* mapping_tag_id */ 0));

#endif
    // clean up
    //legion_index_space_destroy(*runtime, *ctx, expanded_space);
    legion_coloring_destroy(expanded_coloring);
  

  }
  //------------------------------------
  // Naive partitioning
  else {
  
    // create coloring
    legion_coloring_t coloring = legion_coloring_create();

    int_t offset{0};
    for (int_t i=0; i<color_size; ++i) {
      legion_ptr_t lo{ offset };
      legion_ptr_t hi{ offset + ptr[i] - 1 };
      legion_coloring_add_range(coloring, i, lo, hi);
      offset += ptr[i];
    }
  
    auto res = (*parts)->IndexPartitions.emplace(
        is->index_space.id,
        std::make_unique<legion_index_partition_t>(
          legion_index_partition_create_coloring(
            *runtime,
            *ctx,
            is->index_space,
            coloring,
            true,
            /*part color*/ AUTO_GENERATE_ID )));
    if (!res.second) {
      legion_runtime_print_once(*runtime, *ctx, stdout,
          "You are clobbering a previous index partition!\n");
      abort();
    }
    index_part = res.first->second.get();

    // destroy coloring
    legion_coloring_destroy(coloring);
  }
  //------------------------------------

  *part = *index_part;
}

//==============================================================================
/// index space creation
//==============================================================================
void contra_legion_index_space_partition_from_field(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    contra_legion_field_t *field,
    contra_legion_index_space_t * is,
    contra_legion_partitions_t ** parts,
    legion_index_partition_t * part)
{
 
  auto parent = (*parts)->getIndexSpace(field->index_space.id);
  auto & index_part = (*parts)->IndexPartitions[parent];
  auto expanded_index_part = *index_part;

  auto & logical_part =
    (*parts)->LogicalPartitions[{field->field_id, index_part->id}];

   legion_index_space_t color_space = legion_index_partition_get_color_space(
       *runtime,
       expanded_index_part);
    
  // partition with results
  index_part = std::make_unique<legion_index_partition_t>(
    legion_index_partition_create_by_image(
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
      /* mapping_tag_id */ 0));
  
  // old one is no longer any use
  legion_index_partition_destroy(*runtime, *ctx, expanded_index_part);
  legion_index_space_destroy(*runtime, *ctx, field->index_space);

  *part = *index_part;
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

  int_t size = end - start + 1;

  is->index_space = legion_index_space_create(*runtime, *ctx, size);
  legion_index_space_attach_name(*runtime, is->index_space, name, false);  
}

void contra_legion_index_space_create_from_size(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    int_t size,
    contra_legion_index_space_t * is)
{
  is->start = 0;
  is->end = size;
  is->index_space = legion_index_space_create(*runtime, *ctx, size);
}

void contra_legion_index_space_create_from_array(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    dopevector_t *arr,
    contra_legion_index_space_t * is)
{
  std::cout << arr->size << std::endl;
  //is->start = 0;
  //is->end = size-1;
  //is->index_space = legion_index_space_create(*runtime, *ctx, size);
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
/// field creation
//==============================================================================
void contra_legion_field_create_from_partition(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    const char *name,
    size_t data_size,
    void* init,
    legion_index_partition_t * index_part,
    contra_legion_field_t * fld)
{
  contra_legion_index_space_t is;

  is.index_space = 
    legion_index_partition_get_parent_index_space(
        *runtime,
        *index_part);
  legion_domain_t domain = legion_index_space_get_domain(
      *runtime,
      is.index_space);
  legion_rect_1d_t rect = legion_domain_get_rect_1d(domain);

  is.start = rect.lo.x[0];
  is.end = rect.hi.x[0] + 1;

  contra_legion_field_create(runtime, ctx, name, data_size, init, &is, fld);
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
    void ** void_parts,
    contra_legion_field_t * field)
{

  auto parts = reinterpret_cast<contra_legion_partitions_t**>(void_parts);

  if (!*parts) *parts = new contra_legion_partitions_t;
  
  auto parent = (*parts)->getIndexSpace(field->index_space.id);

  // index partition 
  auto & index_part =
    (*parts)->IndexPartitions[parent];

  if (!index_part) {
    index_part = std::make_unique<legion_index_partition_t>(
      legion_index_partition_create_equal(
          *runtime,
          *ctx,
          field->index_space,
          cs->index_space,
          /* granularity */ 1,
          /*color*/ AUTO_GENERATE_ID ));
  }

  // region partition
  auto & logical_part =
    (*parts)->LogicalPartitions[{field->field_id, index_part->id}];
  if (!logical_part) {
    logical_part = std::make_unique<legion_logical_partition_t>(
      legion_logical_partition_create(
          *runtime,
          *ctx,
          field->logical_region,
          *index_part));
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
/// partition destruction
//==============================================================================
void contra_legion_partitions_destroy(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    contra_legion_partitions_t ** parts)
{
  if (!*parts) return;
  
  for (auto & index_part : (*parts)->IndexPartitions)
    legion_index_partition_destroy(*runtime, *ctx, *index_part.second);
  
  for (auto & logical_part : (*parts)->LogicalPartitions)
    legion_logical_partition_destroy(*runtime, *ctx, *logical_part.second);
  
  delete *parts;
  *parts = nullptr;
}

//==============================================================================
// Split an index space
//==============================================================================
void contra_legion_split_range(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    legion_task_t * task,
    contra_legion_index_space_t * is)
{
  legion_domain_t color_domain = legion_task_get_index_domain(*task);
  legion_domain_point_t color = legion_task_get_index_point(*task);

  legion_index_space_t cs = legion_index_space_create_domain(
      *runtime,
      *ctx,
      color_domain);

  legion_index_partition_t index_part =
    legion_index_partition_create_equal(
        *runtime,
        *ctx,
        is->index_space,
        cs,
        /*granularity*/ 1,
        AUTO_GENERATE_ID);
  
  legion_index_space_t new_is = 
    legion_index_partition_get_index_subspace_domain_point(
        *runtime,
        index_part,
        color);

  legion_domain_t new_domain =
    legion_index_space_get_domain(*runtime, new_is);
  
  legion_rect_1d_t rect = legion_domain_get_rect_1d(new_domain);

  is->index_space = new_is;
  is->start = rect.lo.x[0];
  is->end = rect.hi.x[0] + 1;

  legion_index_partition_destroy(*runtime, *ctx, index_part);
  legion_index_space_destroy(*runtime, *ctx, cs);
}


//==============================================================================
// Split an index space
//==============================================================================
void contra_legion_range_from_index_partition(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
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
}

//==============================================================================
/// get field accessor
//==============================================================================
void contra_legion_get_accessor(
    legion_runtime_t * runtime,
    legion_physical_region_t **regionptr,
    uint32_t* num_regions,
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
}


//==============================================================================
/// Accessor write
//==============================================================================
void contra_legion_accessor_write(
    contra_legion_accessor_t * acc,
    const void * data,
    size_t data_size,
    int_t index = 0)
{
  byte_t * offset = static_cast<byte_t*>(acc->data) + data_size*index;
  memcpy(offset, data, data_size);
}

//==============================================================================
/// Accessor read
//==============================================================================
void contra_legion_accessor_read(
    contra_legion_accessor_t * acc,
    void * data,
    size_t data_size,
    int_t index = 0)
{
  const byte_t * offset = static_cast<const byte_t*>(acc->data) + data_size*index;
  memcpy(data, offset, data_size);
}

//==============================================================================
/// accessor destruction
//==============================================================================
void contra_legion_accessor_destroy(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    contra_legion_accessor_t * acc)
{
  legion_accessor_array_1d_destroy(acc->accessor);
}


} // extern

////////////////////////////////////////////////////////////////////////////////
// Legion tasker
////////////////////////////////////////////////////////////////////////////////

namespace contra {

using namespace llvm;
using namespace utils;

//==============================================================================
// Constructor
//==============================================================================
LegionTasker::LegionTasker(
    IRBuilder<> & TheBuilder,
    LLVMContext & TheContext)
  : AbstractTasker(TheBuilder, TheContext)
{
  VoidPtrType_ = llvmType<void*>(TheContext_);
  ByteType_ = VoidPtrType_->getPointerElementType();
  VoidType_ = llvmType<void>(TheContext_);
  SizeType_ = llvmType<std::size_t>(TheContext_);
  Int32Type_ = llvmType<int>(TheContext_);
  BoolType_ = llvmType<bool>(TheContext_);
  CharType_ = llvmType<char>(TheContext_);
  RealmIdType_ = llvmType<realm_id_t>(TheContext_);
  NumRegionsType_ = llvmType<std::uint32_t>(TheContext_); 
  TaskVariantIdType_ = llvmType<legion_variant_id_t>(TheContext_);
  TaskIdType_ = llvmType<legion_task_id_t>(TheContext_);
  ProcIdType_ = llvmType<legion_processor_kind_t>(TheContext_);
  MapperIdType_ = llvmType<legion_mapper_id_t>(TheContext_); 
  MappingTagIdType_ = llvmType<legion_mapping_tag_id_t>(TheContext_);
  FutureIdType_ = llvmType<unsigned>(TheContext_);
  CoordType_ = llvmType<legion_coord_t>(TheContext_);
  Point1dType_ = ArrayType::get(CoordType_, 1);
  IndexSpaceIdType_ = llvmType<legion_index_space_id_t>(TheContext_);
  IndexTreeIdType_ = llvmType<legion_index_tree_id_t>(TheContext_);
  TypeTagType_ = llvmType<legion_type_tag_t>(TheContext_);
  FieldSpaceIdType_ = llvmType<legion_field_space_id_t>(TheContext_);
  FieldIdType_ = llvmType<legion_field_id_t>(TheContext_);
  RegionTreeIdType_ = llvmType<legion_region_tree_id_t>(TheContext_);
  IndexPartitionIdType_ = llvmType<legion_index_partition_id_t>(TheContext_);

  TaskType_ = createOpaqueType("legion_task_t", TheContext_);
  RegionType_ = createOpaqueType("legion_physical_region_t", TheContext_);
  ContextType_ = createOpaqueType("legion_context_t", TheContext_);
  RuntimeType_ = createOpaqueType("legion_runtime_t", TheContext_);
  ExecSetType_ = createOpaqueType("legion_execution_constraint_set_t", TheContext_);
  LayoutSetType_ = createOpaqueType("legion_task_layout_constraint_set_t", TheContext_);
  PredicateType_ = createOpaqueType("legion_predicate_t", TheContext_);
  TaskLauncherType_ = createOpaqueType("legion_task_launcher_t", TheContext_);
  IndexLauncherType_ = createOpaqueType("legion_index_launcher_t", TheContext_);
  FutureType_ = createOpaqueType("legion_future_t", TheContext_);
  TaskConfigType_ = createTaskConfigOptionsType(TheContext_);
  TaskArgsType_ = createTaskArgumentsType(TheContext_);
  DomainPointType_ = createDomainPointType(TheContext_);
  Rect1dType_ = createRect1dType(TheContext_);
  DomainRectType_ = createDomainRectType(TheContext_);
  ArgMapType_ = createOpaqueType("legion_argument_map_t", TheContext_);
  FutureMapType_ = createOpaqueType("legion_future_map_t", TheContext_);
  IndexSpaceType_ = createIndexSpaceType(TheContext_);
  FieldSpaceType_ = createFieldSpaceType(TheContext_);
  FieldAllocatorType_ = createOpaqueType("legion_field_allocator_t", TheContext_);
  LogicalRegionType_ = createLogicalRegionType(TheContext_);
  IndexPartitionType_ = createIndexPartitionType(TheContext_);
  LogicalPartitionType_ = createLogicalPartitionType(TheContext_);
  AccessorArrayType_ = createOpaqueType("legion_accessor_array_1d_t", TheContext_);
  ByteOffsetType_ = createByteOffsetType(TheContext_);

  IndexSpaceDataType_ = createIndexSpaceDataType(TheContext_);
  FieldDataType_ = createFieldDataType(TheContext_);
  AccessorDataType_ = createAccessorDataType(TheContext_);
  PartitionDataType_ = createPartitionDataType(TheContext_);
}

//==============================================================================
// Create the opaque type
//==============================================================================
StructType * LegionTasker::createOpaqueType(
    const std::string & Name,
    LLVMContext & TheContext)
{
  auto OpaqueType = StructType::create( TheContext, Name );

  std::vector<Type*> members{ VoidPtrType_ }; 
  OpaqueType->setBody( members );

  return OpaqueType;
}

//==============================================================================
// Create the opaque type
//==============================================================================
StructType * LegionTasker::createTaskConfigOptionsType(LLVMContext & TheContext)
{
  std::vector<Type*> members(4, BoolType_);
  auto OptionsType = StructType::create( TheContext, members, "task_config_options_t" );
  return OptionsType;
}

//==============================================================================
// Create the opaque type
//==============================================================================
StructType * LegionTasker::createTaskArgumentsType(LLVMContext & TheContext)
{
  std::vector<Type*> members = { VoidPtrType_, SizeType_ };
  auto NewType = StructType::create( TheContext, members, "legion_task_argument_t" );
  return NewType;
}

//==============================================================================
// Create the opaque type
//==============================================================================
StructType * LegionTasker::createDomainPointType(LLVMContext & TheContext)
{
  auto ArrayT = ArrayType::get(CoordType_, MAX_POINT_DIM); 
  std::vector<Type*> members = { Int32Type_, ArrayT };
  auto NewType = StructType::create( TheContext, members, "legion_domain_point_t" );
  return NewType;
}

//==============================================================================
// Create the opaque type
//==============================================================================
StructType * LegionTasker::createRect1dType(LLVMContext & TheContext)
{
  std::vector<Type*> members = { Point1dType_, Point1dType_ };
  auto NewType = StructType::create( TheContext, members, "legion_rect_1d_t" );
  return NewType;
}
  

//==============================================================================
// Create the opaque type
//==============================================================================
StructType * LegionTasker::createDomainRectType(LLVMContext & TheContext)
{
  auto ArrayT = ArrayType::get(CoordType_, 2*LEGION_MAX_DIM); 
  std::vector<Type*> members = { RealmIdType_, Int32Type_, ArrayT };
  auto NewType = StructType::create( TheContext, members, "legion_domain_t" );
  return NewType;
}

//==============================================================================
// Create the index space type
//==============================================================================
StructType * LegionTasker::createIndexSpaceType(LLVMContext & TheContext)
{
  std::vector<Type*> members = { IndexSpaceIdType_, IndexTreeIdType_, TypeTagType_ };
  auto NewType = StructType::create( TheContext, members, "legion_index_space_t" );
  return NewType;
}

//==============================================================================
// Create the field space type
//==============================================================================
StructType * LegionTasker::createFieldSpaceType(LLVMContext & TheContext)
{
  std::vector<Type*> members = { FieldSpaceIdType_ };
  auto NewType = StructType::create( TheContext, members, "legion_field_space_t" );
  return NewType;
}

//==============================================================================
// Create the logical region type
//==============================================================================
StructType * LegionTasker::createLogicalRegionType(LLVMContext & TheContext)
{
  std::vector<Type*> members = { RegionTreeIdType_, IndexSpaceType_, FieldSpaceType_ };
  auto NewType = StructType::create( TheContext, members, "legion_logical_region_t" );
  return NewType;
}

//==============================================================================
// Create the logical region type
//==============================================================================
StructType * LegionTasker::createIndexPartitionType(LLVMContext & TheContext)
{
  std::vector<Type*> members = { IndexPartitionIdType_, IndexTreeIdType_, TypeTagType_ };
  auto NewType = StructType::create( TheContext, members, "legion_index_partition_t" );
  return NewType;
}

//==============================================================================
// Create the logical region type
//==============================================================================
StructType * LegionTasker::createLogicalPartitionType(LLVMContext & TheContext)
{
  std::vector<Type*> members = { RegionTreeIdType_, IndexPartitionType_, FieldSpaceType_ };
  auto NewType = StructType::create( TheContext, members, "legion_index_partition_t" );
  return NewType;
}

//==============================================================================
// Create the byte offset type
//==============================================================================
StructType * LegionTasker::createByteOffsetType(LLVMContext & TheContext)
{
  std::vector<Type*> members = { Int32Type_ };
  auto NewType = StructType::create( TheContext, members, "legion_byte_offset_t" );
  return NewType;
}


//==============================================================================
// Create the field data type
//==============================================================================
StructType * LegionTasker::createFieldDataType(LLVMContext & TheContext)
{
  std::vector<Type*> members = {
    IndexSpaceType_,
    FieldSpaceType_,
    FieldAllocatorType_,
    FieldIdType_,
    LogicalRegionType_ };
  auto NewType = StructType::create( TheContext, members, "contra_legion_field_t" );
  return NewType;
}

//==============================================================================
// Create the field data type
//==============================================================================
StructType * LegionTasker::createAccessorDataType(LLVMContext & TheContext)
{
  std::vector<Type*> members = { FieldIdType_, RegionType_, LogicalRegionType_, 
    DomainRectType_, Rect1dType_, Rect1dType_, ByteOffsetType_,
    AccessorArrayType_, VoidPtrType_ };
  auto NewType = StructType::create( TheContext, members, "contra_legion_accessor_t" );
  return NewType;
}

//==============================================================================
// Create the field data type
//==============================================================================
StructType * LegionTasker::createIndexSpaceDataType(LLVMContext & TheContext)
{
  auto IntT = llvmType<int_t>(TheContext_);
  std::vector<Type*> members = { IntT, IntT, IndexSpaceType_ };
  auto NewType = StructType::create( TheContext, members, "contra_legion_index_space_t" );
  return NewType;
}

//==============================================================================
// Create the field data type
//==============================================================================
StructType * LegionTasker::createPartitionDataType(LLVMContext & TheContext)
{
  std::vector<Type*> members = { IndexPartitionType_, LogicalPartitionType_ };
  auto NewType = StructType::create( TheContext, members, "contra_legion_partition_t" );
  return NewType;
}

  
//==============================================================================
// Create a true predicate
//==============================================================================
AllocaInst* LegionTasker::createPredicateTrue(Module &TheModule)
{
  auto PredicateRT = reduceStruct(PredicateType_, TheModule);
  
  auto PredicateTrueT = FunctionType::get(PredicateRT, false);
  auto PredicateTrueF = TheModule.getOrInsertFunction(
      "legion_predicate_true", PredicateTrueT);
  
  Value* PredicateRV = Builder_.CreateCall(PredicateTrueF, None, "pred_true");
  
  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  auto PredicateA = createEntryBlockAlloca(TheFunction, PredicateType_, "predicate.alloca");
  store(PredicateRV, PredicateA);

  return PredicateA;
}

//==============================================================================
// Codegen the global arguments
//==============================================================================
AllocaInst* LegionTasker::createGlobalArguments(
    Module &TheModule,
    const std::vector<Value*> & ArgVorAs)
{
  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  
  auto TaskArgsA = createEntryBlockAlloca(TheFunction, TaskArgsType_, "args.alloca");

  //----------------------------------------------------------------------------
  // Identify futures
  
  auto NumArgs = ArgVorAs.size();
  
  std::vector<Constant*> ArgEnumC(NumArgs);
  std::vector<unsigned> ValueArgId;
  std::vector<unsigned> FieldArgId;
  
  auto ArgSizesT = ArrayType::get(SizeType_, NumArgs);
  auto ArgSizesA = createEntryBlockAlloca(TheFunction, ArgSizesT);


  for (unsigned i=0; i<NumArgs; i++) {
    ArgType ArgEnum;
    auto ArgVorA = ArgVorAs[i];
    if (isFuture(ArgVorA)) {
      ArgEnum = ArgType::Future;
    }
    else if (isField(ArgVorA)) {
      ArgEnum = ArgType::Field;
      FieldArgId.emplace_back(i);
    }
    else {
      ValueArgId.emplace_back(i);
      ArgEnum = ArgType::None;
    }
    ArgEnumC[i] = llvmValue(TheContext_, CharType_, static_cast<char>(ArgEnum));
  
    auto ArgSizeV = getSize(ArgVorA, SizeType_);
    storeStructMember(ArgSizeV, ArgSizesA, i);
  }

  //----------------------------------------------------------------------------
  // First count sizes

  auto ArgSizeGEP = accessStructMember(TaskArgsA, 1, "arglen");
  auto ArgSizeT = TaskArgsType_->getElementType(1);

  // add 1 byte for each argument first
  Builder_.CreateStore( llvmValue(TheContext_, ArgSizeT, NumArgs), ArgSizeGEP );

  // count user argument sizes
  for (auto i : ValueArgId) {
    ArgSizeGEP = accessStructMember(TaskArgsA, 1, "arglen");
    auto ArgSizesV = Builder_.CreateLoad(ArgSizesT, ArgSizesA);
    auto ArgSizeV = Builder_.CreateExtractValue(ArgSizesV, i);
    increment(ArgSizeGEP, ArgSizeV, "addoffset");
  }
  
  // add 8 bytes for each field argument
  auto NumFieldArgs = FieldArgId.size();
  auto FieldDataSizeV = llvmValue(TheContext_, ArgSizeT, NumFieldArgs*8);
  ArgSizeGEP = accessStructMember(TaskArgsA, 1, "arglen");
  increment(ArgSizeGEP, FieldDataSizeV, "addoffset");

  //----------------------------------------------------------------------------
  // Allocate storate
 
  auto ArgSizeV = loadStructMember(TaskArgsA, 1, "arglen");
  auto TmpA = Builder_.CreateAlloca(ByteType_, nullptr); // not needed but InsertAtEnd doesnt work
  auto MallocI = CallInst::CreateMalloc(TmpA, SizeType_, ByteType_, ArgSizeV,
      nullptr, nullptr, "args" );
  TmpA->eraseFromParent();
  storeStructMember(MallocI, TaskArgsA, 0, "args");
  
  //----------------------------------------------------------------------------
  // create an array with booleans identifying argyment type
  
  auto ArrayT = ArrayType::get(CharType_, NumArgs);
  auto ArrayC = ConstantArray::get(ArrayT, ArgEnumC);
  auto GVStr = new GlobalVariable(TheModule, ArrayT, true, GlobalValue::InternalLinkage, ArrayC);
  auto ZeroC = Constant::getNullValue(Int32Type_);
  auto ArrayGEP = ConstantExpr::getGetElementPtr(nullptr, GVStr, ZeroC, true);
  auto ArgDataPtrV = loadStructMember(TaskArgsA, 0, "args");
  Builder_.CreateMemCpy(ArgDataPtrV, 1, ArrayGEP, 1, llvmValue<size_t>(TheContext_, NumArgs)); 
 
  //----------------------------------------------------------------------------
  // Copy args

  // add 1 byte for each argument first
  ArgSizeGEP = accessStructMember(TaskArgsA, 1, "arglen");
  Builder_.CreateStore( llvmValue(TheContext_, ArgSizeT, NumArgs), ArgSizeGEP );
  
  for (auto i : ValueArgId) {
    auto ArgV = getAsAlloca(Builder_, TheFunction, ArgVorAs[i]);
    // load offset
    ArgSizeV = loadStructMember(TaskArgsA, 1, "arglen");
    // copy
    auto ArgDataPtrV = loadStructMember(TaskArgsA, 0, "args");
    serialize(ArgV, ArgDataPtrV, ArgSizeGEP);
    // increment
    ArgSizeGEP = accessStructMember(TaskArgsA, 1, "arglen");
    auto ArgSizesV = Builder_.CreateLoad(ArgSizesT, ArgSizesA);
    auto ArgSizeV = Builder_.CreateExtractValue(ArgSizesV, i);
    increment(ArgSizeGEP, ArgSizeV, "addoffset");
  }
  
  //----------------------------------------------------------------------------
  // Add field identifiers
    
  auto FieldDataPtrT = FieldDataType_->getPointerTo();
  auto UnsignedT = llvmType<unsigned>(TheContext_);
  auto FieldDataT = FunctionType::get(VoidType_, {FieldDataPtrT, UnsignedT, VoidPtrType_}, false);
  auto FieldDataF = TheModule.getOrInsertFunction("contra_legion_pack_field_data", FieldDataT);

  unsigned regidx = 0;
  for (auto i : FieldArgId) {
    Value* ArgV = ArgVorAs[i];
    // load offset
    ArgSizeV = loadStructMember(TaskArgsA, 1, "arglen");
    // offset data pointer
    auto ArgDataPtrV = loadStructMember(TaskArgsA, 0, "args");
    auto OffsetArgDataPtrV = Builder_.CreateGEP(ArgDataPtrV, ArgSizeV, "args.offset");
    // pack field info
    auto ArgSizeGEP = accessStructMember(TaskArgsA, 1, "arglen");
    std::vector<Value*> ArgVs = {
      ArgV,
      llvmValue(TheContext_, UnsignedT, regidx++),
      OffsetArgDataPtrV };
    Builder_.CreateCall(FieldDataF, ArgVs);
    // increment
    auto FieldDataSizeV = llvmValue(TheContext_, ArgSizeT, 8);
    increment(ArgSizeGEP, FieldDataSizeV, "addoffset");
  }
  

  return TaskArgsA; 
}

//==============================================================================
// Codegen the global future arguments
//==============================================================================
void LegionTasker::createGlobalFutures(
    llvm::Module & TheModule,
    Value* LauncherA,
    const std::vector<Value*> & ArgVorAs,
    bool IsIndex )
{
  auto TheFunction = Builder_.GetInsertBlock()->getParent();

  auto FutureRT = reduceStruct(FutureType_, TheModule);

  StructType* LauncherT = IsIndex ? IndexLauncherType_ : TaskLauncherType_;
  auto LauncherRT = reduceStruct(LauncherT, TheModule);

  std::vector<Type*> AddFutureArgTs = {LauncherRT, FutureRT};
  auto AddFutureT = FunctionType::get(VoidType_, AddFutureArgTs, false);

  FunctionCallee AddFutureF;
  if (IsIndex)
    AddFutureF = TheModule.getOrInsertFunction("legion_index_launcher_add_future", AddFutureT);
  else
    AddFutureF = TheModule.getOrInsertFunction("legion_task_launcher_add_future", AddFutureT);

  auto NumArgs = ArgVorAs.size();
  for (unsigned i=0; i<NumArgs; i++) {
    auto FutureV = ArgVorAs[i];
    if (!isFuture(FutureV)) continue;
    FutureV = getAsAlloca(Builder_, TheFunction, FutureV);
    auto FutureRV = load(FutureV, TheModule, "future");
    auto LauncherRV = load(LauncherA, TheModule, "task_launcher");
    std::vector<Value*> AddFutureArgVs = {LauncherRV, FutureRV};
    Builder_.CreateCall(AddFutureF, AddFutureArgVs);
  }
}

//==============================================================================
// Create the partition data
//==============================================================================
AllocaInst* LegionTasker::createPartitionInfo(
    llvm::Module & TheModule )
{
  auto TheFunction = Builder_.GetInsertBlock()->getParent();

  auto PartInfoA = createEntryBlockAlloca(TheFunction, VoidPtrType_, "indexpartinfo");
  auto NullV = Constant::getNullValue(VoidPtrType_);
  Builder_.CreateStore(NullV, PartInfoA);
    
  return PartInfoA;
}

//==============================================================================
// Destroy the partition data
//==============================================================================
void LegionTasker::destroyPartitionInfo(
    llvm::Module & TheModule )
{
  const auto & LegionE = getCurrentTask();
  const auto & ContextA = LegionE.ContextAlloca;
  const auto & RuntimeA = LegionE.RuntimeAlloca;
  const auto & PartInfoA = LegionE.PartInfoAlloca;

  if (PartInfoA) {
    std::vector<Value*> FunArgVs = { RuntimeA, ContextA, PartInfoA };
    auto FunArgTs = llvmTypes(FunArgVs);
    auto FunT = FunctionType::get(VoidType_, FunArgTs, false);
    auto FunF = TheModule.getOrInsertFunction("contra_legion_partitions_destroy", FunT);
    Builder_.CreateCall(FunF, FunArgVs);
  }

}


//==============================================================================
// Codegen the field arguments
//==============================================================================
void LegionTasker::createFieldArguments(
    llvm::Module & TheModule,
    Value* LauncherA,
    const std::vector<Value*> & ArgVorAs,
    Value* IndexSpaceA,
    Value* PartInfoA )
{
  auto TheFunction = Builder_.GetInsertBlock()->getParent();

  auto NumArgs = ArgVorAs.size();
  
  //----------------------------------------------------------------------------
  // Add region requirements
  if (IndexSpaceA) {
  
    const auto & LegionE = getCurrentTask();
    const auto & ContextA = LegionE.ContextAlloca;
    const auto & RuntimeA = LegionE.RuntimeAlloca;

    std::vector<Type*> FunArgTs = {
      RuntimeA->getType(),
      ContextA->getType(),
      IndexLauncherType_->getPointerTo(),
      IndexSpaceA->getType(),
      VoidPtrType_->getPointerTo(),
      VoidPtrType_->getPointerTo(),
      FieldDataType_->getPointerTo()
    };

    auto FunT = FunctionType::get(VoidType_, FunArgTs, false);
    auto FunF = TheModule.getOrInsertFunction("contra_legion_index_add_region_requirement", FunT);

    for (unsigned i=0; i<NumArgs; i++) {
      auto FieldA = ArgVorAs[i];
      if (!isField(FieldA)) continue;
      FieldA = getAsAlloca(Builder_, TheFunction, FieldA);
      std::vector<Value*> FunArgVs = {
        RuntimeA,
        ContextA,
        LauncherA,
        IndexSpaceA,
        PartInfoA,
        FieldA
      };
      Builder_.CreateCall(FunF, FunArgVs);
    }
  }
  //----------------------------------------------------------------------------
  else {
    
    auto LauncherRT = reduceStruct(TaskLauncherType_, TheModule);

    std::vector<Type*> FunArgTs = {
      LauncherRT,
      FieldDataType_->getPointerTo() };

    auto FunT = FunctionType::get(VoidType_, FunArgTs, false);
    auto FunF = TheModule.getOrInsertFunction("contra_legion_task_add_region_requirement", FunT);
  
    for (unsigned i=0; i<NumArgs; i++) {
      auto FieldV = ArgVorAs[i];
      if (!isField(FieldV)) continue;
      FieldV = getAsAlloca(Builder_, TheFunction, FieldV); 
      std::vector<Value*> FunArgVs = {LauncherA, FieldV};
      Builder_.CreateCall(FunF, FunArgVs);
    }
  }
  //----------------------------------------------------------------------------
  
}


//==============================================================================
// Destroy an opaque type
//==============================================================================
AllocaInst* LegionTasker::createOpaqueType(
    Module& TheModule,
    StructType* OpaqueT,
    const std::string & FuncN,
    const std::string & Name)
{
  auto OpaqueRT = reduceStruct(OpaqueT, TheModule);
  
  auto ExecT = FunctionType::get(OpaqueRT, false);
  auto ExecF = TheModule.getOrInsertFunction(FuncN, ExecT);
  
  Value* OpaqueRV = Builder_.CreateCall(ExecF, None, Name);
  
  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  auto OpaqueA = createEntryBlockAlloca(TheFunction, OpaqueT, Name);
  store(OpaqueRV, OpaqueA);
  return OpaqueA;
}
  

//==============================================================================
// Destroy an opaque type
//==============================================================================
void LegionTasker::destroyOpaqueType(
    Module& TheModule,
    Value* OpaqueA,
    const std::string & FuncN,
    const std::string & Name)
{
  auto OpaqueRV = load(OpaqueA, TheModule, Name);
  auto OpaqueRT = OpaqueRV->getType();

  auto DestroyOpaqueT = FunctionType::get(VoidType_, OpaqueRT, false);
  auto DestroyOpaqueF = TheModule.getOrInsertFunction(FuncN, DestroyOpaqueT);

  Builder_.CreateCall(DestroyOpaqueF, OpaqueRV);
}
  
//==============================================================================
// Destroy task arguments
//==============================================================================
void LegionTasker::destroyGlobalArguments(
    Module& TheModule,
    AllocaInst* TaskArgsA)
{
  auto ArgDataPtrV = loadStructMember(TaskArgsA, 0, "args");
  auto TmpA = Builder_.CreateAlloca(VoidPtrType_, nullptr); // not needed but InsertAtEnd doesnt work
  CallInst::CreateFree(ArgDataPtrV, TmpA);
  TmpA->eraseFromParent();
}


//==============================================================================
// create registration arguments
//==============================================================================
void LegionTasker::createRegistrationArguments(
    Module& TheModule,
    llvm::AllocaInst *& ExecSetA,
    llvm::AllocaInst *& LayoutSetA,
    llvm::AllocaInst *& TaskConfigA)
{
  // get current insertion point
  auto TheFunction = Builder_.GetInsertBlock()->getParent();

  //----------------------------------------------------------------------------
  // execution_constraint_set
  
  ExecSetA = createOpaqueType(TheModule, ExecSetType_,
      "legion_execution_constraint_set_create", "execution_constraint");
  
  //----------------------------------------------------------------------------
  // add constraint
  
  auto ProcIdV = llvmValue(TheContext_, ProcIdType_, LOC_PROC);  
  auto ExecSetV = load(ExecSetA, TheModule, "execution_constraint");
  std::vector<Value*> AddExecArgVs = {ExecSetV, ProcIdV};
  auto AddExecArgTs = llvmTypes(AddExecArgVs);
  auto AddExecT = FunctionType::get(VoidType_, AddExecArgTs, false);
  auto AddExecF = TheModule.getOrInsertFunction(
      "legion_execution_constraint_set_add_processor_constraint", AddExecT);

  Builder_.CreateCall(AddExecF, AddExecArgVs);

  //----------------------------------------------------------------------------
  // task_layout_constraint_set
 
  LayoutSetA = createOpaqueType(TheModule, LayoutSetType_,
      "legion_task_layout_constraint_set_create", "layout_constraint"); 
  
  //----------------------------------------------------------------------------
  // options
  TaskConfigA = createEntryBlockAlloca(TheFunction, TaskConfigType_, "options");
  auto BoolT = TaskConfigType_->getElementType(0);
  auto FalseV = Constant::getNullValue(BoolT);
  Builder_.CreateMemSet(TaskConfigA, FalseV, 4, 1); 
}

//==============================================================================
// Create the function wrapper
//==============================================================================
LegionTasker::PreambleResult LegionTasker::taskPreamble(
    Module &TheModule,
    const std::string & TaskName,
    const std::vector<std::string> & TaskArgNs,
    const std::vector<Type*> & TaskArgTs,
    bool IsIndex,
    const std::map<std::string, VariableType> & VarOverrides)
{
  //----------------------------------------------------------------------------
  // Create task wrapper
  
  std::vector<Type *> WrapperArgTs =
    {VoidPtrType_, SizeType_, VoidPtrType_, SizeType_, RealmIdType_};
  
  auto WrapperT = FunctionType::get(VoidType_, WrapperArgTs, false);
  auto WrapperF = Function::Create(WrapperT, Function::ExternalLinkage,
      TaskName, &TheModule);

  auto Arg = WrapperF->arg_begin();
  Arg->setName("data");
  Arg->addAttr(Attribute::ReadOnly);
  Arg++;
  Arg->setName("datalen");
  Arg->addAttr(Attribute::ReadOnly);
  Arg++;
  Arg->setName("userdata");
  Arg->addAttr(Attribute::ReadOnly);
  Arg++;
  Arg->setName("userlen");
  Arg->addAttr(Attribute::ReadOnly);
  Arg++;
  Arg->setName("procid");
  Arg->addAttr(Attribute::ReadOnly);

  // Create a new basic block to start insertion into.
  BasicBlock *BB = BasicBlock::Create(TheContext_, "entry", WrapperF);
  Builder_.SetInsertPoint(BB);

  // create the context
  auto & LegionE = startTask();
  auto & ContextA = LegionE.ContextAlloca;
  auto & RuntimeA = LegionE.RuntimeAlloca;
  ContextA = createEntryBlockAlloca(WrapperF, ContextType_, "context.alloca");
  RuntimeA = createEntryBlockAlloca(WrapperF, RuntimeType_, "runtime.alloca");


  // allocate arguments
  std::vector<Value*> WrapperArgVs;
  WrapperArgVs.reserve(WrapperArgTs.size());

  unsigned ArgIdx = 0;
  for (auto &Arg : WrapperF->args()) {
    // get arg type
    auto ArgT = WrapperArgTs[ArgIdx];
    // Create an alloca for this variable.
    auto ArgN = std::string(Arg.getName()) + ".alloca";
    auto Alloca = createEntryBlockAlloca(WrapperF, ArgT, ArgN);
    // Store the initial value into the alloca.
    Builder_.CreateStore(&Arg, Alloca);
    WrapperArgVs.emplace_back(Alloca);
    ArgIdx++;
  }

  // loads
  auto DataV = Builder_.CreateLoad(VoidPtrType_, WrapperArgVs[0], "data");
  auto DataLenV = Builder_.CreateLoad(SizeType_, WrapperArgVs[1], "datalen");
  //auto UserDataV = Builder_.CreateLoad(VoidPtrType_, WrapperArgVs[2], "userdata");
  //auto UserLenV = Builder_.CreateLoad(SizeType_, WrapperArgVs[3], "userlen");
  auto ProcIdV = Builder_.CreateLoad(RealmIdType_, WrapperArgVs[4], "proc_id");

  //----------------------------------------------------------------------------
  // call to preamble

  // create temporaries
  auto TaskA = createEntryBlockAlloca(WrapperF, TaskType_, "task.alloca");
 
  auto RegionsT = RegionType_->getPointerTo();
  auto RegionsA = createEntryBlockAlloca(WrapperF, RegionsT, "regions.alloca");
  auto NullV = Constant::getNullValue(RegionsT);
  Builder_.CreateStore(NullV, RegionsA);

  auto NumRegionsA = createEntryBlockAlloca(WrapperF, NumRegionsType_, "num_regions");
  auto ZeroV = llvmValue(TheContext_, NumRegionsType_, 0);
  Builder_.CreateStore(ZeroV, NumRegionsA);
 
  // args
  std::vector<Value*> PreambleArgVs = { DataV, DataLenV, ProcIdV,
    TaskA, RegionsA, NumRegionsA, ContextA, RuntimeA };
  auto PreambleArgTs = llvmTypes(PreambleArgVs);
  
  auto PreambleT = FunctionType::get(VoidType_, PreambleArgTs, false);
  auto PreambleF = TheModule.getOrInsertFunction("legion_task_preamble", PreambleT);
  
  Builder_.CreateCall(PreambleF, PreambleArgVs);
  
  //----------------------------------------------------------------------------
  // Get task args

  auto TaskRV = load(TaskA, TheModule, "task");
  auto TaskGetArgsT = FunctionType::get(VoidPtrType_, TaskRV->getType(), false);
  auto TaskGetArgsF = TheModule.getOrInsertFunction("legion_task_get_args", TaskGetArgsT);
  
  Value* TaskArgsV = Builder_.CreateCall(TaskGetArgsF, TaskRV, "args");
  
  auto TaskArgsA = createEntryBlockAlloca(WrapperF, VoidPtrType_, "args.alloca");
  Builder_.CreateStore(TaskArgsV, TaskArgsA);
  
  //----------------------------------------------------------------------------
  // Allocas for task arguments

  auto NumArgs = TaskArgTs.size();
  std::vector<AllocaInst*> TaskArgAs;
  for (unsigned i=0; i<NumArgs; ++i) {
    auto ArgN = TaskArgNs[i];
    auto ArgStr = ArgN + ".alloca";
    auto vit = VarOverrides.find(ArgN);
    if (vit != VarOverrides.end() && vit->second.isPartition()) {
      auto ArgT = IndexPartitionType_;
      auto ArgA = createEntryBlockAlloca(WrapperF, ArgT, ArgStr);
      TaskArgAs.emplace_back(ArgA);
    }
    else {
      auto ArgT = TaskArgTs[i];
      auto ArgA = createEntryBlockAlloca(WrapperF, ArgT, ArgStr);
      TaskArgAs.emplace_back(ArgA);
    }
  }
  
  //----------------------------------------------------------------------------
  // get user types

  auto ArrayT = ArrayType::get(CharType_, NumArgs);
  auto ArrayA = createEntryBlockAlloca(WrapperF, ArrayT, "isfuture");
  auto ArgDataPtrV = Builder_.CreateLoad(VoidPtrType_, TaskArgsA, "args");
  Builder_.CreateMemCpy(ArrayA, 1, ArgDataPtrV, 1, llvmValue<size_t>(TheContext_, NumArgs)); 
  
  //----------------------------------------------------------------------------
  // unpack user variables
  
  auto OffsetT = SizeType_;
  auto OffsetA = createEntryBlockAlloca(WrapperF, OffsetT, "offset.alloca");
  Builder_.CreateStore( llvmValue(TheContext_, OffsetT, NumArgs), OffsetA );
  
  for (unsigned i=0; i<NumArgs; i++) {
    // Create blocks for the then and else cases.
    BasicBlock * ThenBB = BasicBlock::Create(TheContext_, "then", WrapperF);
    BasicBlock * MergeBB = BasicBlock::Create(TheContext_, "ifcont");
    // evaluate the condition
    auto ArrayV = Builder_.CreateLoad(ArrayT, ArrayA, "isfuture");
    auto ArgTypeV = Builder_.CreateExtractValue(ArrayV, i);
    auto EnumV = llvmValue(TheContext_, ArgTypeV->getType(), static_cast<char>(ArgType::None));
    auto CondV = Builder_.CreateICmpEQ(ArgTypeV, EnumV, "argtype");
    Builder_.CreateCondBr(CondV, ThenBB, MergeBB);
    // Emit then block
    Builder_.SetInsertPoint(ThenBB);
    // copy
    auto ArgSizeV = deserialize(TaskArgAs[i], TaskArgsA, OffsetA);
    // increment
    increment(OffsetA, ArgSizeV, "offset");
    // finish then
    ThenBB->getFirstNonPHI();
    Builder_.CreateBr(MergeBB);
    ThenBB = Builder_.GetInsertBlock();
    // Emit merge block.
    WrapperF->getBasicBlockList().push_back(MergeBB);
    Builder_.SetInsertPoint(MergeBB);
  }

  //----------------------------------------------------------------------------
  // partition any ranges
  if (IsIndex) { 
  
    std::vector<Type*> SplitRangeArgTs = {
      RuntimeType_->getPointerTo(),
      ContextType_->getPointerTo(),
      TaskType_->getPointerTo(),
      IndexSpaceDataType_->getPointerTo()
    };

    auto SplitRangeT = FunctionType::get(VoidType_, SplitRangeArgTs, false);
    auto SplitRangeF = TheModule.getOrInsertFunction("contra_legion_split_range", SplitRangeT);
    
    std::vector<Type*> GetRangeArgTs = {
      RuntimeType_->getPointerTo(),
      ContextType_->getPointerTo(),
      TaskType_->getPointerTo(),
      IndexPartitionType_->getPointerTo(),
      IndexSpaceDataType_->getPointerTo()
    };

    auto GetRangeT = FunctionType::get(VoidType_, GetRangeArgTs, false);
    auto GetRangeF = TheModule.getOrInsertFunction(
        "contra_legion_range_from_index_partition",
        GetRangeT);

    for (unsigned i=0; i<NumArgs; i++) {
      auto ArgN = TaskArgNs[i];
      auto vit = VarOverrides.find(ArgN);
      auto ForceIndex = (vit != VarOverrides.end() && vit->second.isPartition());
      if (isRange(TaskArgAs[i])) {
        Builder_.CreateCall(SplitRangeF, {RuntimeA, ContextA, TaskA, TaskArgAs[i]});
      }
      else if (ForceIndex || isPartition(TaskArgAs[i])) {
        auto ArgA = createEntryBlockAlloca(WrapperF, IndexSpaceDataType_, ArgN);
        Builder_.CreateCall(GetRangeF, {RuntimeA, ContextA, TaskA, TaskArgAs[i], ArgA});
        TaskArgAs[i] = ArgA;
      }
    }
  
  }
  

  //----------------------------------------------------------------------------
  // unpack future variables
  
  auto FutureRT = reduceStruct(FutureType_, TheModule);
  auto TaskRT = reduceStruct(TaskType_, TheModule);
  std::vector<Type*> GetFutureArgTs = {TaskRT, FutureIdType_};

  auto GetFutureT = FunctionType::get(FutureRT, GetFutureArgTs, false);
  auto GetFutureF = TheModule.getOrInsertFunction("legion_task_get_future", GetFutureT);
  
  auto FutureIndexT = FutureIdType_;
  auto FutureIndexA = createEntryBlockAlloca(WrapperF, FutureIndexT, "futureid.alloca");
  Builder_.CreateStore( Constant::getNullValue(FutureIndexT), FutureIndexA );
  
  for (unsigned i=0; i<NumArgs; i++) {
    // Create blocks for the then and else cases.
    BasicBlock * ThenBB = BasicBlock::Create(TheContext_, "then", WrapperF);
    BasicBlock * MergeBB = BasicBlock::Create(TheContext_, "ifcont");
    // evaluate the condition
    auto ArrayV = Builder_.CreateLoad(ArrayT, ArrayA, "isfuture");
    auto ArgTypeV = Builder_.CreateExtractValue(ArrayV, i);
    auto EnumV = llvmValue(TheContext_, ArgTypeV->getType(), static_cast<char>(ArgType::Future));
    auto CondV = Builder_.CreateICmpEQ(ArgTypeV, EnumV, "argtype");
    Builder_.CreateCondBr(CondV, ThenBB, MergeBB);
    // Emit then block
    Builder_.SetInsertPoint(ThenBB);
    // get future
    auto TaskRV = load(TaskA, TheModule, "task");
    auto FutureIndexV = Builder_.CreateLoad(FutureIndexT, FutureIndexA, "futureid");
    std::vector<Value*> GetFutureArgVs = {TaskRV, FutureIndexV};
    auto FutureRV = Builder_.CreateCall(GetFutureF, GetFutureArgVs, "get_future");
    auto FutureA = createEntryBlockAlloca(WrapperF, FutureType_, "future");
    store(FutureRV, FutureA);
    // unpack
    auto ArgA = TaskArgAs[i];
    auto ArgT = ArgA->getType()->getPointerElementType();
    // copy
    auto ArgV = loadFuture(TheModule, FutureA, ArgT);
    Builder_.CreateStore( ArgV, ArgA );
    // consume the future
    destroyFuture(TheModule, FutureA);
    // increment
    increment(FutureIndexA, llvmValue(TheContext_, FutureIndexT, 1), "futureid");
    // finish then
    ThenBB->getFirstNonPHI();
    Builder_.CreateBr(MergeBB);
    ThenBB = Builder_.GetInsertBlock();
    // Emit merge block.
    WrapperF->getBasicBlockList().push_back(MergeBB);
    Builder_.SetInsertPoint(MergeBB);
  }
  
  //----------------------------------------------------------------------------
  // unpack Field variables
      
  auto UInt32Type = llvmType<uint32_t>(TheContext_);
  auto UInt32PtrType = UInt32Type->getPointerTo();
  auto GetFieldDataT =
    FunctionType::get(VoidType_, {VoidPtrType_, UInt32PtrType, UInt32PtrType}, false);
  auto GetFieldDataF =
    TheModule.getOrInsertFunction("contra_legion_unpack_field_data", GetFieldDataT);
    
  auto AccessorDataPtrT = AccessorDataType_->getPointerTo();
  std::vector<Type*> GetFieldArgTs = {
    RuntimeA->getType(),
    RegionsA->getType(),
    NumRegionsA->getType(),
    UInt32PtrType,
    UInt32PtrType,
    AccessorDataPtrT};
  auto GetFieldT = FunctionType::get(VoidType_, GetFieldArgTs, false); 
  auto GetFieldF = TheModule.getOrInsertFunction("contra_legion_get_accessor", GetFieldT);

  auto FieldIndexT = FieldIdType_;
  auto FieldIndexA = createEntryBlockAlloca(WrapperF, FieldIndexT, "field.alloca");
  Builder_.CreateStore( Constant::getNullValue(FieldIndexT), FieldIndexA );
  
  auto FieldIdA = createEntryBlockAlloca(WrapperF, UInt32Type, "fieldid.alloca");
  auto RegionIdA = createEntryBlockAlloca(WrapperF, UInt32Type, "regid.alloca");
  
  for (unsigned i=0; i<NumArgs; i++) {
    // Create blocks for the then and else cases.
    BasicBlock * ThenBB = BasicBlock::Create(TheContext_, "then", WrapperF);
    BasicBlock * MergeBB = BasicBlock::Create(TheContext_, "ifcont");
    // evaluate the condition
    auto ArrayV = Builder_.CreateLoad(ArrayT, ArrayA, "isfuture");
    auto ArgTypeV = Builder_.CreateExtractValue(ArrayV, i);
    auto EnumV = llvmValue(TheContext_, ArgTypeV->getType(), static_cast<char>(ArgType::Field));
    auto CondV = Builder_.CreateICmpEQ(ArgTypeV, EnumV, "argtype");
    Builder_.CreateCondBr(CondV, ThenBB, MergeBB);
    // Emit then block
    Builder_.SetInsertPoint(ThenBB);
    // unpack the field data
    auto ArgGEP = offsetPointer(TaskArgsA, OffsetA, "args");
    Builder_.CreateCall( GetFieldDataF, {ArgGEP, FieldIdA, RegionIdA} );
    // get field pointer
    auto TheBlock = Builder_.GetInsertBlock();
    auto ArgPtrV = CastInst::Create(Instruction::BitCast, TaskArgAs[i], AccessorDataPtrT,
      "cast", TheBlock);
    std::vector<Value*> GetFieldArgVs = {
      RuntimeA,
      RegionsA,
      NumRegionsA,
      RegionIdA,
      FieldIdA,
      ArgPtrV };
    Builder_.CreateCall( GetFieldF, GetFieldArgVs );
    // increment
    increment(FieldIndexA, llvmValue(TheContext_, FieldIndexT, 1), "fieldid");
    increment(OffsetA, llvmValue(TheContext_, OffsetT, 8), "offset");
    // finish then
    ThenBB->getFirstNonPHI();
    Builder_.CreateBr(MergeBB);
    ThenBB = Builder_.GetInsertBlock();
    // Emit merge block.
    WrapperF->getBasicBlockList().push_back(MergeBB);
    Builder_.SetInsertPoint(MergeBB);
  }

  //----------------------------------------------------------------------------
  // If this is an index task
  
  AllocaInst* IndexA = nullptr;
  if (IsIndex) {
     
    auto TaskRV = load(TaskA, TheModule, "task");
    auto DomainPointA = createEntryBlockAlloca(WrapperF, DomainPointType_, "domain_point.alloca");

    std::vector<Value*> GetIndexArgVs = {DomainPointA, TaskRV};
    auto GetIndexArgTs = llvmTypes(GetIndexArgVs);
    auto GetIndexT = FunctionType::get(VoidType_, GetIndexArgTs, false);
    auto GetIndexF = TheModule.getFunction("legion_task_get_index_point");
    if (!GetIndexF) {
      GetIndexF = Function::Create(GetIndexT, Function::InternalLinkage,
          "legion_task_get_index_point", &TheModule);
      auto Arg = GetIndexF->arg_begin();
      Arg->addAttr(Attribute::StructRet);
    }
 
    Builder_.CreateCall(GetIndexF, GetIndexArgVs);

    auto PointDataGEP = accessStructMember(DomainPointA, 1, "point_data");
    auto PointDataT = DomainPointType_->getElementType(1);
    auto PointDataV = Builder_.CreateLoad(PointDataT, PointDataGEP);
    auto IndexV = Builder_.CreateExtractValue(PointDataV, 0);

    IndexA = createEntryBlockAlloca(WrapperF, llvmType<int_t>(TheContext_), "index");
    Builder_.CreateStore( IndexV, IndexA );
  }

  //----------------------------------------------------------------------------
  // Function body
  return {WrapperF, TaskArgAs, IndexA}; 

}

//==============================================================================
// Create the function wrapper
//==============================================================================
LegionTasker::PreambleResult LegionTasker::taskPreamble(
    Module &TheModule,
    const std::string & Name,
    Function* TaskF)
{

  std::string TaskName = "__" + Name + "_task__";
  
  std::vector<Type*> TaskArgTs;
  std::vector<std::string> TaskArgNs;
 
  for (auto & Arg : TaskF->args()) {
    auto ArgT = Arg.getType();
    auto ArgN = Arg.getName().str();
    TaskArgTs.emplace_back(ArgT);
    TaskArgNs.emplace_back(ArgN);
  }

  return taskPreamble(TheModule, TaskName, TaskArgNs, TaskArgTs, false);
}
  
//==============================================================================
// Create the function wrapper
//==============================================================================
void LegionTasker::taskPostamble(Module &TheModule, Value* ResultV)
{

  Value* RetvalV = Constant::getNullValue(VoidPtrType_);
  Value* RetsizeV = llvmValue<std::size_t>(TheContext_, 0);

  AllocaInst* RetvalA;
  auto RetvalT = VoidPtrType_;

  
  //----------------------------------------------------------------------------
  // Have return value
  if (ResultV) {
    
    auto TheFunction = Builder_.GetInsertBlock()->getParent();

    // store result
    auto ResultT = ResultV->getType();
    auto ResultA = createEntryBlockAlloca(TheFunction, ResultT, "result");
    Builder_.CreateStore( ResultV, ResultA );

    // return size
    auto RetsizeT = RetsizeV->getType();
    RetsizeV = getSize(ResultV, RetsizeT);
    auto RetsizeA = createEntryBlockAlloca(TheFunction, RetsizeT, "retsize");
    Builder_.CreateStore( RetsizeV, RetsizeA );

    // allocate space for return value
    RetsizeV = Builder_.CreateLoad(RetsizeT, RetsizeA);
    
    auto TmpA = Builder_.CreateAlloca(ByteType_, nullptr); // not needed but InsertAtEnd doesnt work
    auto MallocI = CallInst::CreateMalloc(TmpA, RetsizeT, ByteType_, RetsizeV,
        nullptr, nullptr, "retval" );
    TmpA->eraseFromParent();

    RetvalA = createEntryBlockAlloca(TheFunction, RetvalT, "retval");
    Builder_.CreateStore(MallocI, RetvalA );

    // copy data
    RetvalV = Builder_.CreateLoad(RetvalT, RetvalA);
    RetsizeV = Builder_.CreateLoad(RetsizeT, RetsizeA);
    Builder_.CreateMemCpy(RetvalV, 1, ResultA, 1, RetsizeV); 
    serialize(ResultA, RetvalV);


    // final loads
    RetsizeV = Builder_.CreateLoad(RetsizeT, RetsizeA);
    RetvalV = Builder_.CreateLoad(RetvalT, RetvalA);
    
  }
  
  //----------------------------------------------------------------------------
  // Destroy Index partitions

  destroyPartitionInfo(TheModule);

  //----------------------------------------------------------------------------
  // Call postable
  
  // temporaries
  const auto & LegionE = getCurrentTask();
  const auto & ContextA = LegionE.ContextAlloca;
  const auto & RuntimeA = LegionE.RuntimeAlloca;

  auto RuntimeV = load(RuntimeA, TheModule, "runtime");
  auto ContextV = load(ContextA, TheModule, "context");
  
  // args
  std::vector<Value*> PostambleArgVs = { RuntimeV, ContextV, RetvalV, RetsizeV };
  sanitize(PostambleArgVs, TheModule);
  std::vector<Type*> PostambleArgTs = llvmTypes(PostambleArgVs);

  // call
  auto PostambleT = FunctionType::get(VoidType_, PostambleArgTs, false);
  auto PostambleF = TheModule.getOrInsertFunction("legion_task_postamble", PostambleT);
  
  Builder_.CreateCall(PostambleF, PostambleArgVs);
  
  //----------------------------------------------------------------------------
  // Free memory
  if (ResultV) {
    auto RetvalT = RetvalV->getType();
    RetvalV = Builder_.CreateLoad(RetvalT, RetvalA);
    auto TmpA = Builder_.CreateAlloca(VoidPtrType_, nullptr); // not needed but InsertAtEnd doesnt work
    CallInst::CreateFree(RetvalV, TmpA);
    TmpA->eraseFromParent();
  }

  
  finishTask();
}

//==============================================================================
// Postregister tasks
//==============================================================================
void LegionTasker::postregisterTask(
    Module &TheModule,
    const std::string & Name,
    const TaskInfo & Task )
{

  //----------------------------------------------------------------------------
  // arguments
  
  llvm::AllocaInst *ExecSetA, *LayoutSetA, *TaskConfigA;
  createRegistrationArguments(TheModule, ExecSetA, LayoutSetA, TaskConfigA);
  
  
  //----------------------------------------------------------------------------
  // registration
  
  auto TaskT = Task.getFunction()->getFunctionType();
  auto TaskF = TheModule.getOrInsertFunction(Task.getName(), TaskT).getCallee();
  
  Value* TaskIdV = llvmValue(TheContext_, TaskIdType_, Task.getId());
  auto TaskIdVariantV = llvmValue(TheContext_, TaskVariantIdType_, AUTO_GENERATE_ID);
  auto TaskNameV = llvmString(TheContext_, TheModule, Name + " task");
  auto VariantNameV = llvmString(TheContext_, TheModule, Name + " variant");

  auto ExecSetV = load(ExecSetA, TheModule, "execution_constraints");
  auto LayoutSetV = load(LayoutSetA, TheModule, "layout_constraints");
 
  auto TaskConfigV = load(TaskConfigA, TheModule, "options");

  auto UserDataV = Constant::getNullValue(VoidPtrType_);
  auto UserLenV = llvmValue<std::size_t>(TheContext_, 0);
  
  auto BoolT = TaskConfigType_->getElementType(0);
  auto TrueV = Constant::getNullValue(BoolT);

  std::vector<Value*> PreArgVs = { TaskIdV, TaskIdVariantV, TaskNameV,
    VariantNameV, TrueV, ExecSetV, LayoutSetV, TaskConfigV, TaskF, UserDataV, UserLenV };
  
  auto PreArgTs = llvmTypes(PreArgVs);
  auto PreRetT = TaskVariantIdType_;

  auto PreT = FunctionType::get(PreRetT, PreArgTs, false);
  auto PreF = TheModule.getOrInsertFunction(
      "legion_runtime_register_task_variant_fnptr", PreT);

  TaskIdV = Builder_.CreateCall(PreF, PreArgVs, "task_variant_id");
}

//==============================================================================
// Preregister tasks
//==============================================================================
void LegionTasker::preregisterTask(
    Module &TheModule,
    const std::string & Name,
    const TaskInfo & Task )
{

  //----------------------------------------------------------------------------
  // arguments
  
  llvm::AllocaInst *ExecSetA, *LayoutSetA, *TaskConfigA;
  createRegistrationArguments(TheModule, ExecSetA, LayoutSetA, TaskConfigA);
  
  //----------------------------------------------------------------------------
  // registration
  
  auto TaskT = Task.getFunctionType();
  auto TaskF = TheModule.getOrInsertFunction(Task.getName(), TaskT).getCallee();
  
  Value* TaskIdV = llvmValue(TheContext_, TaskIdType_, Task.getId());
  auto TaskIdVariantV = llvmValue(TheContext_, TaskVariantIdType_, AUTO_GENERATE_ID);
  auto TaskNameV = llvmString(TheContext_, TheModule, Name + " task");
  auto VariantNameV = llvmString(TheContext_, TheModule, Name + " variant");

  auto ExecSetV = load(ExecSetA, TheModule, "execution_constraints");
  auto LayoutSetV = load(LayoutSetA, TheModule, "layout_constraints");
 
  auto TaskConfigV = load(TaskConfigA, TheModule, "options");

  auto UserDataV = Constant::getNullValue(VoidPtrType_);
  auto UserLenV = llvmValue<std::size_t>(TheContext_, 0);

  std::vector<Value*> PreArgVs = { TaskIdV, TaskIdVariantV, TaskNameV,
    VariantNameV, ExecSetV, LayoutSetV, TaskConfigV, TaskF, UserDataV, UserLenV };
  
  auto PreArgTs = llvmTypes(PreArgVs);
  auto PreRetT = TaskVariantIdType_;

  auto PreT = FunctionType::get(PreRetT, PreArgTs, false);
  auto PreF = TheModule.getOrInsertFunction(
      "legion_runtime_preregister_task_variant_fnptr", PreT);

  TaskIdV = Builder_.CreateCall(PreF, PreArgVs, "task_variant_id");
  
  destroyOpaqueType(TheModule, ExecSetA, "legion_execution_constraint_set_destroy",
      "exec_set");
  destroyOpaqueType(TheModule, LayoutSetA, "legion_task_layout_constraint_set_destroy",
      "layout_set");
}
  
//==============================================================================
// Set top level task
//==============================================================================
void LegionTasker::setTopLevelTask(Module &TheModule, int TaskId )
{

  auto TaskIdV = llvmValue(TheContext_, TaskIdType_, TaskId);
  std::vector<Value*> SetArgVs = { TaskIdV };
  auto SetArgTs = llvmTypes(SetArgVs);

  auto SetT = FunctionType::get(VoidType_, SetArgTs, false);
  auto SetF = TheModule.getOrInsertFunction(
      "legion_runtime_set_top_level_task_id", SetT);

  Builder_.CreateCall(SetF, SetArgVs);
}
  
//==============================================================================
// start runtime
//==============================================================================
Value* LegionTasker::startRuntime(Module &TheModule, int Argc, char ** Argv)
{


  auto StartupT = FunctionType::get(VoidType_, VoidType_, false);
  auto StartupF = TheModule.getOrInsertFunction("contra_legion_startup", StartupT);
  Builder_.CreateCall(StartupF);

  auto VoidPtrArrayT = VoidPtrType_->getPointerTo();

  std::vector<Type*> StartArgTs = { Int32Type_, VoidPtrArrayT, BoolType_ };
  auto StartT = FunctionType::get(Int32Type_, StartArgTs, false);
  auto StartF = TheModule.getOrInsertFunction("legion_runtime_start", StartT);

  auto ArgcV = llvmValue(TheContext_, Int32Type_, Argc);

  std::vector<Constant*> ArgVs;
  for (int i=0; i<Argc; ++i) {
    ArgVs.emplace_back( llvmString(TheContext_, TheModule, Argv[i]) );
  }
  auto ArrayT = ArrayType::get(VoidPtrType_, Argc);
  auto ConstantArr = ConstantArray::get(ArrayT, ArgVs);
  auto GVStr = new GlobalVariable(TheModule, ArrayT, true,
      GlobalValue::InternalLinkage, ConstantArr);

  auto ZeroC = Constant::getNullValue(IntegerType::getInt32Ty(TheContext_));
  std::vector<Value*> IndicesC = {ZeroC, ZeroC};
  auto ArgvV = ConstantExpr::getGetElementPtr(
      nullptr, GVStr, IndicesC, true);

  auto BackV = llvmValue(TheContext_, BoolType_, false);

  std::vector<Value*> StartArgVs = { ArgcV, ArgvV, BackV };
  auto RetI = Builder_.CreateCall(StartF, StartArgVs, "start");
  return RetI;
}

//==============================================================================
// Launch a task
//==============================================================================
Value* LegionTasker::launch(
    Module &TheModule,
    const std::string & Name,
    int TaskId,
    const std::vector<Value*> & ArgVs)
{
  auto TheFunction = Builder_.GetInsertBlock()->getParent();

  //----------------------------------------------------------------------------
  // Global arguments
  std::vector<unsigned> FutureArgId;
  std::vector<unsigned> FieldArgId;
  auto TaskArgsA = createGlobalArguments(TheModule, ArgVs);
  
 
  //----------------------------------------------------------------------------
  // Predicate
  auto PredicateA = createPredicateTrue(TheModule);
  
  //----------------------------------------------------------------------------
  // Launch
 
  auto MapperIdV = llvmValue(TheContext_, MapperIdType_, 0); 
  auto MappingTagIdV = llvmValue(TheContext_, MappingTagIdType_, 0); 
  auto PredicateV = load(PredicateA, TheModule, "predicate");
  auto TaskIdV = llvmValue(TheContext_, TaskIdType_, TaskId);
  
  auto ArgDataPtrV = loadStructMember(TaskArgsA, 0, "args");
  auto ArgSizeV = loadStructMember(TaskArgsA, 1, "arglen");

  auto LauncherRT = reduceStruct(TaskLauncherType_, TheModule);

  std::vector<Value*> LaunchArgVs = {TaskIdV, ArgDataPtrV, ArgSizeV, 
    PredicateV, MapperIdV, MappingTagIdV};
  auto LaunchArgTs = llvmTypes(LaunchArgVs);

  auto LaunchT = FunctionType::get(LauncherRT, LaunchArgTs, false);
  auto LaunchF = TheModule.getOrInsertFunction("legion_task_launcher_create", LaunchT);

  Value* LauncherRV = Builder_.CreateCall(LaunchF, LaunchArgVs, "launcher_create");
  auto LauncherA = createEntryBlockAlloca(TheFunction, TaskLauncherType_, "task_launcher.alloca");
  store(LauncherRV, LauncherA);
  
  //----------------------------------------------------------------------------
  // Add futures
  createGlobalFutures(TheModule, LauncherA, ArgVs, false );
  
  //----------------------------------------------------------------------------
  // Add fields
  createFieldArguments(TheModule, LauncherA, ArgVs );

  
  //----------------------------------------------------------------------------
  // Execute
  
  const auto & LegionE = getCurrentTask();
  const auto & ContextA = LegionE.ContextAlloca;
  const auto & RuntimeA = LegionE.RuntimeAlloca;
  
  LauncherRV = load(LauncherA, TheModule, "task_launcher");
  auto ContextV = load(ContextA, TheModule, "context");
  auto RuntimeV = load(RuntimeA, TheModule, "runtime");

  // args
  std::vector<Value*> ExecArgVs = { RuntimeV, ContextV, LauncherRV };
  auto ExecArgTs = llvmTypes(ExecArgVs);
  auto FutureRT = reduceStruct(FutureType_, TheModule);

  auto ExecT = FunctionType::get(FutureRT, ExecArgTs, false);
  auto ExecF = TheModule.getOrInsertFunction("legion_task_launcher_execute", ExecT);
  
  auto FutureRV = Builder_.CreateCall(ExecF, ExecArgVs, "launcher_exec");
  auto FutureA = createEntryBlockAlloca(TheFunction, FutureType_, "future.alloca");
  store(FutureRV, FutureA);

  //----------------------------------------------------------------------------
  // Destroy launcher
  
  destroyOpaqueType(TheModule, LauncherA, "legion_task_launcher_destroy", "task_launcher");
  
  //----------------------------------------------------------------------------
  // Deallocate storate
  destroyGlobalArguments(TheModule, TaskArgsA);

  return Builder_.CreateLoad(FutureType_, FutureA);
}

//==============================================================================
// Launch an index task
//==============================================================================
Value* LegionTasker::launch(
    Module &TheModule,
    const std::string & Name,
    int TaskId,
    const std::vector<Value*> & ArgAs,
    Value* RangeV,
    bool CleanupPartitions )
{
  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  
  // temporaries
  auto & LegionE = getCurrentTask();
  const auto & ContextA = LegionE.ContextAlloca;
  const auto & RuntimeA = LegionE.RuntimeAlloca;
  auto & PartInfoA = LegionE.PartInfoAlloca;

  if (!PartInfoA) PartInfoA = createPartitionInfo(TheModule);

  //----------------------------------------------------------------------------
  // Global arguments
  std::vector<unsigned> FutureArgId;
  std::vector<unsigned> FieldArgId;
  auto TaskArgsA = createGlobalArguments(TheModule, ArgAs);

  //----------------------------------------------------------------------------
  // Predicate
  auto PredicateA = createPredicateTrue(TheModule);
  
  //----------------------------------------------------------------------------
  // Create domain  
#if 0 
  auto DomainLoA = createEntryBlockAlloca(TheFunction, Point1dType_, "lo");
  auto DomainHiA = createEntryBlockAlloca(TheFunction, Point1dType_, "hi");

  auto StartV = Builder_.CreateExtractValue(RangeV, 0, "start");
  auto ZeroC = Constant::getNullValue(Int32Type_);
  std::vector<Value*> IndicesC = {ZeroC,  ZeroC};
  auto DomainLoGEP = Builder_.CreateGEP(DomainLoA, IndicesC, "lo");
	Builder_.CreateStore(StartV, DomainLoGEP);
  
  auto EndV = Builder_.CreateExtractValue(RangeV, 1, "end");
  auto DomainHiGEP = Builder_.CreateGEP(DomainHiA, IndicesC, "hi");
	Builder_.CreateStore(EndV, DomainHiGEP);

  auto LaunchBoundA = createEntryBlockAlloca(TheFunction, Rect1dType_, "launch_bound");
  
  auto DomainLoV = Builder_.CreateLoad(Point1dType_, DomainLoA);
  storeStructMember(DomainLoV, LaunchBoundA, 0);
  
  auto DomainHiV = Builder_.CreateLoad(Point1dType_, DomainHiA);
  storeStructMember(DomainHiV, LaunchBoundA, 1);

  auto DomainLoRV = load(DomainLoA, TheModule, "lo");
  auto DomainHiRV = load(DomainHiA, TheModule, "hi");
  auto DomainRectA = createEntryBlockAlloca(TheFunction, DomainRectType_, "domain");
  std::vector<Value*> DomainFromArgVs = { DomainRectA, DomainLoRV, DomainHiRV };
  auto DomainFromArgTs = llvmTypes(DomainFromArgVs);

  auto DomainFromT = FunctionType::get(VoidType_, DomainFromArgTs, false);
  auto DomainFromF = TheModule.getFunction("legion_domain_from_rect_1d");
  if (!DomainFromF) {
    DomainFromF = Function::Create(DomainFromT, Function::InternalLinkage,
        "legion_domain_from_rect_1d", &TheModule);
    auto Arg = DomainFromF->arg_begin();
    Arg->addAttr(Attribute::StructRet);
  }
  
  Builder_.CreateCall(DomainFromF, DomainFromArgVs);

#else

  Value* IndexSpaceA = getAsAlloca(Builder_, TheFunction, RangeV);
  
  auto DomainRectA = createEntryBlockAlloca(TheFunction, DomainRectType_, "domain");

  std::vector<Value*> DomainFromArgVs = { RuntimeA, IndexSpaceA, DomainRectA };
  auto DomainFromArgTs = llvmTypes(DomainFromArgVs);

  auto DomainFromT = FunctionType::get(VoidType_, DomainFromArgTs, false);
  auto DomainFromF = TheModule.getOrInsertFunction("contra_legion_domain_create", DomainFromT);
  
  Builder_.CreateCall(DomainFromF, DomainFromArgVs);

#endif
  
  //----------------------------------------------------------------------------
  // argument map 
  
  auto ArgMapA = createOpaqueType(TheModule, ArgMapType_,
      "legion_argument_map_create", "arg_map");
  
  //----------------------------------------------------------------------------
  // Launch
 
  auto MapperIdV = llvmValue(TheContext_, MapperIdType_, 0); 
  auto MappingTagIdV = llvmValue(TheContext_, MappingTagIdType_, 0); 
  auto PredicateV = load(PredicateA, TheModule, "predicate");
  auto TaskIdV = llvmValue(TheContext_, TaskIdType_, TaskId);
  auto MustV = Constant::getNullValue(BoolType_);
  auto ArgMapV = load(ArgMapA, TheModule, "arg_map");
  
  auto ArgDataPtrV = loadStructMember(TaskArgsA, 0, "args");
  auto ArgSizeV = loadStructMember(TaskArgsA, 1, "arglen");

  auto LauncherRT = reduceStruct(IndexLauncherType_, TheModule);

  std::vector<Value*> LaunchArgVs = {TaskIdV, DomainRectA, ArgDataPtrV, ArgSizeV, 
    ArgMapV, PredicateV, MustV, MapperIdV, MappingTagIdV};
  auto LaunchArgTs = llvmTypes(LaunchArgVs);

  auto LaunchT = FunctionType::get(LauncherRT, LaunchArgTs, false);
  auto LaunchF = TheModule.getFunction("legion_index_launcher_create");
  if (!LaunchF) {
    LaunchF = Function::Create(LaunchT, Function::InternalLinkage,
        "legion_index_launcher_create", &TheModule);
    auto Arg = LaunchF->arg_begin();
    ++Arg;
    Arg->addAttr(Attribute::ByVal);
  }

  Value* LauncherRV = Builder_.CreateCall(LaunchF, LaunchArgVs, "launcher_create");
  auto LauncherA = createEntryBlockAlloca(TheFunction, IndexLauncherType_, "task_launcher.alloca");
  store(LauncherRV, LauncherA);
  
  //----------------------------------------------------------------------------
  // Add futures
  createGlobalFutures(TheModule, LauncherA, ArgAs, true );
 
  //----------------------------------------------------------------------------
  // Add fields
  createFieldArguments(
      TheModule,
      LauncherA,
      ArgAs,
      IndexSpaceA,
      PartInfoA);
  
  //----------------------------------------------------------------------------
  // Execute
  
  LauncherRV = load(LauncherA, TheModule, "task_launcher");
  auto ContextV = load(ContextA, TheModule, "context");
  auto RuntimeV = load(RuntimeA, TheModule, "runtime");

  // args
  std::vector<Value*> ExecArgVs = { RuntimeV, ContextV, LauncherRV };
  auto ExecArgTs = llvmTypes(ExecArgVs);
  auto FutureMapRT = reduceStruct(FutureMapType_, TheModule);

  auto ExecT = FunctionType::get(FutureMapRT, ExecArgTs, false);
  auto ExecF = TheModule.getOrInsertFunction("legion_index_launcher_execute", ExecT);
  
  Value* FutureMapRV = Builder_.CreateCall(ExecF, ExecArgVs, "launcher_exec");
  auto FutureMapA = createEntryBlockAlloca(TheFunction, FutureMapType_, "future_map.alloca");
  store(FutureMapRV, FutureMapA);
  
	//----------------------------------------------------------------------------
  // Destroy argument map
  
  destroyOpaqueType(TheModule, ArgMapA, "legion_argument_map_destroy", "arg_map");

	//----------------------------------------------------------------------------
  // Destroy future map

  auto WaitT = FunctionType::get(VoidType_, FutureMapRT, false);
  auto WaitF = TheModule.getOrInsertFunction("legion_future_map_wait_all_results", WaitT);
  Builder_.CreateCall(WaitF, FutureMapRV);
  
  destroyOpaqueType(TheModule, FutureMapA, "legion_future_map_destroy", "future_map");
  
  //----------------------------------------------------------------------------
  // cleanup
  
  destroyOpaqueType(TheModule, LauncherA, "legion_index_launcher_destroy", "task_launcher");
  
  if (CleanupPartitions) destroyPartitionInfo(TheModule);
  
  destroyGlobalArguments(TheModule, TaskArgsA);

  //return Builder_.CreateLoad(FutureMapType_, FutureMapA);
  return nullptr;
}


//==============================================================================
// get a future value
//==============================================================================
Value* LegionTasker::loadFuture(
    Module &TheModule,
    Value* FutureA,
    Type *DataT)
{
  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  
  // args
  auto FutureRV = load(FutureA, TheModule, "future");
  auto FutureRT = reduceStruct(FutureType_, TheModule);
  auto GetFutureT = FunctionType::get(VoidPtrType_, FutureRT, false);
  auto GetFutureF = TheModule.getOrInsertFunction("legion_future_get_untyped_pointer",
      GetFutureT);

  Value* DataPtrV = Builder_.CreateCall(GetFutureF, FutureRV, "future");
  auto DataPtrA = createEntryBlockAlloca(TheFunction, VoidPtrType_, "future.alloca");
  Builder_.CreateStore(DataPtrV, DataPtrA);

  auto DataA = createEntryBlockAlloca(TheFunction, DataT);
  deserialize(DataA, DataPtrA);

  return Builder_.CreateLoad(DataT, DataA, "future");
}

//==============================================================================
// insert a future value
//==============================================================================
AllocaInst* LegionTasker::createFuture(
    Module &,
    Function* TheFunction,
    const std::string & Name)
{
  auto FutureA = createEntryBlockAlloca(TheFunction, FutureType_, "future.alloca");
  return FutureA;
}

//==============================================================================
// destroey a future value
//==============================================================================
void LegionTasker::destroyFuture(Module &TheModule, Value* FutureA)
{
  destroyOpaqueType(TheModule, FutureA, "legion_future_destroy", "future");
}
  
//==============================================================================
// copy a value into a future
//==============================================================================
void LegionTasker::toFuture(
    Module & TheModule,
    Value* ValueV,
    Value* FutureA)
{
  // load runtime
  const auto & LegionE = getCurrentTask();
  const auto & RuntimeA = LegionE.RuntimeAlloca;
  auto RuntimeV = load(RuntimeA, TheModule, "runtime");

  auto ValueT = ValueV->getType();
  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  auto ValueA = createEntryBlockAlloca(TheFunction, ValueT);
  Builder_.CreateStore( ValueV, ValueA );

  auto TheBlock = Builder_.GetInsertBlock();
  auto ValuePtrV = CastInst::Create(Instruction::BitCast, ValueA, VoidPtrType_,
      "cast", TheBlock);

  auto ValueSizeV = getTypeSize<size_t>(Builder_, ValueT);

  std::vector<Value*> FunArgVs = {RuntimeV, ValuePtrV, ValueSizeV};
  auto FunArgTs = llvmTypes(FunArgVs);
    
  auto FutureRT = reduceStruct(FutureType_, TheModule);

  auto FunT = FunctionType::get(FutureRT, FunArgTs, false);
  auto FunF = TheModule.getOrInsertFunction("legion_future_from_untyped_pointer", FunT);
    
  auto FutureRV = Builder_.CreateCall(FunF, FunArgVs, "future");
  store(FutureRV, FutureA);

}

//==============================================================================
// copy a future
//==============================================================================
void LegionTasker::copyFuture(
    Module & TheModule,
    Value* ValueV,
    Value* FutureA)
{
  // load runtime
  auto FutureRT = reduceStruct(FutureType_, TheModule);

  auto FunT = FunctionType::get(FutureRT, FutureRT, false);
  auto FunF = TheModule.getOrInsertFunction("legion_future_copy", FunT);
  auto ValueRV = Builder_.CreateExtractValue(ValueV, 0);
  auto FutureRV = Builder_.CreateCall(FunF, ValueRV, "future");
  store(FutureRV, FutureA);
}

//==============================================================================
// Is this a future type
//==============================================================================
bool LegionTasker::isFuture(Value* FutureA) const
{
  auto FutureT = FutureA->getType();
  if (isa<AllocaInst>(FutureA)) FutureT = FutureT->getPointerElementType();
  return (FutureT == FutureType_);
}

//==============================================================================
// Is this a field type
//==============================================================================
bool LegionTasker::isField(Value* FieldA) const
{
  auto FieldT = FieldA->getType();
  if (isa<AllocaInst>(FieldA)) FieldT = FieldT->getPointerElementType();
  return (FieldT == FieldDataType_);
}


//==============================================================================
// Create a legion field
//==============================================================================
AllocaInst* LegionTasker::createField(
    Module & TheModule,
    Function* TheFunction,
    const std::string & VarN,
    Type* VarT,
    Value* RangeV,
    Value* VarV)
{
  auto FieldA = createEntryBlockAlloca(TheFunction, FieldDataType_, "field");
  
  const auto & LegionE = getCurrentTask();
  const auto & ContextA = LegionE.ContextAlloca;
  const auto & RuntimeA = LegionE.RuntimeAlloca;
  
  auto NameV = llvmString(TheContext_, TheModule, VarN);

  Value* DataSizeV;
  if (VarV) {
    DataSizeV = getTypeSize<size_t>(Builder_, VarT);
    VarV = getAsAlloca(Builder_, TheFunction, VarV);
  }
  else {
    DataSizeV = llvmValue<size_t>(TheContext_, 0);
    VarV = Constant::getNullValue(VoidPtrType_);
  }
    
  Value* IndexSpaceA = getAsAlloca(Builder_, TheFunction, RangeV);
  
  std::vector<Value*> FunArgVs = {
    RuntimeA,
    ContextA,
    NameV,
    DataSizeV, 
    VarV,
    IndexSpaceA,
    FieldA};
  auto FunArgTs = llvmTypes(FunArgVs);
    
  auto FunT = FunctionType::get(VoidType_, FunArgTs, false);
  
  std::string FunN = isRange(RangeV) ?
    "contra_legion_field_create" : "contra_legion_field_create_from_partition";
  
  auto FunF = TheModule.getOrInsertFunction(FunN, FunT);
  Builder_.CreateCall(FunF, FunArgVs);
    
  return FieldA;
}

//==============================================================================
// destroey a field
//==============================================================================
void LegionTasker::destroyField(Module &TheModule, Value* FieldA)
{
  const auto & LegionE = getCurrentTask();
  const auto & ContextA = LegionE.ContextAlloca;
  const auto & RuntimeA = LegionE.RuntimeAlloca;

  std::vector<Value*> FunArgVs = {RuntimeA, ContextA, FieldA};
  auto FunArgTs = llvmTypes(FunArgVs);
    
  auto FunT = FunctionType::get(VoidType_, FunArgTs, false);
  auto FunF = TheModule.getOrInsertFunction("contra_legion_field_destroy", FunT);
    
  Builder_.CreateCall(FunF, FunArgVs);
}

//==============================================================================
// Is this an range type
//==============================================================================
bool LegionTasker::isRange(Type* RangeT) const
{
  return (RangeT == IndexSpaceDataType_);
}

bool LegionTasker::isRange(Value* RangeA) const
{
  auto RangeT = RangeA->getType();
  if (isa<AllocaInst>(RangeA)) RangeT = RangeT->getPointerElementType();
  return isRange(RangeT);
}


//==============================================================================
// create a range
//==============================================================================
AllocaInst* LegionTasker::createRange(
    Module & TheModule,
    Function* TheFunction,
    const std::string & Name,
    Value* StartV,
    Value* EndV)
{
  auto IndexSpaceA = createEntryBlockAlloca(TheFunction, IndexSpaceDataType_, "index");

  if (isInsideTask()) {
    const auto & LegionE = getCurrentTask();
    const auto & ContextA = LegionE.ContextAlloca;
    const auto & RuntimeA = LegionE.RuntimeAlloca;
    
    auto NameV = llvmString(TheContext_, TheModule, Name);

    std::vector<Value*> FunArgVs = {RuntimeA, ContextA, NameV, StartV, EndV, IndexSpaceA};
    auto FunArgTs = llvmTypes(FunArgVs);
      
    auto FunT = FunctionType::get(VoidType_, FunArgTs, false);
    auto FunF = TheModule.getOrInsertFunction("contra_legion_index_space_create", FunT);
      
    Builder_.CreateCall(FunF, FunArgVs);
  }
  else { 
    storeStructMember(StartV, IndexSpaceA, 0, "start");
    auto OneC = llvmValue<int_t>(TheContext_, 1);
    EndV = Builder_.CreateAdd(EndV, OneC);
    storeStructMember(EndV, IndexSpaceA, 1, "end");
  }

  
  return IndexSpaceA;

}

//==============================================================================
// create a range
//==============================================================================
AllocaInst* LegionTasker::createRange(
    Module & TheModule,
    Function* TheFunction,
    Value* ValueV,
    const std::string & Name)
{
  auto IndexSpaceA = createEntryBlockAlloca(TheFunction, IndexSpaceDataType_, "index");

  if (isInsideTask()) {
    const auto & LegionE = getCurrentTask();
    const auto & ContextA = LegionE.ContextAlloca;
    const auto & RuntimeA = LegionE.RuntimeAlloca;
    
    std::vector<Value*> FunArgVs = {RuntimeA, ContextA, ValueV, IndexSpaceA};
    auto FunArgTs = llvmTypes(FunArgVs);
      
    auto FunT = FunctionType::get(VoidType_, FunArgTs, false);
    auto FunF = TheModule.getOrInsertFunction(
        "contra_legion_index_space_create_from_size", FunT);
      
    Builder_.CreateCall(FunF, FunArgVs);
  }
  else { 
    auto ZeroC = llvmValue<int_t>(TheContext_, 0);
    storeStructMember(ZeroC, IndexSpaceA, 0, "start");
    storeStructMember(ValueV, IndexSpaceA, 1, "end");
  }

  
  return IndexSpaceA;

}

//==============================================================================
// create a range
//==============================================================================
AllocaInst* LegionTasker::createRange(
    Module & TheModule,
    Function* TheFunction,
    Type*,
    Value* ValueV,
    const std::string & Name)
{
  auto IndexSpaceA = createEntryBlockAlloca(TheFunction, IndexSpaceDataType_, "index");
  auto ValueA = getAsAlloca(Builder_, TheFunction, ValueV);

  if (isInsideTask()) {
    const auto & LegionE = getCurrentTask();
    const auto & ContextA = LegionE.ContextAlloca;
    const auto & RuntimeA = LegionE.RuntimeAlloca;
    
    std::vector<Value*> FunArgVs = {RuntimeA, ContextA, ValueA, IndexSpaceA};
    auto FunArgTs = llvmTypes(FunArgVs);
      
    auto FunT = FunctionType::get(VoidType_, FunArgTs, false);
    auto FunF = TheModule.getOrInsertFunction(
        "contra_legion_index_space_create_from_array", FunT);
      
    Builder_.CreateCall(FunF, FunArgVs);
  }
  else { 
    
    auto ValueT = ValueA->getType();
    auto IntT = llvmType<int_t>(TheContext_);
    auto FunT = FunctionType::get(IntT, ValueT, false);
    auto FunF = TheModule.getOrInsertFunction("contra_legion_sum_array", FunT);
    auto SumV = Builder_.CreateCall(FunF, ValueA);
    
    auto ZeroC = llvmValue<int_t>(TheContext_, 0);
    storeStructMember(ZeroC, IndexSpaceA, 0, "start");
    storeStructMember(SumV, IndexSpaceA, 1, "end");
  }

  
  return IndexSpaceA;

}

//==============================================================================
// create a range
//==============================================================================
AllocaInst* LegionTasker::partition(
    Module & TheModule,
    Function* TheFunction,
    Value* IndexSpaceA,
    Value* ValueA)
{
  auto & LegionE = getCurrentTask();
  const auto & ContextA = LegionE.ContextAlloca;
  const auto & RuntimeA = LegionE.RuntimeAlloca;
  auto & PartInfoA = LegionE.PartInfoAlloca;

  if (!PartInfoA) PartInfoA = createPartitionInfo(TheModule);
  
  IndexSpaceA = getAsAlloca(Builder_, TheFunction, IndexSpaceA);
  
  auto IndexPartA = createEntryBlockAlloca(TheFunction, IndexPartitionType_);

  //------------------------------------
  if (isRange(ValueA)) {
    ValueA = getAsAlloca(Builder_, TheFunction, ValueA);

    std::vector<Value*> FunArgVs = {
      RuntimeA,
      ContextA,
      ValueA,
      IndexSpaceA,
      PartInfoA,
      IndexPartA};
    auto FunArgTs = llvmTypes(FunArgVs);

    auto FunT = FunctionType::get(VoidType_, FunArgTs, false);
    auto FunF = TheModule.getOrInsertFunction(
        "contra_legion_index_space_partition", FunT);
  
    Builder_.CreateCall(FunF, FunArgVs);
  }
  //------------------------------------
  else if (isField(ValueA)) {
    std::vector<Value*> FunArgVs = {
      RuntimeA,
      ContextA,
      ValueA,
      IndexSpaceA,
      PartInfoA,
      IndexPartA};
    auto FunArgTs = llvmTypes(FunArgVs);

    auto FunT = FunctionType::get(VoidType_, FunArgTs, false);
    auto FunF = TheModule.getOrInsertFunction(
        "contra_legion_index_space_partition_from_field", FunT);
  
    Builder_.CreateCall(FunF, FunArgVs);
  }
  //------------------------------------
  else {
    auto ValueV = getAsValue(Builder_, ValueA);

    std::vector<Value*> FunArgVs = {
      RuntimeA,
      ContextA,
      ValueV,
      IndexSpaceA,
      PartInfoA,
      IndexPartA};
    auto FunArgTs = llvmTypes(FunArgVs);

    auto FunT = FunctionType::get(VoidType_, FunArgTs, false);
    auto FunF = TheModule.getOrInsertFunction(
        "contra_legion_index_space_partition_from_size", FunT);
  
    Builder_.CreateCall(FunF, FunArgVs);
  }
  //------------------------------------

  return IndexPartA;
    
}

//==============================================================================
// create a range
//==============================================================================
AllocaInst* LegionTasker::partition(
    Module & TheModule,
    Function* TheFunction,
    Value* IndexSpaceA,
    Type*,
    Value* ValueV,
    bool ReportSizeError)
{

  auto & LegionE = getCurrentTask();
  const auto & ContextA = LegionE.ContextAlloca;
  const auto & RuntimeA = LegionE.RuntimeAlloca;
  auto & PartInfoA = LegionE.PartInfoAlloca;

  if (!PartInfoA) PartInfoA = createPartitionInfo(TheModule);

  auto IndexPartA = createEntryBlockAlloca(TheFunction, IndexPartitionType_);
    
  IndexSpaceA = getAsAlloca(Builder_, TheFunction, IndexSpaceA);
  auto ValueA = getAsAlloca(Builder_, TheFunction, ValueV);
  
  std::vector<Value*> FunArgVs = {
    RuntimeA,
    ContextA,
    ValueA,
    IndexSpaceA,
    PartInfoA,
    IndexPartA,
    llvmValue<bool>(TheContext_, ReportSizeError)
  };
  auto FunArgTs = llvmTypes(FunArgVs);
    
  auto FunT = FunctionType::get(VoidType_, FunArgTs, false);
  auto FunF = TheModule.getOrInsertFunction(
      "contra_legion_index_space_partition_from_array", FunT);
    
  Builder_.CreateCall(FunF, FunArgVs);
  
  return IndexPartA;

}


//==============================================================================
// destroey a field
//==============================================================================
void LegionTasker::destroyRange(Module &TheModule, Value* RangeV)
{
  if (isInsideTask()) {
    const auto & LegionE = getCurrentTask();
    const auto & ContextA = LegionE.ContextAlloca;
    const auto & RuntimeA = LegionE.RuntimeAlloca;

    auto TheFunction = Builder_.GetInsertBlock()->getParent();
    Value* IndexSpaceA = getAsAlloca(Builder_, TheFunction, RangeV);
    
    std::vector<Value*> FunArgVs = {RuntimeA, ContextA, IndexSpaceA};
    auto FunArgTs = llvmTypes(FunArgVs);
      
    auto FunT = FunctionType::get(VoidType_, FunArgTs, false);
    auto FunF = TheModule.getOrInsertFunction("contra_legion_index_space_destroy", FunT);
      
    Builder_.CreateCall(FunF, FunArgVs);
  }
}

//==============================================================================
// get a range start
//==============================================================================
llvm::Value* LegionTasker::getRangeStart(Module &TheModule, Value* RangeV)
{
  return Builder_.CreateExtractValue(RangeV, 0);
}

//==============================================================================
// get a range start
//==============================================================================
llvm::Value* LegionTasker::getRangeEnd(Module &TheModule, Value* RangeV)
{
  Value* EndV = Builder_.CreateExtractValue(RangeV, 1);
  auto OneC = llvmValue<int_t>(TheContext_, 1);
  return Builder_.CreateSub(EndV, OneC);
}


//==============================================================================
// get a range size
//==============================================================================
llvm::Value* LegionTasker::getRangeSize(Module &TheModule, Value* RangeV)
{
  auto StartV = Builder_.CreateExtractValue(RangeV, 0);
  auto EndV = Builder_.CreateExtractValue(RangeV, 1);
  return Builder_.CreateSub(EndV, StartV);
}

//==============================================================================
// get a range value
//==============================================================================
llvm::Value* LegionTasker::loadRangeValue(
    Module &TheModule,
    Type* ElementT,
    Value* RangeA,
    Value* IndexV)
{
  auto ZeroC = ConstantInt::get(TheContext_, APInt(32, 0, true));
  Value* StartV = Builder_.CreateGEP(RangeA, {ZeroC, ZeroC});
  StartV = Builder_.CreateLoad(llvmType<int_t>(TheContext_), StartV);
  return Builder_.CreateAdd(StartV, IndexV);
}


//==============================================================================
// Is this an accessor type
//==============================================================================
bool LegionTasker::isAccessor(Type* AccessorT) const
{
  return (AccessorT == AccessorDataType_);
}

bool LegionTasker::isAccessor(Value* AccessorA) const
{
  auto AccessorT = AccessorA->getType();
  if (isa<AllocaInst>(AccessorA)) AccessorT = AccessorT->getPointerElementType();
  return isAccessor(AccessorT);
}

//==============================================================================
// Store a value into an accessor
//==============================================================================
void LegionTasker::storeAccessor(
    Module & TheModule,
    Value* ValueV,
    Value* AccessorV,
    Value* IndexV) const
{
  auto TheFunction = Builder_.GetInsertBlock()->getParent();

  auto ValueA = getAsAlloca(Builder_, TheFunction, ValueV);
  auto ValueT = ValueA->getAllocatedType();

  Value* AccessorA = getAsAlloca(Builder_, TheFunction, AccessorV);
    
  auto DataSizeV = getTypeSize<size_t>(Builder_, ValueT);
  
  std::vector<Value*> FunArgVs = { AccessorA, ValueA, DataSizeV };
  
  if (IndexV) {
    FunArgVs.emplace_back( getAsValue(Builder_, IndexV) );
  }
  else {
    FunArgVs.emplace_back( llvmValue<int_t>(TheContext_, 0) );
  }
  
  auto FunArgTs = llvmTypes(FunArgVs);
    
  auto FunT = FunctionType::get(VoidType_, FunArgTs, false);
  auto FunF = TheModule.getOrInsertFunction("contra_legion_accessor_write", FunT);

  Builder_.CreateCall(FunF, FunArgVs);
}

//==============================================================================
// Load a value from an accessor
//==============================================================================
Value* LegionTasker::loadAccessor(
    Module & TheModule, 
    Type * ValueT,
    Value* AccessorV,
    Value* IndexV) const
{
  auto TheFunction = Builder_.GetInsertBlock()->getParent();

  auto AccessorA = getAsAlloca(Builder_, TheFunction, AccessorDataType_, AccessorV);
    
  auto ValueA = createEntryBlockAlloca(TheFunction, ValueT);
  auto DataSizeV = getTypeSize<size_t>(Builder_, ValueT);

  std::vector<Value*> FunArgVs = { AccessorA, ValueA, DataSizeV };
  
  if (IndexV) {
    FunArgVs.emplace_back( getAsValue(Builder_, IndexV) );
  }
  else {
    FunArgVs.emplace_back( llvmValue<int_t>(TheContext_, 0) );
  }

  auto FunArgTs = llvmTypes(FunArgVs);
    
  auto FunT = FunctionType::get(VoidType_, FunArgTs, false);
  auto FunF = TheModule.getOrInsertFunction("contra_legion_accessor_read", FunT);

  Builder_.CreateCall(FunF, FunArgVs);

  return Builder_.CreateLoad(ValueT, ValueA);
}

//==============================================================================
// destroey an accessor
//==============================================================================
void LegionTasker::destroyAccessor(
    Module &TheModule,
    Value* AccessorA)
{
  const auto & LegionE = getCurrentTask();
  const auto & ContextA = LegionE.ContextAlloca;
  const auto & RuntimeA = LegionE.RuntimeAlloca;

  std::vector<Value*> FunArgVs = {RuntimeA, ContextA, AccessorA};
  auto FunArgTs = llvmTypes(FunArgVs);
    
  auto FunT = FunctionType::get(VoidType_, FunArgTs, false);
  auto FunF = TheModule.getOrInsertFunction("contra_legion_accessor_destroy", FunT);
    
  Builder_.CreateCall(FunF, FunArgVs);
}

//==============================================================================
// Is this an range type
//==============================================================================
bool LegionTasker::isPartition(Type* PartT) const
{
  return (PartT == IndexPartitionType_);
}

bool LegionTasker::isPartition(Value* PartA) const
{
  auto PartT = PartA->getType();
  if (isa<AllocaInst>(PartA)) PartT = PartT->getPointerElementType();
  return isPartition(PartT);
}

//==============================================================================
// Make a point with a value
//==============================================================================
Value* LegionTasker::makePoint(std::intmax_t val) const
{
  auto ValC = llvmValue<legion_coord_t>(TheContext_, val);
  return ConstantArray::get(Point1dType_, ValC);
}

Value* LegionTasker::makePoint(Value* ValV) const
{
  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  auto PointA = createEntryBlockAlloca(TheFunction, Point1dType_);
  auto ZeroC = Constant::getNullValue(Int32Type_);
  std::vector<Value*> IndicesC = {ZeroC,  ZeroC};
  auto PointGEP = Builder_.CreateGEP(PointA, IndicesC);
  ValV = getAsValue(Builder_, ValV);
	Builder_.CreateStore(ValV, PointGEP);
  return PointA; 
}


}
