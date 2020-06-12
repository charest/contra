#include "config.hpp"

#include "codegen.hpp"
#include "errors.hpp"
#include "legion.hpp"
#include "utils/llvm_utils.hpp"
#include "librt/dopevector.hpp"

#include <unordered_map>
#include <vector>

#if defined( _WIN32 )
#include <Windows.h>
#else
#include <sys/time.h>
#endif


////////////////////////////////////////////////////////////////////////////////
// Legion runtime
////////////////////////////////////////////////////////////////////////////////


namespace contra {

}

extern "C" {

struct contra_legion_index_space_t {
  int_t start;
  int_t end;
  int_t step;
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
  
  struct IndexPartitionDeleter {
    legion_runtime_t * runtime;
    legion_context_t * context;
    IndexPartitionDeleter(legion_runtime_t * rt, legion_context_t * ctx) :
      runtime(rt), context(ctx)
    {}
    void operator()(legion_index_partition_t *ptr)
    {
      legion_index_partition_destroy(*runtime, *context, *ptr);
      delete ptr;
    }
  };
  
  struct LogicalPartitionDeleter {
    legion_runtime_t * runtime;
    legion_context_t * context;
    LogicalPartitionDeleter(legion_runtime_t * rt, legion_context_t * ctx) :
      runtime(rt), context(ctx)
    {}
    void operator()(legion_logical_partition_t *ptr)
    {
      //legion_logical_partition_destroy(*runtime, *context, *ptr);
      delete ptr;
    }
  };

  using IndexPartitionVal = std::unique_ptr<legion_index_partition_t, IndexPartitionDeleter>;
  using IndexPartitionMap = std::unordered_map< legion_index_space_id_t, IndexPartitionVal >;
  std::forward_list< IndexPartitionMap > IndexPartitions;
  
  using LogicalPartitionKey = std::pair<legion_field_id_t, legion_index_partition_id_t>;
  using LogicalPartitionVal = std::unique_ptr<legion_logical_partition_t, LogicalPartitionDeleter>;

  struct LogicalPartitionHash {
    std::size_t operator () (const LogicalPartitionKey &p) const
    {
      auto a = static_cast<uint64_t>(p.first);
      auto b = static_cast<uint32_t>(p.second);
      return ((a<<32) | (b));
    }
  };

  using LogicalPartitionMap = 
    std::unordered_map< LogicalPartitionKey, LogicalPartitionVal, LogicalPartitionHash >;
  
  LogicalPartitionMap LogicalPartitions;

  auto createIndexPartition(
      legion_runtime_t *rt,
      legion_context_t *ctx,
      legion_index_space_id_t id)
  {
    // found
    IndexPartitionDeleter Deleter(rt, ctx);
    for (auto & Scope : IndexPartitions) {
      auto it = Scope.find(id);
      if (it != Scope.end()) {
        it->second = IndexPartitionVal(new legion_index_partition_t, Deleter);
        return it->second.get();
      }
    }
    // not found
    auto res = IndexPartitions.front().emplace(
        id,
        IndexPartitionVal(new legion_index_partition_t, Deleter) );
    return res.first->second.get();
  }
  
  std::pair<legion_index_partition_t*, bool>
    getOrCreateIndexPartition(
      legion_runtime_t *rt,
      legion_context_t *ctx,
      legion_index_space_id_t id)
  {
    // found
    for (const auto & Scope : IndexPartitions) {
      auto it = Scope.find(id);
      if (it!=Scope.end()) {
        return std::make_pair(it->second.get(), true);
      }
    }
    // not found
    IndexPartitionDeleter Deleter(rt, ctx);
    auto res = IndexPartitions.front().emplace(
        id,
        IndexPartitionVal(new legion_index_partition_t, Deleter) );
    return std::make_pair(res.first->second.get(), false);
  }
  
  legion_index_partition_t* getIndexPartition(legion_index_space_id_t id)
  { 
    for (const auto & Scope : IndexPartitions) {
      auto it = Scope.find(id);
      if (it!=Scope.end()) return it->second.get();
    }
    return nullptr;
  }

  auto createLogicalPartition(
      legion_runtime_t *rt,
      legion_context_t *ctx,
      legion_field_id_t fid,
      legion_index_partition_id_t pid)
  {
    // found 
    LogicalPartitionDeleter Deleter(rt, ctx);
    auto it = LogicalPartitions.find(std::make_pair(fid,pid));
    if (it != LogicalPartitions.end()) {
      it->second = LogicalPartitionVal(new legion_logical_partition_t, Deleter);
      return it->second.get();
    }
    // not found
    auto res = LogicalPartitions.emplace(
        std::make_pair(fid, pid),
        LogicalPartitionVal(new legion_logical_partition_t, Deleter) );
    return res.first->second.get();
  }
  
  std::pair<legion_logical_partition_t*, bool>
    getOrCreateLogicalPartition(
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
  
  
  legion_logical_partition_t* getLogicalPartition(
      legion_field_id_t fid,
      legion_index_partition_id_t pid)
  { 
    auto it = LogicalPartitions.find(std::make_pair(fid,pid));
    if (it!=LogicalPartitions.end()) return it->second.get();
    return nullptr;
  }

  void push() 
  {
    IndexPartitions.push_front({});
  }

  void pop()
  {
    IndexPartitions.pop_front();
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
  //auto index_part = (*parts)->createIndexPartition(runtime, ctx, is->index_space.id);
 
  *part = legion_index_partition_create_equal(
      *runtime,
      *ctx,
      is->index_space,
      cs->index_space,
      /* granularity */ 1,
      /*color*/ AUTO_GENERATE_ID );
  
  //*part = *index_part;
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
  legion_index_space_t color_space = legion_index_space_create(*runtime, *ctx, size);

  //auto index_part = (*parts)->createIndexPartition(runtime, ctx, is->index_space.id);
  
  *part = legion_index_partition_create_equal(
      *runtime,
      *ctx,
      is->index_space,
      color_space,
      /* granularity */ 1,
      /*color*/ AUTO_GENERATE_ID );

  legion_index_space_destroy(*runtime, *ctx, color_space);

  //*part = *index_part;
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

  //auto index_part = (*parts)->createIndexPartition(runtime, ctx, is->index_space.id);

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

  //*part = *index_part;
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

  //legion_index_partition_destroy(*runtime, *ctx, *index_part);
    
  // partition with results
  *index_part = legion_index_partition_create_by_image(
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
  is->step = 1;

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
  is->step = 1;
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
  is.step = 1;

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
    legion_index_partition_t * specified_part,
    contra_legion_field_t * field)
{
  auto parts = reinterpret_cast<contra_legion_partitions_t**>(void_parts);

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
/// partition push
//==============================================================================
void contra_legion_partitions_push(
    contra_legion_partitions_t ** parts)
{
  if (!*parts) *parts = new contra_legion_partitions_t;
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
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    contra_legion_partitions_t ** parts)
{
  //for (auto & part : (*parts)->LogicalPartitions) part.second.reset();
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
  is->step = 1;

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
  is->step = 1;
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
#if defined( _WIN32 )
  
  // use windows api
  LARGE_INTEGER time,freq;
  if (!QueryPerformanceFrequency(&freq)) 
    THROW_CONTRA_ERROR( "Error getting clock frequency." );
  if (!QueryPerformanceCounter(&time))
    THROW_CONTRA_ERROR( "Error getting wall time." );
  return (real_t)time.QuadPart / freq.QuadPart;

#else
  
  // Use system time call
  struct timeval tm;
  if (gettimeofday( &tm, 0 )) THROW_CONTRA_ERROR( "Error getting wall time." );
  return (real_t)tm.tv_sec + (real_t)tm.tv_usec * 1.e-6;

#endif
}

void contra_legion_timer_start(real_t * time)
{
  *time = get_wall_time() * 1e3;
}

void contra_legion_timer_stop(real_t * time)
{
  //std::cout << get_wall_time()*1e3 - *time << std::endl;
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
LegionTasker::LegionTasker(utils::BuilderHelper & TheHelper)
  : AbstractTasker(TheHelper)
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
  std::vector<Type*> members = { IntT, IntT, IntT, IndexSpaceType_ };
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
  auto PredicateRV = TheHelper_.callFunction(
      TheModule,
      "legion_predicate_true",
      PredicateRT,
      {},
      "pred_true");
  
  auto PredicateA = TheHelper_.createEntryBlockAlloca(PredicateType_, "predicate.alloca");
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
  auto TaskArgsA = TheHelper_.createEntryBlockAlloca(TaskArgsType_, "args.alloca");

  //----------------------------------------------------------------------------
  // Identify futures
  
  auto NumArgs = ArgVorAs.size();
  
  std::vector<char> ArgEnums(NumArgs);
  std::vector<unsigned> ValueArgId;
  std::vector<unsigned> FieldArgId;
  
  auto ArgSizesT = ArrayType::get(SizeType_, NumArgs);
  auto ArgSizesA = TheHelper_.createEntryBlockAlloca(ArgSizesT);


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
    ArgEnums[i] = static_cast<char>(ArgEnum);
  
    auto ArgSizeV = getSerializedSize(ArgVorA, SizeType_);
    TheHelper_.insertValue(ArgSizesA, ArgSizeV, i);
  }

  //----------------------------------------------------------------------------
  // First count sizes

  // add 1 byte for each argument first
  auto ArgSizeT = TaskArgsType_->getElementType(1);
  TheHelper_.insertValue( TaskArgsA, llvmValue(TheContext_, ArgSizeT, NumArgs), 1);

  // count user argument sizes
  for (auto i : ValueArgId) {
    auto ArgSizeGEP = TheHelper_.getElementPointer(TaskArgsA, 1);
    auto ArgSizeV = TheHelper_.extractValue(ArgSizesA, i);
    TheHelper_.increment(ArgSizeGEP, ArgSizeV, "addoffset");
  }
  
  // add 8 bytes for each field argument
  auto NumFieldArgs = FieldArgId.size();
  auto ArgSizeGEP = TheHelper_.getElementPointer(TaskArgsA, 1);
  TheHelper_.increment(ArgSizeGEP, NumFieldArgs*8, "addoffset");

  //----------------------------------------------------------------------------
  // Allocate storate
 
  auto ArgSizeV = TheHelper_.extractValue(TaskArgsA, 1);
  auto MallocI = TheHelper_.createMalloc(ByteType_, ArgSizeV, "args");
  TheHelper_.insertValue(TaskArgsA, MallocI, 0);
  
  //----------------------------------------------------------------------------
  // create an array with booleans identifying argyment type
  
  auto ArrayGEP = llvmArray(TheContext_, TheModule, ArgEnums);
  auto ArgDataPtrV = TheHelper_.extractValue(TaskArgsA, 0);
  Builder_.CreateMemCpy(ArgDataPtrV, 1, ArrayGEP, 1, llvmValue<size_t>(TheContext_, NumArgs)); 
 
  //----------------------------------------------------------------------------
  // Copy args

  // add 1 byte for each argument first
  TheHelper_.insertValue(TaskArgsA, llvmValue(TheContext_, ArgSizeT, NumArgs), 1);
  
  for (auto i : ValueArgId) {
    auto ArgV = TheHelper_.getAsAlloca(ArgVorAs[i]);
    // load offset
    ArgSizeV = TheHelper_.extractValue(TaskArgsA, 1);
    // copy
    auto ArgDataPtrV = TheHelper_.extractValue(TaskArgsA, 0);
    serialize(ArgV, ArgDataPtrV, ArgSizeGEP);
    // increment
    ArgSizeGEP = TheHelper_.getElementPointer(TaskArgsA, 1);
    auto ArgSizeV = TheHelper_.extractValue(ArgSizesA, i);
    TheHelper_.increment(ArgSizeGEP, ArgSizeV, "addoffset");
  }
  
  //----------------------------------------------------------------------------
  // Add field identifiers
    
  auto FieldDataPtrT = FieldDataType_->getPointerTo();
  auto UnsignedT = llvmType<unsigned>(TheContext_);
  auto FieldDataF = TheHelper_.createFunction(
      TheModule,
      "contra_legion_pack_field_data",
      VoidType_,
      {FieldDataPtrT, UnsignedT, VoidPtrType_});

  unsigned regidx = 0;
  for (auto i : FieldArgId) {
    Value* ArgV = ArgVorAs[i];
    // load offset
    ArgSizeV = TheHelper_.extractValue(TaskArgsA, 1);
    // offset data pointer
    auto ArgDataPtrV = TheHelper_.extractValue(TaskArgsA, 0);
    auto OffsetArgDataPtrV = TheHelper_.offsetPointer(ArgDataPtrV, ArgSizeV);
    // pack field info
    auto ArgSizeGEP = TheHelper_.getElementPointer(TaskArgsA, 1);
    std::vector<Value*> ArgVs = {
      ArgV,
      llvmValue(TheContext_, UnsignedT, regidx++),
      OffsetArgDataPtrV };
    Builder_.CreateCall(FieldDataF, ArgVs);
    // increment
    TheHelper_.increment(ArgSizeGEP, 8, "addoffset");
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
  auto FutureRT = reduceStruct(FutureType_, TheModule);

  StructType* LauncherT = IsIndex ? IndexLauncherType_ : TaskLauncherType_;
  auto LauncherRT = reduceStruct(LauncherT, TheModule);

  std::vector<Type*> AddFutureArgTs = {LauncherRT, FutureRT};

  std::string FunN = IsIndex ?
    "legion_index_launcher_add_future" : "legion_task_launcher_add_future";
  auto AddFutureF = TheHelper_.createFunction(
      TheModule,
      FunN,
      VoidType_,
      AddFutureArgTs);

  auto NumArgs = ArgVorAs.size();
  for (unsigned i=0; i<NumArgs; i++) {
    auto FutureV = ArgVorAs[i];
    if (!isFuture(FutureV)) continue;
    FutureV = TheHelper_.getAsAlloca(FutureV);
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
  auto PartInfoA = TheHelper_.createEntryBlockAlloca(VoidPtrType_, "indexpartinfo");
  auto NullV = Constant::getNullValue(VoidPtrType_);
  Builder_.CreateStore(NullV, PartInfoA);

  pushPartitionInfo(TheModule, PartInfoA);
    
  return PartInfoA;
}

//==============================================================================
// Create the partition data
//==============================================================================
void LegionTasker::pushPartitionInfo(
    Module & TheModule,
    AllocaInst* PartInfoA)
{
  TheHelper_.callFunction(
      TheModule,
      "contra_legion_partitions_push",
      VoidType_,
      {PartInfoA});
}

void LegionTasker::popPartitionInfo(
    Module & TheModule,
    AllocaInst* PartInfoA)
{
  TheHelper_.callFunction(
      TheModule,
      "contra_legion_partitions_pop",
      VoidType_,
      {PartInfoA});
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
    TheHelper_.callFunction(
        TheModule,
        "contra_legion_partitions_destroy",
        VoidType_,
        FunArgVs);
  }

}


//==============================================================================
// Codegen the field arguments
//==============================================================================
void LegionTasker::createFieldArguments(
    llvm::Module & TheModule,
    Value* LauncherA,
    const std::vector<Value*> & ArgVorAs,
    const std::vector<Value*> & PartVorAs,
    Value* IndexSpaceA,
    Value* PartInfoA )
{
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
      IndexPartitionType_->getPointerTo(),
      FieldDataType_->getPointerTo()
    };

    auto FunF = TheHelper_.createFunction(
        TheModule,
        "contra_legion_index_add_region_requirement",
        VoidType_,
        FunArgTs);

    for (unsigned i=0; i<NumArgs; i++) {
      auto FieldA = ArgVorAs[i];
      if (!isField(FieldA)) continue;
      FieldA = TheHelper_.getAsAlloca(FieldA);
      Value* PartA = Constant::getNullValue(IndexPartitionType_->getPointerTo());
      if (PartVorAs[i]) PartA =TheHelper_.getAsAlloca(PartVorAs[i]);
      std::vector<Value*> FunArgVs = {
        RuntimeA,
        ContextA,
        LauncherA,
        IndexSpaceA,
        PartInfoA,
        PartA,
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

    auto FunF = TheHelper_.createFunction(
        TheModule,
        "contra_legion_task_add_region_requirement",
        VoidType_,
        FunArgTs);
  
    for (unsigned i=0; i<NumArgs; i++) {
      auto FieldV = ArgVorAs[i];
      if (!isField(FieldV)) continue;
      FieldV = TheHelper_.getAsAlloca(FieldV); 
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
  
  Value* OpaqueRV = TheHelper_.callFunction(
      TheModule,
      FuncN,
      OpaqueRT,
      {},
      Name);
  
  auto OpaqueA = TheHelper_.createEntryBlockAlloca(OpaqueT, Name);
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

  TheHelper_.callFunction(
      TheModule,
      FuncN,
      VoidType_,
      {OpaqueRV});
}
  
//==============================================================================
// Destroy task arguments
//==============================================================================
void LegionTasker::destroyGlobalArguments(
    Module& TheModule,
    AllocaInst* TaskArgsA)
{
  auto ArgDataPtrV = TheHelper_.extractValue(TaskArgsA, 0);
  TheHelper_.createFree(ArgDataPtrV);
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
  //----------------------------------------------------------------------------
  // execution_constraint_set
  
  ExecSetA = createOpaqueType(TheModule, ExecSetType_,
      "legion_execution_constraint_set_create", "execution_constraint");
  
  //----------------------------------------------------------------------------
  // add constraint
  
  auto ProcIdV = llvmValue(TheContext_, ProcIdType_, LOC_PROC);  
  auto ExecSetV = load(ExecSetA, TheModule, "execution_constraint");
  std::vector<Value*> AddExecArgVs = {ExecSetV, ProcIdV};
  
  TheHelper_.callFunction(
      TheModule,
      "legion_execution_constraint_set_add_processor_constraint",
      VoidType_,
      AddExecArgVs);

  //----------------------------------------------------------------------------
  // task_layout_constraint_set
 
  LayoutSetA = createOpaqueType(TheModule, LayoutSetType_,
      "legion_task_layout_constraint_set_create", "layout_constraint"); 
  
  //----------------------------------------------------------------------------
  // options
  TaskConfigA = TheHelper_.createEntryBlockAlloca(TaskConfigType_, "options");
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
  ContextA = TheHelper_.createEntryBlockAlloca(WrapperF, ContextType_, "context.alloca");
  RuntimeA = TheHelper_.createEntryBlockAlloca(WrapperF, RuntimeType_, "runtime.alloca");


  // allocate arguments
  std::vector<Value*> WrapperArgVs;
  WrapperArgVs.reserve(WrapperArgTs.size());

  unsigned ArgIdx = 0;
  for (auto &Arg : WrapperF->args()) {
    // get arg type
    auto ArgT = WrapperArgTs[ArgIdx];
    // Create an alloca for this variable.
    auto ArgN = std::string(Arg.getName()) + ".alloca";
    auto Alloca = TheHelper_.createEntryBlockAlloca(WrapperF, ArgT, ArgN);
    // Store the initial value into the alloca.
    Builder_.CreateStore(&Arg, Alloca);
    WrapperArgVs.emplace_back(Alloca);
    ArgIdx++;
  }

  // loads
  auto DataV = TheHelper_.load(WrapperArgVs[0], "data");
  auto DataLenV = TheHelper_.load(WrapperArgVs[1], "datalen");
  //auto UserDataV = Builder_.CreateLoad(VoidPtrType_, WrapperArgVs[2], "userdata");
  //auto UserLenV = Builder_.CreateLoad(SizeType_, WrapperArgVs[3], "userlen");
  auto ProcIdV = TheHelper_.load(WrapperArgVs[4], "proc_id");

  //----------------------------------------------------------------------------
  // call to preamble

  // create temporaries
  auto TaskA = TheHelper_.createEntryBlockAlloca(WrapperF, TaskType_, "task.alloca");
 
  auto RegionsT = RegionType_->getPointerTo();
  auto RegionsA = TheHelper_.createEntryBlockAlloca(WrapperF, RegionsT, "regions.alloca");
  auto NullV = Constant::getNullValue(RegionsT);
  Builder_.CreateStore(NullV, RegionsA);

  auto NumRegionsA = TheHelper_.createEntryBlockAlloca(WrapperF, NumRegionsType_, "num_regions");
  auto ZeroV = llvmValue(TheContext_, NumRegionsType_, 0);
  Builder_.CreateStore(ZeroV, NumRegionsA);
 
  // args
  std::vector<Value*> PreambleArgVs = {
    DataV,
    DataLenV,
    ProcIdV,
    TaskA,
    RegionsA,
    NumRegionsA,
    ContextA,
    RuntimeA };
  TheHelper_.callFunction(
      TheModule,
      "legion_task_preamble",
      VoidType_,
      PreambleArgVs);
  
  //----------------------------------------------------------------------------
  // Get task args

  auto TaskRV = load(TaskA, TheModule, "task");
  Value* TaskArgsV = TheHelper_.callFunction(
      TheModule,
      "legion_task_get_args",
      VoidPtrType_,
      {TaskRV},
      "args");
  
  auto TaskArgsA = TheHelper_.createEntryBlockAlloca(WrapperF, VoidPtrType_, "args.alloca");
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
      auto ArgA = TheHelper_.createEntryBlockAlloca(WrapperF, ArgT, ArgStr);
      TaskArgAs.emplace_back(ArgA);
    }
    else {
      auto ArgT = TaskArgTs[i];
      auto ArgA = TheHelper_.createEntryBlockAlloca(WrapperF, ArgT, ArgStr);
      TaskArgAs.emplace_back(ArgA);
    }
  }
  
  //----------------------------------------------------------------------------
  // get user types

  auto ArrayT = ArrayType::get(CharType_, NumArgs);
  auto ArrayA = TheHelper_.createEntryBlockAlloca(WrapperF, ArrayT, "isfuture");
  auto ArgDataPtrV = TheHelper_.load(TaskArgsA, "args");
  Builder_.CreateMemCpy(ArrayA, 1, ArgDataPtrV, 1, llvmValue<size_t>(TheContext_, NumArgs)); 
  
  //----------------------------------------------------------------------------
  // unpack user variables
  
  auto OffsetT = SizeType_;
  auto OffsetA = TheHelper_.createEntryBlockAlloca(WrapperF, OffsetT, "offset.alloca");
  Builder_.CreateStore( llvmValue(TheContext_, OffsetT, NumArgs), OffsetA );
  
  for (unsigned i=0; i<NumArgs; i++) {
    // Create blocks for the then and else cases.
    BasicBlock * ThenBB = BasicBlock::Create(TheContext_, "then", WrapperF);
    BasicBlock * MergeBB = BasicBlock::Create(TheContext_, "ifcont");
    // evaluate the condition
    auto ArgTypeV = TheHelper_.extractValue(ArrayA, i);
    auto EnumV = llvmValue(TheContext_, ArgTypeV->getType(), static_cast<char>(ArgType::None));
    auto CondV = Builder_.CreateICmpEQ(ArgTypeV, EnumV, "argtype");
    Builder_.CreateCondBr(CondV, ThenBB, MergeBB);
    // Emit then block
    Builder_.SetInsertPoint(ThenBB);
    // copy
    auto ArgSizeV = deserialize(TaskArgAs[i], TaskArgsA, OffsetA);
    // increment
    TheHelper_.increment(OffsetA, ArgSizeV, "offset");
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

    auto GetRangeF = TheHelper_.createFunction(
        TheModule,
        "contra_legion_range_from_index_partition",
        VoidType_,
        GetRangeArgTs);

    for (unsigned i=0; i<NumArgs; i++) {
      auto ArgN = TaskArgNs[i];
      auto vit = VarOverrides.find(ArgN);
      auto ForceIndex = (vit != VarOverrides.end() && vit->second.isPartition());
      if (isRange(TaskArgAs[i])) {
        Builder_.CreateCall(SplitRangeF, {RuntimeA, ContextA, TaskA, TaskArgAs[i]});
      }
      else if (ForceIndex || isPartition(TaskArgAs[i])) {
        auto ArgA = TheHelper_.createEntryBlockAlloca(WrapperF, IndexSpaceDataType_, ArgN);
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
  auto GetFutureF = TheHelper_.createFunction(
      TheModule,
      "legion_task_get_future",
      FutureRT,
      GetFutureArgTs);
  
  auto FutureIndexT = FutureIdType_;
  auto FutureIndexA = TheHelper_.createEntryBlockAlloca(WrapperF, FutureIndexT, "futureid.alloca");
  Builder_.CreateStore( Constant::getNullValue(FutureIndexT), FutureIndexA );
  
  for (unsigned i=0; i<NumArgs; i++) {
    // Create blocks for the then and else cases.
    BasicBlock * ThenBB = BasicBlock::Create(TheContext_, "then", WrapperF);
    BasicBlock * MergeBB = BasicBlock::Create(TheContext_, "ifcont");
    // evaluate the condition
    auto ArgTypeV = TheHelper_.extractValue(ArrayA, i);
    auto EnumV = llvmValue(TheContext_, ArgTypeV->getType(), static_cast<char>(ArgType::Future));
    auto CondV = Builder_.CreateICmpEQ(ArgTypeV, EnumV, "argtype");
    Builder_.CreateCondBr(CondV, ThenBB, MergeBB);
    // Emit then block
    Builder_.SetInsertPoint(ThenBB);
    // get future
    auto TaskRV = load(TaskA, TheModule, "task");
    auto FutureIndexV = TheHelper_.load(FutureIndexA, "futureid");
    std::vector<Value*> GetFutureArgVs = {TaskRV, FutureIndexV};
    auto FutureRV = Builder_.CreateCall(GetFutureF, GetFutureArgVs, "get_future");
    auto FutureA = TheHelper_.createEntryBlockAlloca(WrapperF, FutureType_, "future");
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
    TheHelper_.increment(FutureIndexA, 1, "futureid");
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
  auto GetFieldDataF = TheHelper_.createFunction(
      TheModule,
      "contra_legion_unpack_field_data",
      VoidType_,
      {VoidPtrType_, UInt32PtrType, UInt32PtrType});
    
  auto AccessorDataPtrT = AccessorDataType_->getPointerTo();
  std::vector<Type*> GetFieldArgTs = {
    RuntimeA->getType(),
    RegionsA->getType(),
    NumRegionsA->getType(),
    UInt32PtrType,
    UInt32PtrType,
    AccessorDataPtrT};
  auto GetFieldF = TheHelper_.createFunction(
      TheModule,
      "contra_legion_get_accessor",
      VoidType_,
      GetFieldArgTs);

  auto FieldIndexT = FieldIdType_;
  auto FieldIndexA = TheHelper_.createEntryBlockAlloca(WrapperF, FieldIndexT, "field.alloca");
  Builder_.CreateStore( Constant::getNullValue(FieldIndexT), FieldIndexA );
  
  auto FieldIdA = TheHelper_.createEntryBlockAlloca(WrapperF, UInt32Type, "fieldid.alloca");
  auto RegionIdA = TheHelper_.createEntryBlockAlloca(WrapperF, UInt32Type, "regid.alloca");
  
  for (unsigned i=0; i<NumArgs; i++) {
    // Create blocks for the then and else cases.
    BasicBlock * ThenBB = BasicBlock::Create(TheContext_, "then", WrapperF);
    BasicBlock * MergeBB = BasicBlock::Create(TheContext_, "ifcont");
    // evaluate the condition
    auto ArgTypeV = TheHelper_.extractValue(ArrayA, i);
    auto EnumV = llvmValue(TheContext_, ArgTypeV->getType(), static_cast<char>(ArgType::Field));
    auto CondV = Builder_.CreateICmpEQ(ArgTypeV, EnumV, "argtype");
    Builder_.CreateCondBr(CondV, ThenBB, MergeBB);
    // Emit then block
    Builder_.SetInsertPoint(ThenBB);
    // unpack the field data
    auto ArgGEP = TheHelper_.offsetPointer(TaskArgsA, OffsetA);
    Builder_.CreateCall( GetFieldDataF, {ArgGEP, FieldIdA, RegionIdA} );
    // get field pointer
    auto ArgPtrV = TheHelper_.createBitCast(TaskArgAs[i], AccessorDataPtrT);
    std::vector<Value*> GetFieldArgVs = {
      RuntimeA,
      RegionsA,
      NumRegionsA,
      RegionIdA,
      FieldIdA,
      ArgPtrV };
    Builder_.CreateCall( GetFieldF, GetFieldArgVs );
    // increment
    TheHelper_.increment(FieldIndexA, 1, "fieldid");
    TheHelper_.increment(OffsetA, 8, "offset");
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
    auto DomainPointA = TheHelper_.createEntryBlockAlloca(WrapperF, DomainPointType_, "domain_point.alloca");

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

    auto PointDataGEP = TheHelper_.getElementPointer(DomainPointA, 1);
    auto PointDataV = TheHelper_.load(PointDataGEP);
    auto IndexV = Builder_.CreateExtractValue(PointDataV, 0);

    IndexA = TheHelper_.createEntryBlockAlloca(WrapperF, llvmType<int_t>(TheContext_), "index");
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
  bool HasNonVoidResult = ResultV && !ResultV->getType()->isVoidTy();
  if (HasNonVoidResult) {
    
    // store result
    auto ResultT = ResultV->getType();
    auto ResultA = TheHelper_.createEntryBlockAlloca(ResultT, "result");
    Builder_.CreateStore( ResultV, ResultA );

    // return size
    auto RetsizeT = RetsizeV->getType();
    RetsizeV = getSerializedSize(ResultV, RetsizeT);
    auto RetsizeA = TheHelper_.createEntryBlockAlloca(RetsizeT, "retsize");
    Builder_.CreateStore( RetsizeV, RetsizeA );

    // allocate space for return value
    RetsizeV = TheHelper_.load(RetsizeA);
    auto MallocI = TheHelper_.createMalloc(ByteType_, RetsizeV, "retval");
    RetvalA = TheHelper_.createEntryBlockAlloca(RetvalT, "retval");
    Builder_.CreateStore(MallocI, RetvalA );

    // copy data
    RetvalV = TheHelper_.load(RetvalA);
    RetsizeV = TheHelper_.load(RetsizeA);
    Builder_.CreateMemCpy(RetvalV, 1, ResultA, 1, RetsizeV); 
    serialize(ResultA, RetvalV);


    // final loads
    RetsizeV = TheHelper_.load(RetsizeA);
    RetvalV = TheHelper_.load(RetvalA);
    
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
  TheHelper_.callFunction(
      TheModule,
      "legion_task_postamble",
      VoidType_,
      PostambleArgVs);
  
  //----------------------------------------------------------------------------
  // Free memory
  if (HasNonVoidResult) {
    RetvalV = TheHelper_.load(RetvalA);
    TheHelper_.createFree(RetvalV);
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
  
  auto PreRetT = TaskVariantIdType_;
  TaskIdV = TheHelper_.callFunction(
      TheModule,
      "legion_runtime_register_task_variant_fnptr",
      PreRetT,
      PreArgVs,
      "task_variant_id");
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
  
  auto PreRetT = TaskVariantIdType_;
  TaskIdV = TheHelper_.callFunction(
      TheModule,
      "legion_runtime_preregister_task_variant_fnptr",
      PreRetT,
      PreArgVs,
      "task_variant_id");
  
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
  TheHelper_.callFunction(
      TheModule,
      "legion_runtime_set_top_level_task_id",
      VoidType_,
      SetArgVs);
}
  
//==============================================================================
// start runtime
//==============================================================================
Value* LegionTasker::startRuntime(Module &TheModule, int Argc, char ** Argv)
{

  TheHelper_.callFunction(
      TheModule,
      "contra_legion_startup",
      VoidType_);

  auto ArgcV = llvmValue(TheContext_, Int32Type_, Argc);

  std::vector<Constant*> ArgVs;
  for (int i=0; i<Argc; ++i)
    ArgVs.emplace_back( llvmString(TheContext_, TheModule, Argv[i]) );

  auto ZeroC = Constant::getNullValue(IntegerType::getInt32Ty(TheContext_));
  auto ArgvV = llvmArray(TheContext_, TheModule, ArgVs, {ZeroC, ZeroC});
  
  auto BackV = llvmValue(TheContext_, BoolType_, false);

  std::vector<Value*> StartArgVs = { ArgcV, ArgvV, BackV };
  auto RetI = TheHelper_.callFunction(
      TheModule,
      "legion_runtime_start",
      Int32Type_,
      StartArgVs,
      "start");
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
  
  auto ArgDataPtrV = TheHelper_.extractValue(TaskArgsA, 0);
  auto ArgSizeV = TheHelper_.extractValue(TaskArgsA, 1);

  auto LauncherRT = reduceStruct(TaskLauncherType_, TheModule);

  std::vector<Value*> LaunchArgVs = {TaskIdV, ArgDataPtrV, ArgSizeV, 
    PredicateV, MapperIdV, MappingTagIdV};
  Value* LauncherRV = TheHelper_.callFunction(
      TheModule,
      "legion_task_launcher_create",
      LauncherRT,
      LaunchArgVs,
      "launcher_create");
  auto LauncherA = TheHelper_.createEntryBlockAlloca(TaskLauncherType_, "task_launcher.alloca");
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
  auto FutureRT = reduceStruct(FutureType_, TheModule);
  auto FutureRV = TheHelper_.callFunction(
      TheModule,
      "legion_task_launcher_execute",
      FutureRT,
      ExecArgVs,
      "launcher_exec");
  auto FutureA = TheHelper_.createEntryBlockAlloca(FutureType_, "future.alloca");
  store(FutureRV, FutureA);

  //----------------------------------------------------------------------------
  // Destroy launcher
  
  destroyOpaqueType(TheModule, LauncherA, "legion_task_launcher_destroy", "task_launcher");
  
  //----------------------------------------------------------------------------
  // Deallocate storate
  destroyGlobalArguments(TheModule, TaskArgsA);

  return TheHelper_.load(FutureA);
}

//==============================================================================
// Launch an index task
//==============================================================================
Value* LegionTasker::launch(
    Module &TheModule,
    const std::string & Name,
    int TaskId,
    const std::vector<Value*> & ArgAs,
    const std::vector<Value*> & PartAs,
    Value* RangeV,
    bool CleanupPartitions )
{
  auto RealT = llvmType<real_t>(TheContext_);
  auto TimerA = TheHelper_.createEntryBlockAlloca(RealT);
  
  TheHelper_.callFunction(
      TheModule,
      "contra_legion_timer_start",
      VoidType_,
      {TimerA});
  
  // temporaries
  auto & LegionE = getCurrentTask();
  const auto & ContextA = LegionE.ContextAlloca;
  const auto & RuntimeA = LegionE.RuntimeAlloca;
  auto & PartInfoA = LegionE.PartInfoAlloca;

  if (!PartInfoA) PartInfoA = createPartitionInfo(TheModule);
  pushPartitionInfo(TheModule, PartInfoA);

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

  Value* IndexSpaceA = TheHelper_.getAsAlloca(RangeV);
  
  auto DomainRectA = TheHelper_.createEntryBlockAlloca(DomainRectType_, "domain");

  std::vector<Value*> DomainFromArgVs = { RuntimeA, IndexSpaceA, DomainRectA };
  TheHelper_.callFunction(
      TheModule,
      "contra_legion_domain_create",
      VoidType_,
      DomainFromArgVs);
  
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
  
  auto ArgDataPtrV = TheHelper_.extractValue(TaskArgsA, 0);
  auto ArgSizeV = TheHelper_.extractValue(TaskArgsA, 1);

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
  auto LauncherA = TheHelper_.createEntryBlockAlloca(IndexLauncherType_, "task_launcher.alloca");
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
      PartAs,
      IndexSpaceA,
      PartInfoA);
  
  TheHelper_.callFunction(
      TheModule,
      "contra_legion_timer_stop",
      VoidType_,
      {TimerA});
  
  //----------------------------------------------------------------------------
  // Execute
  
  LauncherRV = load(LauncherA, TheModule, "task_launcher");
  auto ContextV = load(ContextA, TheModule, "context");
  auto RuntimeV = load(RuntimeA, TheModule, "runtime");

  std::vector<Value*> ExecArgVs = { RuntimeV, ContextV, LauncherRV };
  auto FutureMapRT = reduceStruct(FutureMapType_, TheModule);
  Value* FutureMapRV = TheHelper_.callFunction(
      TheModule,
      "legion_index_launcher_execute",
      FutureMapRT,
      ExecArgVs,
      "launcher_exec");
  auto FutureMapA = TheHelper_.createEntryBlockAlloca(FutureMapType_, "future_map.alloca");
  store(FutureMapRV, FutureMapA);
  
	//----------------------------------------------------------------------------
  // Destroy argument map
  
  destroyOpaqueType(TheModule, ArgMapA, "legion_argument_map_destroy", "arg_map");

	//----------------------------------------------------------------------------
  // Destroy future map

  TheHelper_.callFunction(
      TheModule,
      "legion_future_map_wait_all_results",
      VoidType_,
      {FutureMapRV});
  
  destroyOpaqueType(TheModule, FutureMapA, "legion_future_map_destroy", "future_map");
  
  //----------------------------------------------------------------------------
  // cleanup
  
  destroyOpaqueType(TheModule, LauncherA, "legion_index_launcher_destroy", "task_launcher");
  
  popPartitionInfo(TheModule, PartInfoA);
  
  destroyGlobalArguments(TheModule, TaskArgsA);

  //return Builder_.CreateLoad(FutureMapType_, FutureMapA);
  return nullptr;
}


//==============================================================================
// get a future value
//==============================================================================
Value* LegionTasker::loadFuture(
    Module &TheModule,
    Value* Future,
    Type *DataT)
{
  // args
  auto FutureA = TheHelper_.getAsAlloca(Future);
  auto FutureRV = load(FutureA, TheModule, "future");
  Value* DataPtrV = TheHelper_.callFunction(
      TheModule,
      "legion_future_get_untyped_pointer",
      VoidPtrType_,
      {FutureRV},
      "future");
  
  auto DataA = TheHelper_.createEntryBlockAlloca(DataT);
  deserialize(DataA, DataPtrV);

  return TheHelper_.load(DataA, "future");
}

//==============================================================================
// insert a future value
//==============================================================================
AllocaInst* LegionTasker::createFuture(
    Module &,
    const std::string & Name)
{
  auto FutureA = TheHelper_.createEntryBlockAlloca(FutureType_, "future.alloca");
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
  auto ValueA = TheHelper_.createEntryBlockAlloca(ValueT);
  Builder_.CreateStore( ValueV, ValueA );

  auto ValuePtrV = TheHelper_.createBitCast(ValueA, VoidPtrType_);

  auto ValueSizeV = TheHelper_.getTypeSize<size_t>(ValueT);

  std::vector<Value*> FunArgVs = {RuntimeV, ValuePtrV, ValueSizeV};
    
  auto FutureRT = reduceStruct(FutureType_, TheModule);
  auto FutureRV = TheHelper_.callFunction(
      TheModule,
      "legion_future_from_untyped_pointer",
      FutureRT,
      FunArgVs,
      "future");
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

  auto ValueRV = Builder_.CreateExtractValue(ValueV, 0);
  auto FutureRV = TheHelper_.callFunction(
      TheModule,
      "legion_future_copy",
      FutureRT,
      {ValueRV},
      "future");
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
    const std::string & VarN,
    Type* VarT,
    Value* RangeV,
    Value* VarV)
{
  auto FieldA = TheHelper_.createEntryBlockAlloca(FieldDataType_, "field");
  
  const auto & LegionE = getCurrentTask();
  const auto & ContextA = LegionE.ContextAlloca;
  const auto & RuntimeA = LegionE.RuntimeAlloca;
  
  auto NameV = llvmString(TheContext_, TheModule, VarN);

  Value* DataSizeV;
  if (VarV) {
    DataSizeV = TheHelper_.getTypeSize<size_t>(VarT);
    VarV = TheHelper_.getAsAlloca(VarV);
  }
  else {
    DataSizeV = llvmValue<size_t>(TheContext_, 0);
    VarV = Constant::getNullValue(VoidPtrType_);
  }
    
  Value* IndexSpaceA = TheHelper_.getAsAlloca(RangeV);
  
  std::vector<Value*> FunArgVs = {
    RuntimeA,
    ContextA,
    NameV,
    DataSizeV, 
    VarV,
    IndexSpaceA,
    FieldA};
  
  std::string FunN = isRange(RangeV) ?
    "contra_legion_field_create" : "contra_legion_field_create_from_partition";
  
  TheHelper_.callFunction(
      TheModule,
      FunN,
      VoidType_,
      FunArgVs);
    
  return FieldA;
}

//==============================================================================
// Create a legion field
//==============================================================================
void LegionTasker::createField(
    Module & TheModule,
    Value* FieldA,
    const std::string & VarN,
    Type* VarT,
    Value* RangeV,
    Value* VarV)
{
  
  const auto & LegionE = getCurrentTask();
  const auto & ContextA = LegionE.ContextAlloca;
  const auto & RuntimeA = LegionE.RuntimeAlloca;
  
  auto NameV = llvmString(TheContext_, TheModule, VarN);

  Value* DataSizeV;
  if (VarV) {
    DataSizeV = TheHelper_.getTypeSize<size_t>(VarT);
    VarV = TheHelper_.getAsAlloca(VarV);
  }
  else {
    DataSizeV = llvmValue<size_t>(TheContext_, 0);
    VarV = Constant::getNullValue(VoidPtrType_);
  }
    
  Value* IndexSpaceA = TheHelper_.getAsAlloca(RangeV);
  
  std::vector<Value*> FunArgVs = {
    RuntimeA,
    ContextA,
    NameV,
    DataSizeV, 
    VarV,
    IndexSpaceA,
    FieldA};
  
  std::string FunN = isRange(RangeV) ?
    "contra_legion_field_create" : "contra_legion_field_create_from_partition";
  
  TheHelper_.callFunction(
      TheModule,
      FunN,
      VoidType_,
      FunArgVs);
    
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
    
  TheHelper_.callFunction(
      TheModule,
      "contra_legion_field_destroy",
      VoidType_,
      FunArgVs);
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
    const std::string & Name,
    Value* StartV,
    Value* EndV,
    Value* StepV)
{
  auto IndexSpaceA = TheHelper_.createEntryBlockAlloca(IndexSpaceDataType_, "index");

  StartV = TheHelper_.getAsValue(StartV);
  EndV = TheHelper_.getAsValue(EndV);
  if (StepV) StepV = TheHelper_.getAsValue(StepV);

  if (isInsideTask()) {
    const auto & LegionE = getCurrentTask();
    const auto & ContextA = LegionE.ContextAlloca;
    const auto & RuntimeA = LegionE.RuntimeAlloca;
    
    auto NameV = llvmString(TheContext_, TheModule, Name);

    std::vector<Value*> FunArgVs = {RuntimeA, ContextA, NameV, StartV, EndV, IndexSpaceA};
    
    TheHelper_.callFunction(
        TheModule,
        "contra_legion_index_space_create",
        VoidType_,
        FunArgVs);
  }
  else { 
    TheHelper_.insertValue(IndexSpaceA, StartV, 0);
    auto OneC = llvmValue<int_t>(TheContext_, 1);
    EndV = Builder_.CreateAdd(EndV, OneC);
    TheHelper_.insertValue(IndexSpaceA, EndV, 1);
    if (!StepV) StepV = OneC;
    TheHelper_.insertValue(IndexSpaceA, StepV, 2);
  }

  
  return IndexSpaceA;

}

//==============================================================================
// create a range
//==============================================================================
AllocaInst* LegionTasker::createRange(
    Module & TheModule,
    Value* ValueV,
    const std::string & Name)
{
  auto IndexSpaceA = TheHelper_.createEntryBlockAlloca(IndexSpaceDataType_, "index");

  if (isInsideTask()) {
    const auto & LegionE = getCurrentTask();
    const auto & ContextA = LegionE.ContextAlloca;
    const auto & RuntimeA = LegionE.RuntimeAlloca;
    
    std::vector<Value*> FunArgVs = {RuntimeA, ContextA, ValueV, IndexSpaceA};
    
    TheHelper_.callFunction(
        TheModule,
        "contra_legion_index_space_create_from_size",
        VoidType_,
        FunArgVs);
  }
  else { 
    auto ZeroC = llvmValue<int_t>(TheContext_, 0);
    auto OneC = llvmValue<int_t>(TheContext_, 1);
    TheHelper_.insertValue(IndexSpaceA, ZeroC,  0);
    TheHelper_.insertValue(IndexSpaceA, ValueV, 1);
    TheHelper_.insertValue(IndexSpaceA, OneC,   2);
  }

  return IndexSpaceA;

}

//==============================================================================
// create a range
//==============================================================================
AllocaInst* LegionTasker::createRange(
    Module & TheModule,
    Type*,
    Value* ValueV,
    const std::string & Name)
{
  auto IndexSpaceA = TheHelper_.createEntryBlockAlloca(IndexSpaceDataType_, "index");
  auto ValueA = TheHelper_.getAsAlloca(ValueV);

  if (isInsideTask()) {
    const auto & LegionE = getCurrentTask();
    const auto & ContextA = LegionE.ContextAlloca;
    const auto & RuntimeA = LegionE.RuntimeAlloca;
    
    std::vector<Value*> FunArgVs = {RuntimeA, ContextA, ValueA, IndexSpaceA};
    
    TheHelper_.callFunction(
        TheModule,
        "contra_legion_index_space_create_from_array",
        VoidType_,
        FunArgVs);
  }
  else { 
    
    auto IntT = llvmType<int_t>(TheContext_);
    auto SumV = TheHelper_.callFunction(
        TheModule,
        "contra_legion_sum_array",
        IntT,
        {ValueA});
    
    auto ZeroC = llvmValue<int_t>(TheContext_, 0);
    auto OneC = llvmValue<int_t>(TheContext_, 1);
    TheHelper_.insertValue(IndexSpaceA, ZeroC, 0);
    TheHelper_.insertValue(IndexSpaceA, SumV,  1);
    TheHelper_.insertValue(IndexSpaceA, OneC,  2);
  }

  
  return IndexSpaceA;

}

//==============================================================================
// create a range
//==============================================================================
AllocaInst* LegionTasker::partition(
    Module & TheModule,
    Value* IndexSpaceA,
    Value* IndexPartitionA,
    Value* ValueA)
{
  auto & LegionE = getCurrentTask();
  const auto & ContextA = LegionE.ContextAlloca;
  const auto & RuntimeA = LegionE.RuntimeAlloca;
  auto & PartInfoA = LegionE.PartInfoAlloca;

  if (!PartInfoA) PartInfoA = createPartitionInfo(TheModule);
  
  IndexSpaceA = TheHelper_.getAsAlloca(IndexSpaceA);
  
  auto IndexPartA = TheHelper_.createEntryBlockAlloca(IndexPartitionType_);

  //------------------------------------
  if (isRange(ValueA)) {
    ValueA = TheHelper_.getAsAlloca(ValueA);

    std::vector<Value*> FunArgVs = {
      RuntimeA,
      ContextA,
      ValueA,
      IndexSpaceA,
      PartInfoA,
      IndexPartA};
    
    TheHelper_.callFunction(
        TheModule,
        "contra_legion_index_space_partition",
        VoidType_,
        FunArgVs);
  }
  //------------------------------------
  else if (isField(ValueA)) {
    ValueA = TheHelper_.getAsAlloca(ValueA);
    IndexPartitionA = TheHelper_.getAsAlloca(IndexPartitionA);
    std::vector<Value*> FunArgVs = {
      RuntimeA,
      ContextA,
      ValueA,
      IndexSpaceA,
      IndexPartitionA,
      PartInfoA,
      IndexPartA};
    
    TheHelper_.callFunction(
        TheModule,
        "contra_legion_index_space_partition_from_field",
        VoidType_,
        FunArgVs);
  }
  //------------------------------------
  else {
    auto ValueV = TheHelper_.getAsValue(ValueA);

    std::vector<Value*> FunArgVs = {
      RuntimeA,
      ContextA,
      ValueV,
      IndexSpaceA,
      PartInfoA,
      IndexPartA};
    
    TheHelper_.callFunction(
        TheModule,
        "contra_legion_index_space_partition_from_size",
        VoidType_,
        FunArgVs);
  }
  //------------------------------------

  return IndexPartA;
    
}

//==============================================================================
// create a range
//==============================================================================
AllocaInst* LegionTasker::partition(
    Module & TheModule,
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

  auto IndexPartA = TheHelper_.createEntryBlockAlloca(IndexPartitionType_);
    
  IndexSpaceA = TheHelper_.getAsAlloca(IndexSpaceA);
  auto ValueA = TheHelper_.getAsAlloca(ValueV);
  
  std::vector<Value*> FunArgVs = {
    RuntimeA,
    ContextA,
    ValueA,
    IndexSpaceA,
    PartInfoA,
    IndexPartA,
    llvmValue<bool>(TheContext_, ReportSizeError)
  };
  
  TheHelper_.callFunction(
      TheModule,
      "contra_legion_index_space_partition_from_array",
      VoidType_,
      FunArgVs);
  
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

    Value* IndexSpaceA = TheHelper_.getAsAlloca(RangeV);
    
    std::vector<Value*> FunArgVs = {RuntimeA, ContextA, IndexSpaceA};
    
    TheHelper_.callFunction(
        TheModule,
        "contra_legion_index_space_destroy",
        VoidType_,
        FunArgVs);
  }
}

//==============================================================================
// get a range start
//==============================================================================
llvm::Value* LegionTasker::getRangeStart(Module &TheModule, Value* RangeV)
{ return TheHelper_.extractValue(RangeV, 0); }

//==============================================================================
// get a range start
//==============================================================================
llvm::Value* LegionTasker::getRangeEnd(Module &TheModule, Value* RangeV)
{
  Value* EndV = TheHelper_.extractValue(RangeV, 1);
  auto OneC = llvmValue<int_t>(TheContext_, 1);
  return Builder_.CreateSub(EndV, OneC);
}


//==============================================================================
// get a range size
//==============================================================================
llvm::Value* LegionTasker::getRangeSize(Module &TheModule, Value* RangeV)
{
  auto StartV = TheHelper_.extractValue(RangeV, 0);
  auto EndV = TheHelper_.extractValue(RangeV, 1);
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
  auto StartV = TheHelper_.extractValue(RangeA, 0); 
  IndexV = TheHelper_.getAsValue(IndexV);
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
  auto ValueA = TheHelper_.getAsAlloca(ValueV);
  auto ValueT = ValueA->getAllocatedType();

  Value* AccessorA = TheHelper_.getAsAlloca(AccessorV);
    
  auto DataSizeV = TheHelper_.getTypeSize<size_t>(ValueT);
  
  std::vector<Value*> FunArgVs = { AccessorA, ValueA, DataSizeV };
  
  if (IndexV) {
    FunArgVs.emplace_back( TheHelper_.getAsValue(IndexV) );
  }
  else {
    FunArgVs.emplace_back( llvmValue<int_t>(TheContext_, 0) );
  }
  
  TheHelper_.callFunction(
      TheModule,
      "contra_legion_accessor_write",
      VoidType_,
      FunArgVs);
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
  auto AccessorA = TheHelper_.getAsAlloca(AccessorV);
    
  auto ValueA = TheHelper_.createEntryBlockAlloca(ValueT);
  auto DataSizeV = TheHelper_.getTypeSize<size_t>(ValueT);

  std::vector<Value*> FunArgVs = { AccessorA, ValueA, DataSizeV };
  
  if (IndexV) {
    FunArgVs.emplace_back( TheHelper_.getAsValue(IndexV) );
  }
  else {
    FunArgVs.emplace_back( llvmValue<int_t>(TheContext_, 0) );
  }

  TheHelper_.callFunction(
      TheModule,
      "contra_legion_accessor_read",
      VoidType_,
      FunArgVs);

  return TheHelper_.load(ValueA);
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
  
  TheHelper_.callFunction(
      TheModule,
      "contra_legion_accessor_destroy",
      VoidType_,
      FunArgVs);
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
// destroey an accessor
//==============================================================================
void LegionTasker::destroyPartition(
    Module &TheModule,
    Value* PartitionA)
{
  const auto & LegionE = getCurrentTask();
  const auto & ContextA = LegionE.ContextAlloca;
  const auto & RuntimeA = LegionE.RuntimeAlloca;

  std::vector<Value*> FunArgVs = {RuntimeA, ContextA, PartitionA};
  
  TheHelper_.callFunction(
      TheModule,
      "contra_legion_partition_destroy",
      VoidType_,
      FunArgVs);
}

}
