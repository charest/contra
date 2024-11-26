#ifndef CONTRA_LEGION_RT_HPP
#define CONTRA_LEGION_RT_HPP

#include "librt/dopevector.hpp"
#include "librt/timer.hpp"

#include <legion.h>
#include <legion/legion_c.h>


#include <forward_list>
#include <unordered_map>

////////////////////////////////////////////////////////////////////////////////
// Create new legion Reduction op
////////////////////////////////////////////////////////////////////////////////

namespace contra {

typedef void (*apply_t)(void *, const void*, off_t, off_t, size_t, bool);
typedef void (*fold_t)(void *, const void*, off_t, off_t, size_t, bool);
typedef void (*init_t)(void *, size_t);

class ReductionOp : public Realm::ReductionOpUntyped {

  apply_t apply_ptr;
  fold_t fold_ptr;
  init_t init_ptr;

public:

  ReductionOp(size_t size, apply_t apply_p, fold_t fold_p, init_t init_p) :
      ReductionOpUntyped(size, size, 0, true, true),
      apply_ptr(apply_p),
      fold_ptr(fold_p),
      init_ptr(init_p)
  {}
#if 0
    sizeof_this = sizeof(ReductionOp);
    sizeof_lhs = size;
    sizeof_rhs = size;
    sizeof_userdata = 0;
    identity = nullptr;
    userdata = nullptr;
    //cpu_apply_excl_fn = &ReductionKernels::cpu_apply_wrapper<REDOP, true>;
    //cpu_apply_nonexcl_fn = &ReductionKernels::cpu_apply_wrapper<REDOP, false>;
    //cpu_fold_excl_fn = &ReductionKernels::cpu_fold_wrapper<REDOP, true>;
    //cpu_fold_nonexcl_fn = &ReductionKernels::cpu_fold_wrapper<REDOP, false>;
  }
#endif

  virtual ReductionOpUntyped *clone(void) const
  { return new ReductionOp(sizeof_lhs, apply_ptr, fold_ptr, init_ptr); }
      
  virtual void apply(
      void *lhs,
      const void *rhs,
      size_t count,
			bool exclusive = false) const
  {
    if(exclusive) (*apply_ptr)(lhs, rhs, sizeof_lhs, sizeof_rhs, count, true);
    else          (*apply_ptr)(lhs, rhs, sizeof_lhs, sizeof_rhs, count, false);
  }

  virtual void apply_strided(
      void *lhs,
      const void *rhs,
			off_t lhs_stride,
      off_t rhs_stride,
      size_t count,
			bool exclusive = false) const
  {
    if(exclusive) (*apply_ptr)(lhs, rhs, lhs_stride, rhs_stride, count, true);
    else          (*apply_ptr)(lhs, rhs, lhs_stride, rhs_stride, count, false);
  }

  virtual void fold(
      void *rhs1,
      const void *rhs2,
      size_t count,
	    bool exclusive = false) const
  {
    if(exclusive) (*fold_ptr)(rhs1, rhs2, sizeof_lhs, sizeof_rhs, count, true);
    else          (*fold_ptr)(rhs1, rhs2, sizeof_lhs, sizeof_rhs, count, false);
  }

  virtual void fold_strided(
      void *lhs,
      const void *rhs,
		  off_t lhs_stride,
      off_t rhs_stride,
      size_t count,
		  bool exclusive = false) const
  {
    if(exclusive) (*fold_ptr)(lhs, rhs, lhs_stride, rhs_stride, count, true);
    else          (*fold_ptr)(lhs, rhs, lhs_stride, rhs_stride, count, false);
  }

  virtual void init(void *ptr, size_t count) const
  { (*init_ptr)(ptr, count); }

};

}

extern "C" {

////////////////////////////////////////////////////////////////////////////////
// Types needed for legion runtime
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
struct contra_legion_index_space_t {
  int_t start;
  int_t end;
  int_t step;
  legion_index_space_t index_space;
};

//==============================================================================
struct contra_legion_field_t {
  legion_index_space_t index_space;
  legion_field_space_t field_space;
  legion_field_allocator_t field_allocator;
  legion_field_id_t field_id;
  legion_logical_region_t logical_region;
};

//==============================================================================
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
  std::size_t data_size;
};

//==============================================================================
// Index partition data
//==============================================================================
struct contra_legion_partitions_t {
  
  //------------------------------------
  // Public Types/Usings
  
  struct IndexPartitionDeleter {
    legion_runtime_t * runtime;
    legion_context_t * context;
    bool owner;
    IndexPartitionDeleter(
        legion_runtime_t * rt,
        legion_context_t * ctx,
        bool owns = true) :
      runtime(rt), context(ctx), owner(owns)
    {}
    void operator()(legion_index_partition_t *ptr)
    {
      if (owner) legion_index_partition_destroy(*runtime, *context, *ptr);
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
  
  //------------------------------------
  // Public Data Members
  std::forward_list< IndexPartitionMap > IndexPartitions;
  LogicalPartitionMap LogicalPartitions;
  
  //------------------------------------
  // Public interface 
  std::pair<legion_index_partition_t*, bool>
    getOrCreateIndexPartition(
      legion_runtime_t *rt,
      legion_context_t *ctx,
      legion_index_space_id_t id,
      bool owner = true);
  
  std::pair<legion_logical_partition_t*, bool>
    getOrCreateLogicalPartition(
      legion_runtime_t *rt,
      legion_context_t *ctx,
      legion_field_id_t fid,
      legion_index_partition_id_t pid);
  
  legion_logical_partition_t* getLogicalPartition(
      legion_field_id_t fid,
      legion_index_partition_id_t pid);

  void push() 
  { IndexPartitions.push_front({}); }

  void pop()
  { IndexPartitions.pop_front(); }

};

////////////////////////////////////////////////////////////////////////////////
// Function prototypes for legion runtime
////////////////////////////////////////////////////////////////////////////////


// Additional startup operations
void contra_legion_startup();

/// index space creation
int_t contra_legion_sum_array(dopevector_t * arr);

/// index space creation
void contra_legion_index_space_partition(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    contra_legion_index_space_t * cs,
    contra_legion_index_space_t * is,
    legion_index_partition_t * part);

/// index space creation
void contra_legion_index_space_partition_from_size(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    int_t size,
    contra_legion_index_space_t * is,
    legion_index_partition_t * part);

/// index space creation
void contra_legion_index_space_partition_from_array(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    dopevector_t *arr,
    contra_legion_index_space_t * is,
    legion_index_partition_t * part,
    bool do_report);

/// index space creation
void contra_legion_index_space_partition_from_field(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    contra_legion_field_t *field,
    contra_legion_index_space_t * is,
    legion_index_partition_t * index_part,
    contra_legion_partitions_t ** parts,
    legion_index_partition_t * part);


/// index space partitioning
void contra_legion_index_space_create(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    const char *name,
    int_t start,
    int_t end,
    contra_legion_index_space_t * is);

/// index space partitioning
void contra_legion_index_space_create_from_index_partition(
    legion_runtime_t * runtime,
    legion_context_t *,
    legion_task_t * task,
    legion_index_partition_t * part,
    contra_legion_index_space_t * is);

/// register an index partition with an index sapce
void contra_legion_register_index_partition(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    contra_legion_index_space_t * is,
    legion_index_partition_t * index_part,
    contra_legion_partitions_t ** parts);

/// index spce destruction
void contra_legion_index_space_destroy(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    contra_legion_index_space_t * is);

/// index domain creation
void contra_legion_domain_create(
    legion_runtime_t * runtime,
    contra_legion_index_space_t * is,
    legion_domain_t * domain);


/// field creation
void contra_legion_field_create(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    const char *name,
    size_t data_size,
    void* init,
    contra_legion_index_space_t * is,
    contra_legion_field_t * fld);

/// field destruction
void contra_legion_field_destroy(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    contra_legion_field_t * field);

/// field data
void contra_legion_pack_field_data(
    contra_legion_field_t * f,
    unsigned regidx,
    void * data);

void contra_legion_unpack_field_data(
    const void *data,
    uint32_t*fid,
    uint32_t*rid);

/// field addition
void contra_legion_index_add_region_requirement(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    legion_index_launcher_t * launcher,
    contra_legion_index_space_t * cs,
    contra_legion_partitions_t ** parts,
    legion_index_partition_t * specified_part,
    contra_legion_field_t * field);

/// field addition
void contra_legion_task_add_region_requirement(
    legion_task_launcher_t * launcher,
    contra_legion_field_t * field );

/// partition create
void contra_legion_partitions_create(
    contra_legion_partitions_t ** parts);

/// partition push
void contra_legion_partitions_push(
    contra_legion_partitions_t ** parts);

/// partition pop
void contra_legion_partitions_pop(
    contra_legion_partitions_t ** parts);

/// partition destruction
void contra_legion_partitions_destroy(
    legion_runtime_t *,
    legion_context_t *,
    contra_legion_partitions_t ** parts);

/// get field accessor
void contra_legion_get_accessor(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    legion_physical_region_t **regionptr,
    uint32_t* /*num_regions*/,
    uint32_t* region_id,
    uint32_t* field_id,
    contra_legion_accessor_t* acc);

/// Accessor write
void contra_legion_accessor_write(
    contra_legion_accessor_t * acc,
    const void * data,
    int_t index = 0);

/// Accessor read
void contra_legion_accessor_read(
    contra_legion_accessor_t * acc,
    void * data,
    int_t index = 0);

/// accessor destruction
void contra_legion_accessor_destroy(
    legion_runtime_t *,
    legion_context_t *,
    contra_legion_accessor_t * acc);

/// Partition destruction
void contra_legion_partition_destroy(
    legion_runtime_t * runtime,
    legion_context_t * ctx,
    legion_index_partition_t * part);

/// get the timer
real_t get_wall_time(void);
void contra_legion_timer_start(real_t *);
void contra_legion_timer_stop(real_t *);

/// create a reduction op
void contra_legion_create_reduction(
    legion_reduction_op_id_t redop,
    contra::apply_t apply_ptr,
    contra::fold_t fold_ptr,
    contra::init_t init_ptr,
    std::size_t data_size);

} // extern


#endif // LIBRT_LEGION_RT_HPP
