#include "mpi_rt.hpp"
#include "librtmpi/mpi_utils.hpp"

#include <cstring>
#include <iostream>

using namespace contra;

mpi_runtime_t MpiRuntime;

//==============================================================================
// Check errors
//==============================================================================
void mpi_runtime_t::check(int  errcode) {
  if (errcode) {
    if (isRoot()) {
      char * str = nullptr;
      int len = 0;
      MPI_Error_string(errcode, str, &len);
      std::cerr << "MPI Error failed with error code " << errcode << std::endl;
      std::cerr << str << std::endl;
    }
    MPI_Abort(MPI_COMM_WORLD, errcode);
  }
}
  

extern "C" {
  
//==============================================================================
/// startup runtime
//==============================================================================
void contra_mpi_init()
{ 
  int rank;
  auto err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MpiRuntime.check(err);

  int size;
  err = MPI_Comm_size(MPI_COMM_WORLD, &size);
  MpiRuntime.check(err);

  MpiRuntime.setup(rank, size);
}

//==============================================================================
/// mark we are in a task
//==============================================================================
void contra_mpi_mark_task()
{ MpiRuntime.incrementTaskCounter(); }

//==============================================================================
/// unmark we are in a task
//==============================================================================
void contra_mpi_unmark_task()
{ MpiRuntime.decrementTaskCounter(); }

//==============================================================================
/// Test if we need to guard and if we are root
//==============================================================================
bool contra_mpi_test_root()
{ return MpiRuntime.getTaskCounter() > 0 || MpiRuntime.isRoot(); }

//==============================================================================
/// Get the rank
//==============================================================================
int_t contra_mpi_rank()
{ return MpiRuntime.getRank(); }

//==============================================================================
/// Get the comm size
//==============================================================================
int_t contra_mpi_size()
{ return MpiRuntime.getSize(); }

//==============================================================================
/// Get the loop bounds
//==============================================================================
void contra_mpi_loop_bounds(
    contra_index_space_t * is,
    int_t * start,
    int_t * end,
    int_t * step,
    int_t * dist)
{
  int_t comm_size = MpiRuntime.getSize();
  int_t comm_rank = MpiRuntime.getRank();

  auto size = is->size();
  auto chunk = size / comm_size;
  auto remain = size % comm_size;

  dist[0] = 0;
  
  for (int_t i=0; i<comm_size; ++i) {
    dist[i+1] = dist[i] + chunk;
    if (i < remain) dist[i+1]++;
  }
  
  *step = is->step;
  *start = dist[comm_rank] * is->step;
  *end = dist[comm_rank+1] * is->step;
}

//==============================================================================
/// create partition info
//==============================================================================
void contra_mpi_task_info_create(contra_mpi_task_info_t** info)
{ *info = new contra_mpi_task_info_t; }

//==============================================================================
// destroy partition info
//==============================================================================
void contra_mpi_task_info_destroy(contra_mpi_task_info_t** info)
{ delete (*info); }

//==============================================================================
// destroy partition info
//==============================================================================
void contra_mpi_register_index_partition(
    contra_index_space_t * is,
    contra_mpi_partition_t * part,
    contra_mpi_task_info_t** info)
{
  (*info)->register_partition(is, part);
}

//==============================================================================
// Create a field
//==============================================================================
void contra_mpi_field_create(
    const char * name,
    int_t data_size,
    const void* init,
    contra_index_space_t * is,
    contra_mpi_field_t * fld)
{
  auto fid = MpiRuntime.registerField(data_size, init);
  fld->setup(is, data_size, fid);
}

//==============================================================================
// Fetch a field
//==============================================================================
void contra_mpi_field_fetch(
    contra_index_space_t * is,
    int_t * dist,
    contra_mpi_partition_t * part,
    contra_mpi_field_t * fld)
{
  auto & Field = MpiRuntime.getRegisteredField(fld->id);
  auto comm_rank = MpiRuntime.getRank();
  auto comm_size = MpiRuntime.getSize();
      
  //----------------------------------------------------------------------------
  // allocated somewhere
  if (fld->is_allocated()) {

    auto is_same = fld->partition->id == part->id;
    
    if (!is_same && !part->indices) {
      auto part_offsets = part->offsets;
      auto data_size = fld->data_size;
      bool exchange = false;
      
      std::vector<int_t> sendcounts(comm_size, 0);
      std::vector<int_t> recvcounts(comm_size, 0);
      std::vector<int_t> sendpos(comm_size);
      
      auto comm_fld_begin = fld->rank_begin(comm_rank);
      auto comm_fld_end = fld->rank_end(comm_rank);
        
      auto comm_part_begin = part_offsets[dist[comm_rank]];
      auto comm_part_end = part_offsets[dist[comm_rank+1]];

      int_t recvcnt = 0;
      for (decltype(comm_size) i=0; i<comm_size; ++i) {
        // send
        auto begin = std::max(comm_fld_begin, part_offsets[dist[i]]);
        auto end = std::min(comm_fld_end, part_offsets[dist[i+1]]);
        sendpos[i] = (begin - comm_fld_begin) * data_size;
        sendcounts[i] = end>begin ? (end-begin) * data_size : 0;
        // recv
        begin = std::max(fld->rank_begin(i), comm_part_begin);
        end = std::min(fld->rank_end(i), comm_part_end);
        recvcounts[i] = end>begin ? (end-begin) * data_size : 0;
        recvcnt += recvcounts[i];
        if (i!=comm_rank && (sendcounts[i] || recvcounts[i])) exchange = true;
      }
        
      //------------------------------------
      // at least some info must be exchanged
      if  (exchange) {

        auto fld_data = static_cast<byte_t*>(fld->data);
      
        auto & Request = MpiRuntime.requestField(fld_data, recvcnt, 2*comm_size);

        auto recvbuf = static_cast<byte_t*>(Request.getBuffer());
        auto & requests = Request.getRequests();
       
        int tag = 0;
        auto mpi_byte_t = librtmpi::typetraits<byte_t>::type();

        recvcnt = 0;
        for (decltype(comm_size) i=0; i<comm_size; ++i) {
          auto count = recvcounts[i];
          if(count > 0) {
            auto buf = &recvbuf[recvcnt];
            requests.emplace_back();
            auto & my_request = requests.back();
            auto ret = MPI_Irecv(buf, count, mpi_byte_t, i, tag, MPI_COMM_WORLD, &my_request);
            MpiRuntime.check(ret);
          recvcnt += count;
          }
        }
      
        int_t sendcnt = 0;
        for (decltype(comm_size) i=0; i<comm_size; ++i) {
          auto count = sendcounts[i];
          if(count > 0) {
            auto buf = fld_data + sendpos[i];
            requests.emplace_back();
            auto & my_request = requests.back();
            auto ret = MPI_Isend(buf, count, mpi_byte_t, i, tag, MPI_COMM_WORLD, &my_request);
            MpiRuntime.check(ret);
            sendcnt += count;
          }
        }

      }
      // done excanghe
      //------------------------------------
      
      contra_mpi_partition_destroy(fld->partition);
      fld->redistribute(part, dist, comm_size);
      MpiRuntime.incrementPartition(part->id);
      
    } // ! is_same
    else if (!is_same && part->indices) {
      
      //------------------------------------
      // check if partition needs exchange
    
      // figure out which indices i have
      auto num_parts = part->num_parts;
      auto current_dist = part->indices->distribution;

      if (!std::equal(current_dist, current_dist+num_parts, dist)) {

        auto first_part = current_dist[comm_rank];
        auto last_part = current_dist[comm_rank+1];

        auto int_size = sizeof(int_t);
        
        std::vector<int_t> sendcounts(comm_size, 0);
        std::vector<int_t> recvcounts(comm_size, 0);

        bool exchange = false;

        for (decltype(comm_size) i=0; i<comm_size; ++i) {
          auto begin = std::max(dist[i], first_part);
          auto end = std::min(dist[i+1], last_part);
          begin = part->offsets[begin];
          end = part->offsets[end];
          sendcounts[i] = end>begin ? (end-begin) * int_size : 0;
          begin = std::max(dist[comm_rank], current_dist[i]);
          end = std::min(dist[comm_rank+1], current_dist[i+1]);
          begin = part->offsets[begin];
          end = part->offsets[end];
          recvcounts[i] = end>begin ? (end-begin) * int_size : 0;
          if (i!=comm_rank && (sendcounts[i] || recvcounts[i])) exchange = true;
        }
        
        //------------------------------------
        // at least some info must be exchanged
        if  (exchange) {
          std::cerr << "redistribution of partition from field not implemented" << std::endl;
          abort();
        }
      }
  
      //------------------------------------
      // Now fetch values
      
      auto part_indices = static_cast<int_t*>(part->indices->data);
      
      auto field_part = fld->partition;
      auto field_offset_start = field_part->offsets_begin();
      auto field_offset_end = field_part->offsets_end();
      auto field_dist = fld->distribution;
      auto num_field_parts = field_part->num_parts;

      std::vector<int_t> field_part_owners(num_field_parts);
      for (decltype(comm_size) i=0; i<comm_size; ++i)
        for (auto p=field_dist[i]; p<field_dist[i+1]; ++p)
          field_part_owners[p] = i;

      auto dist_start = dist[comm_rank];
      auto dist_end = dist[comm_rank+1];
      auto local_dist = dist_end - dist_start;
      
      size_t tot_indices = 0;
      for (int_t p=0; p<local_dist; ++p) 
        tot_indices += part->size(dist_start + p);

      std::vector<unsigned> index_owners;
      index_owners.reserve(tot_indices);

      std::vector<int_t> sendcounts(comm_size, 0);
      for (size_t i=0; i<tot_indices; ++i) {
        auto it = std::lower_bound(field_offset_start, field_offset_end, part_indices[i]);
        auto pid = std::distance(field_offset_start, it);
        auto r = field_part_owners[pid];
        sendcounts[r]++;
        index_owners.emplace_back(r);
      }
      
      std::vector<int_t> senddispls(comm_size+1);
      senddispls[0] = 0;
      for(decltype(comm_size) r = 0; r < comm_size; ++r)
        senddispls[r + 1] = senddispls[r] + sendcounts[r];

      std::vector<int_t> send_indices(senddispls[comm_size]);
      std::fill(sendcounts.begin(), sendcounts.end(), 0);
      
      std::vector<int_t> recvloc(tot_indices);
      for (size_t i=0; i<tot_indices; ++i) {
        auto r = index_owners[i];
        auto pos = senddispls[r] + sendcounts[r];
        send_indices[pos] = part_indices[i];
        sendcounts[r]++;
        recvloc[pos] = i;
      }

      auto mpi_int_t = librtmpi::typetraits<int_t>::type();
      std::vector<int_t> recvcounts(comm_size, 0);

      auto ret = MPI_Alltoall(
          sendcounts.data(),
          1,
          mpi_int_t,
          recvcounts.data(),
          1,
          mpi_int_t,
          MPI_COMM_WORLD);
      MpiRuntime.check(ret);
      
      std::vector<int_t> recvdispls(comm_size+1);
      recvdispls[0] = 0;
      for(decltype(comm_size) r = 0; r < comm_size; ++r)
        recvdispls[r + 1] = recvdispls[r] + recvcounts[r];

      std::vector<int_t> recv_indices(recvdispls[comm_size]);
      ret = librtmpi::alltoallv(
          send_indices,
          sendcounts,
          senddispls,
          recv_indices,
          recvcounts,
          recvdispls,
          MPI_COMM_WORLD);
      MpiRuntime.check(ret);

      std::swap(send_indices, recv_indices);
      std::swap(sendcounts, recvcounts);
      std::swap(senddispls, recvdispls);
      
      // now exchange field data
      auto field_data = static_cast<byte_t*>(fld->data);
      auto data_size = fld->data_size;
      
      auto & Request = MpiRuntime.requestField(
          field_data,
          senddispls[comm_size] * data_size,
          recvdispls[comm_size] * data_size,
          2*comm_size);
      
      std::swap(Request.Locations, recvloc);

      auto sendbuf = static_cast<byte_t*>(Request.getBuffer(0));
      auto recvbuf = static_cast<byte_t*>(Request.getBuffer(1));
      auto & requests = Request.getRequests();
      
      int tag = 0;
      auto mpi_byte_t = librtmpi::typetraits<byte_t>::type();

      size_t recvcnt = 0;
      for (decltype(comm_size) i=0; i<comm_size; ++i) {
        auto count = recvcounts[i] * data_size;
        if(count > 0) {
          auto buf = &recvbuf[recvcnt];
          requests.emplace_back();
          auto & my_request = requests.back();
          auto ret = MPI_Irecv(buf, count, mpi_byte_t, i, tag, MPI_COMM_WORLD, &my_request);
          MpiRuntime.check(ret);
        recvcnt += count;
        }
      }
      
      auto field_id_start = fld->rank_begin(comm_rank);
      int_t sendcnt = 0;
      for (decltype(comm_size) i=0; i<comm_size; ++i) {
        auto count = sendcounts[i] * data_size;
        if(count > 0) {
          auto buf = &sendbuf[sendcnt];
          auto indice_start = senddispls[i];
          for (int_t j=0; j<sendcounts[i]; ++j) {
            auto pos = send_indices[ indice_start + j ] - field_id_start;
            pos *= data_size;
            memcpy(buf + j*data_size, field_data + pos, data_size); 
          }
          requests.emplace_back();
          auto & my_request = requests.back();
          auto ret = MPI_Isend(buf, count, mpi_byte_t, i, tag, MPI_COMM_WORLD, &my_request);
          MpiRuntime.check(ret);
          sendcnt += count;
        }
      }

    }

  }
  //----------------------------------------------------------------------------
  // Need to allocate
  else {
    auto len = fld->allocate(part, dist, comm_rank, comm_size);
    if (Field.Init && len) {
      auto fld_data  = static_cast<byte_t*>(fld->data);
      auto data_size = fld->data_size;
      for (int_t i=0; i<len; ++i)
        memcpy(fld_data + i*data_size, Field.getInit(), data_size);
    }
    MpiRuntime.incrementPartition(part->id);
  }
}

//==============================================================================
// Destroy a field
//==============================================================================
void contra_mpi_field_destroy(contra_mpi_field_t * fld)
{
  if (fld->partition) contra_mpi_partition_destroy(fld->partition);
  fld->destroy();
}

//==============================================================================
/// index space partitioning
//==============================================================================
void contra_mpi_index_space_create_from_partition(
    int_t i,
    contra_mpi_partition_t * part,
    contra_index_space_t * is)
{
  is->setup(part->offsets[i], part->offsets[i+1]);
}

//==============================================================================
/// index space creation
//==============================================================================
void contra_mpi_partition_from_index_space(
    contra_index_space_t * cs,
    contra_index_space_t * is,
    contra_mpi_partition_t * part)
{
  auto num_parts = cs->size();
  contra_mpi_partition_from_size(
      num_parts,
      is,
      part);
}

//==============================================================================
/// index space creation
//==============================================================================
void contra_mpi_partition_from_size(
    int_t num_parts,
    contra_index_space_t * is,
    contra_mpi_partition_t * part)
{
  auto index_size = is->size();
  auto chunk_size = index_size / num_parts; 
  auto remainder = index_size % num_parts;

  auto offsets = new int_t[num_parts+1];
  offsets[0] = 0;

  for (int_t i=0; i<num_parts; ++i) {
    offsets[i+1] = offsets[i] + chunk_size;
    if (remainder>0) {
      offsets[i+1]++;
      remainder--;
    }
  }

  auto pid = MpiRuntime.registerPartition();
  part->setup( index_size, num_parts, is, offsets, pid);
}

//==============================================================================
/// index space creation
//==============================================================================
void contra_mpi_partition_from_array(
    dopevector_t *arr,
    contra_index_space_t * is,
    contra_mpi_partition_t * part)
{
  auto size_ptr = static_cast<const int_t*>(arr->data);
  int_t expanded_size = 0;
  int_t num_parts = arr->size;
  for (int_t i=0; i<num_parts; ++i) expanded_size += size_ptr[i];

  //------------------------------------
  // if the sizes are differeint 
  auto index_size = is->size();
  if (expanded_size != index_size) {
    std::cerr << "Index spaces partitioned by arrays MUST match the size of "
      << "the original index space." << std::endl;
    abort();  
  }
  //------------------------------------
  // Naive partitioning
  else {
 
    auto offsets = new int_t[num_parts+1];
    offsets[0] = 0;

    for (int_t i=0; i<num_parts; ++i) 
      offsets[i+1] = offsets[i] + size_ptr[i];

    auto pid = MpiRuntime.registerPartition();
    part->setup( index_size, num_parts, is, offsets, pid);

  }
  //------------------------------------
}

//==============================================================================
/// index space creation
//==============================================================================
void contra_mpi_partition_from_field(
    contra_mpi_field_t *fld,
    contra_index_space_t * is,
    contra_mpi_partition_t * fld_part,
    contra_mpi_partition_t * part)
{
  auto num_parts = fld_part->num_parts;
  
  // all ranks keep track of offsets
  auto offsets = new int_t[num_parts+1];
  offsets[0] = 0;
  for (int_t i=0; i<num_parts; ++i) {
    auto size = fld_part->size(i);
    offsets[i+1] = offsets[i] + size;
  }

  // determine  the size of my portion
  auto comm_rank = MpiRuntime.getRank();
  auto first_part = fld->distribution[comm_rank];
  auto last_part = fld->distribution[comm_rank+1];
  auto first_offset = offsets[first_part];
  auto last_offset = offsets[last_part];
 
  auto size = last_offset - first_offset;

  if (fld_part->indices) {
    std::cout << "not implemented yet" << std::endl;
    abort();
  }

  auto pid = MpiRuntime.registerPartition();
  part->setup(size, num_parts, is, fld, offsets, pid);
}

//==============================================================================
/// Accessor write
//==============================================================================
contra_mpi_partition_t*  contra_mpi_partition_get(
    contra_index_space_t * cs,
    contra_mpi_field_t * fld,
    contra_mpi_task_info_t **info)
{
  // no partitioning specified
  auto res = (*info)->getOrCreatePartition(fld->index_space);
  auto part = res.first;
  if (!res.second) {
    contra_mpi_partition_from_index_space(
        cs,
        fld->index_space,
        part);
    (*info)->register_partition(fld->index_space, part);
  }

  return part;
}

//==============================================================================
// Destroy a partition
//==============================================================================
void contra_mpi_partition_destroy(contra_mpi_partition_t * part)
{ 
  if (MpiRuntime.decrementPartition(part->id)) {
    part->destroy();
  }
}

//==============================================================================
/// Set an accessors current partition.
//==============================================================================
void contra_mpi_accessor_setup(
    int_t i,
    contra_mpi_partition_t * part,
    contra_mpi_field_t * fld,
    contra_mpi_accessor_t * acc)
{
  auto data_size = fld->data_size;
  auto comm_rank = MpiRuntime.getRank();
  
  //----------------------------------------------------------------------------
  // Partition with nidices
  if (auto part_indices = part->indices) {
    
    auto res = MpiRuntime.findFieldRequest(fld->data);
    auto & exchange_data = *res.first;
    auto & requests = exchange_data.getRequests();
    std::vector<MPI_Status> status(requests.size());
    auto ret = MPI_Waitall(requests.size(), requests.data(), status.data());
    MpiRuntime.check(ret);
    auto recvbuf = static_cast<byte_t*>(exchange_data.getBuffer(1));
    const auto & recvloc = exchange_data.Locations;

    auto rank_start = part_indices->rank_begin(comm_rank);
    auto start = part_indices->partition->offsets[i];
    auto end = part_indices->partition->offsets[i+1];
    auto size = end - start;
    auto offset = start - rank_start;
    
    acc->setup(size, data_size);
    auto acc_data = static_cast<byte_t*>(acc->data);
     
    for (int_t i=0; i<size; ++i) {
      auto dest = acc_data + i*data_size;
      auto src = recvbuf + recvloc[i + offset]*data_size;
      memcpy(dest, src, data_size); 
    }

  }
  //----------------------------------------------------------------------------
  // Regular partition
  else {
  
    auto res = MpiRuntime.findFieldRequest(fld->data);
    if (res.second) {
      auto & exchange_data = *res.first;
      auto & requests = exchange_data.getRequests();
      std::vector<MPI_Status> status(requests.size());
      auto ret = MPI_Waitall(requests.size(), requests.data(), status.data());
      MpiRuntime.check(ret);
      auto buf = exchange_data.transferBuffer();
      fld->transfer(buf);
    }
  
    auto fld_data = static_cast<byte_t*>(fld->data);

    auto pos = fld->partition->offsets[i] - fld->rank_begin(comm_rank);
    acc->setup( fld_data + data_size*pos, data_size );
  }
}

//==============================================================================
/// Accessor write
//==============================================================================
void contra_mpi_accessor_write(
    contra_mpi_accessor_t * acc,
    const void * data,
    int_t index)
{
  byte_t * offset = static_cast<byte_t*>(acc->data) + acc->data_size*index;
  memcpy(offset, data, acc->data_size);
}

//==============================================================================
/// Accessor read
//==============================================================================
void contra_mpi_accessor_read(
    contra_mpi_accessor_t * acc,
    void * data,
    int_t index)
{
  const byte_t * offset = static_cast<const byte_t*>(acc->data) + acc->data_size*index;
  memcpy(data, offset, acc->data_size);
}

//==============================================================================
/// Accessor read
//==============================================================================
void contra_mpi_accessor_destroy(
    contra_mpi_accessor_t * acc)
{ acc->destroy(); }

} // extern
