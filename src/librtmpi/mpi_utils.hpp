#ifndef CONTRA_UTILS_MPI_UTILS_HPP
#define CONTRA_UTILS_MPI_UTILS_HPP

#include <mpi.h>

namespace librtmpi {

////////////////////////////////////////////////////////////////////////////////
// Determine mpi type
////////////////////////////////////////////////////////////////////////////////
template<typename TYPE>
struct typetraits {
  static MPI_Datatype type();
};

template<>
struct typetraits<char> {
  static MPI_Datatype type() { return MPI_CHAR; }
};

template<>
struct typetraits<short> {
  static MPI_Datatype type() { return MPI_SHORT; }
};

template<>
struct typetraits<int> {
  static MPI_Datatype type() { return MPI_INT; }
};

template<>
struct typetraits<long> {
  static MPI_Datatype type() { return MPI_LONG; }
};

template<>
struct typetraits<long long> {
  static MPI_Datatype type() { return MPI_LONG_LONG; }
};

template<>
struct typetraits<signed char> {
  static MPI_Datatype type() { return MPI_SIGNED_CHAR; }
};

template<>
struct typetraits<unsigned char> {
  static MPI_Datatype type() { return MPI_UNSIGNED_CHAR; }
};

template<>
struct typetraits<unsigned short> {
  static MPI_Datatype type() { return MPI_UNSIGNED_SHORT; }
};

template<>
struct typetraits<unsigned> {
  static MPI_Datatype type() { return MPI_UNSIGNED; }
};

template<>
struct typetraits<unsigned long> {
  static MPI_Datatype type() { return MPI_UNSIGNED_LONG; }
};

template<>
struct typetraits<unsigned long long> {
  static MPI_Datatype type() { return MPI_UNSIGNED_LONG_LONG; }
};

template<>
struct typetraits<float> {
  static MPI_Datatype type() { return MPI_FLOAT; }
};

template<>
struct typetraits<double> {
  static MPI_Datatype type() { return MPI_DOUBLE; }
};

template<>
struct typetraits<long double> {
  static MPI_Datatype type() { return MPI_LONG_DOUBLE; }
};

template<>
struct typetraits<wchar_t> {
  static MPI_Datatype type() { return MPI_WCHAR; }
};


////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
template<typename SEND_TYPE, typename ID_TYPE, typename RECV_TYPE>
auto
alltoallv(const SEND_TYPE & sendbuf,
  const ID_TYPE & sendcounts,
  const ID_TYPE & senddispls,
  RECV_TYPE & recvbuf,
  const ID_TYPE & recvcounts,
  const ID_TYPE & recvdispls,
  decltype(MPI_COMM_WORLD) comm) {

  const auto mpi_send_t =
    typetraits<typename SEND_TYPE::value_type>::type();
  const auto mpi_recv_t =
    typetraits<typename RECV_TYPE::value_type>::type();

  auto num_ranks = sendcounts.size();

  // create storage for the requests
  std::vector<MPI_Request> requests;
  requests.reserve(2 * num_ranks);

  // post receives
  int tag = 0;

  for(size_t rank = 0; rank < num_ranks; ++rank) {
    auto count = recvcounts[rank];
    if(count > 0) {
      auto buf = recvbuf.data() + recvdispls[rank];
      requests.resize(requests.size() + 1);
      auto & my_request = requests.back();
      auto ret =
        MPI_Irecv(buf, count, mpi_recv_t, rank, tag, comm, &my_request);
      if(ret != MPI_SUCCESS)
        return ret;
    }
  }

  // send the data
  for(size_t rank = 0; rank < num_ranks; ++rank) {
    auto count = sendcounts[rank];
    if(count > 0) {
      auto buf = sendbuf.data() + senddispls[rank];
      requests.resize(requests.size() + 1);
      auto & my_request = requests.back();
      auto ret =
        MPI_Isend(buf, count, mpi_send_t, rank, tag, comm, &my_request);
      if(ret != MPI_SUCCESS)
        return ret;
    }
  }

  // wait for everything to complete
  std::vector<MPI_Status> status(requests.size());
  auto ret = MPI_Waitall(requests.size(), requests.data(), status.data());

  return ret;
}

} // namespace

#endif // CONTRA_UTILS_MPI_UTILS_HPP
