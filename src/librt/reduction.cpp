#include "reduction.hpp"

#include <algorithm>

extern "C" {

//==============================================================================
// Reduction ops for real
//==============================================================================

//--------------------------------------
// Sum
void contra_sum_apply_real(real_t * lhs, real_t * rhs, bool * exclusive)
{
  if (*exclusive) {
    (*lhs) += (*rhs);
  }
  else {
    volatile long long *target = (volatile long long *)lhs;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float + (*rhs);
    }
    while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
  }
}
void contra_sum_fold_real(real_t * rhs1, real_t * rhs2, bool * exclusive)
{
  if (*exclusive) {
    (*rhs1) += (*rhs2);
  }
  else {
    volatile long long *target = (volatile long long *)rhs1;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float + (*rhs2);
    }
    while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
  }
}

//--------------------------------------
// Sub
void contra_sub_apply_real(real_t * lhs, real_t * rhs, bool * exclusive)
{
  if (*exclusive) {
    (*lhs) -= (*rhs);
  }
  else {
    volatile long long *target = (volatile long long *)lhs;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float - (*rhs);
    }
    while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
  }
}
void contra_sub_fold_real(real_t * rhs1, real_t * rhs2, bool * exclusive)
{
  if (*exclusive) {
    (*rhs1) += (*rhs2);
  }
  else {
    volatile long long *target = (volatile long long *)rhs1;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float + (*rhs2);
    }
    while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
  }
}

//--------------------------------------
// Mul
void contra_mul_apply_real(real_t * lhs, real_t * rhs, bool * exclusive)
{
  if (*exclusive) {
    (*lhs) *= (*rhs);
  }
  else {
    volatile long long *target = (volatile long long *)lhs;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float * (*rhs);
    }
    while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
  }
}
void contra_mul_fold_real(real_t * rhs1, real_t * rhs2, bool * exclusive)
{
  if (*exclusive) {
    (*rhs1) *= (*rhs2);
  }
  else {
    volatile long long *target = (volatile long long *)rhs1;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float * (*rhs2);
    }
    while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
  }
}

//--------------------------------------
// Div
void contra_div_apply_real(real_t * lhs, real_t * rhs, bool * exclusive)
{
  if (*exclusive) {
    (*lhs) /= (*rhs);
  }
  else {
    volatile long long *target = (volatile long long *)lhs;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float / (*rhs);
    }
    while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
  }
}
void contra_div_fold_real(real_t * rhs1, real_t * rhs2, bool * exclusive)
{
  if (*exclusive) {
    (*rhs1) *= (*rhs2);
  }
  else {
    volatile long long *target = (volatile long long *)rhs1;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float * (*rhs2);
    }
    while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
  }
}

//--------------------------------------
// Min
void contra_min_apply_real(real_t * lhs, real_t * rhs, bool * exclusive)
{
  if (*exclusive) {
    if ((*rhs) < (*lhs)) (*lhs) = (*rhs);
  }
  else {
    volatile long long *target = (volatile long long *)lhs;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = std::min(oldval.as_float, *rhs);
      if (newval.as_float == oldval.as_float)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));

  }
}
void contra_min_fold_real(real_t * rhs1, real_t * rhs2, bool * exclusive)
{
  if (*exclusive) {
    if ((*rhs2) < (*rhs1)) (*rhs1) = (*rhs2);
  }
  else {
    volatile long long *target = (volatile long long *)rhs1;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = std::min(oldval.as_float, *rhs2);
      if (newval.as_float == oldval.as_float)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));

  }
}

//--------------------------------------
// Max
void contra_max_apply_real(real_t * lhs, real_t * rhs, bool * exclusive)
{
  if (*exclusive) {
    if ((*rhs) > (*lhs)) (*lhs) = (*rhs);
  }
  else {
    volatile long long *target = (volatile long long *)lhs;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = std::max(oldval.as_float, *rhs);
      if (newval.as_float == oldval.as_float)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));

  }
}
void contra_max_fold_real(real_t * rhs1, real_t * rhs2, bool * exclusive)
{
  if (*exclusive) {
    if ((*rhs2) > (*rhs1)) (*rhs1) = (*rhs2);
  }
  else {
    volatile long long *target = (volatile long long *)rhs1;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = std::max(oldval.as_float, *rhs2);
      if (newval.as_float == oldval.as_float)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));

  }
}

//==============================================================================
// Reduction ops for int
//==============================================================================

//--------------------------------------
// Sum
void contra_sum_apply_int(int_t * lhs, int_t * rhs, bool * exclusive)
{
  if (*exclusive)
    (*lhs) += (*rhs);
  else
    __sync_fetch_and_add(lhs, *rhs);
}
void contra_sum_fold_int(int_t * rhs1, int_t * rhs2, bool * exclusive)
{
  if (*exclusive)
    (*rhs1) += (*rhs2);
  else
    __sync_fetch_and_add(rhs1, *rhs2);
}

//--------------------------------------
// Sub
void contra_sub_apply_int(int_t * lhs, int_t * rhs, bool * exclusive)
{
  if (*exclusive)
    (*lhs) -= (*rhs);
  else
    __sync_fetch_and_sub(lhs, *rhs);
}
void contra_sub_fold_int(int_t * rhs1, int_t * rhs2, bool * exclusive)
{
  if (*exclusive)
    (*rhs1) += (*rhs2);
  else
    __sync_fetch_and_add(rhs1, *rhs2);
}

//--------------------------------------
// Mul
void contra_mul_apply_int(int_t * lhs, int_t * rhs, bool * exclusive)
{
  if (*exclusive) {
    (*lhs) *= (*rhs);
  }
  else {
    volatile long long *target = (volatile long long *)lhs;
    long long oldval, newval;
    do {
      oldval = *target;
      newval = oldval * (*rhs);
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
  }
}
void contra_mul_fold_int(int_t * rhs1, int_t * rhs2, bool * exclusive)
{
  if (*exclusive) {
    (*rhs1) *= (*rhs2);
  }
  else {
    volatile long long *target = (volatile long long *)rhs1;
    long long oldval, newval;
    do {
      oldval = *target;
      newval = oldval * (*rhs2);
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
  }
}

//--------------------------------------
// Div
void contra_div_apply_int(int_t * lhs, int_t * rhs, bool * exclusive)
{
  if (*exclusive) {
    (*lhs) /= (*rhs);
  }
  else {
    volatile long long *target = (volatile long long *)lhs;
    long long oldval, newval;
    do {
      oldval = *target;
      newval = oldval / (*rhs);
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
  }
}
void contra_div_fold_int(int_t * rhs1, int_t * rhs2, bool * exclusive)
{
  if (*exclusive) {
    (*rhs1) *= (*rhs2);
  }
  else {
    volatile long long *target = (volatile long long *)rhs1;
    long long oldval, newval;
    do {
      oldval = *target;
      newval = oldval * (*rhs2);
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
  }
}

//--------------------------------------
// Min
void contra_min_apply_int(int_t * lhs, int_t * rhs, bool * exclusive)
{
  if (*exclusive) {
    if ((*rhs) < (*lhs)) (*lhs) = (*rhs);
  }
  else {
    volatile long long *target = (volatile long long *)lhs;
    long long oldval, newval;
    do {
      oldval = *target;
      newval = std::min<long long>(oldval, *rhs);
      if (newval == oldval) break;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
  }
}
void contra_min_fold_int(int_t * rhs1, int_t * rhs2, bool * exclusive)
{
  if (*exclusive) {
    if ((*rhs2) < (*rhs1)) (*rhs1) = (*rhs2);
  }
  else {
    volatile long long *target = (volatile long long *)rhs1;
    long long oldval, newval;
    do {
      oldval = *target;
      newval = std::min<long long>(oldval, *rhs2);
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
  }
}

//--------------------------------------
// Max
void contra_max_apply_int(int_t * lhs, int_t * rhs, bool * exclusive)
{
  if (*exclusive) {
    if ((*rhs) > (*lhs)) (*lhs) = (*rhs);
  }
  else {
    volatile long long *target = (volatile long long *)lhs;
    long long oldval, newval;
    do {
      oldval = *target;
      newval = std::max<long long>(oldval, *rhs);
      if (newval == oldval) break;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
  }
}
void contra_max_fold_int(int_t * rhs1, int_t * rhs2, bool * exclusive)
{
  if (*exclusive) {
    if ((*rhs2) > (*rhs1)) (*rhs1) = (*rhs2);
  }
  else {
    volatile long long *target = (volatile long long *)rhs1;
    long long oldval, newval;
    do {
      oldval = *target;
      newval = std::max<long long>(oldval, *rhs2);
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
  }
}
} // extern
