#ifndef RTLIB_REDUCTION_HPP
#define RTLIB_REDUCTION_HPP

#include "config.hpp"

#include "dllexport.h"

extern "C" {

DLLEXPORT void contra_sum_apply_real(real_t *, real_t *, bool *);
DLLEXPORT void contra_sum_fold_real(real_t *, real_t *, bool *);
DLLEXPORT void contra_sub_apply_real(real_t *, real_t *, bool *);
DLLEXPORT void contra_sub_fold_real(real_t *, real_t *, bool *);
DLLEXPORT void contra_mul_apply_real(real_t *, real_t *, bool *);
DLLEXPORT void contra_mul_fold_real(real_t *, real_t *, bool *);
DLLEXPORT void contra_div_apply_real(real_t *, real_t *, bool *);
DLLEXPORT void contra_div_fold_real(real_t *, real_t *, bool *);
DLLEXPORT void contra_min_apply_real(real_t *, real_t *, bool *);
DLLEXPORT void contra_min_fold_real(real_t *, real_t *, bool *);
DLLEXPORT void contra_max_apply_real(real_t *, real_t *, bool *);
DLLEXPORT void contra_max_fold_real(real_t *, real_t *, bool *);

DLLEXPORT void contra_sum_apply_int(int_t *, int_t *, bool *);
DLLEXPORT void contra_sum_fold_int(int_t *, int_t *, bool *);
DLLEXPORT void contra_sub_apply_int(int_t *, int_t *, bool *);
DLLEXPORT void contra_sub_fold_int(int_t *, int_t *, bool *);
DLLEXPORT void contra_mul_apply_int(int_t *, int_t *, bool *);
DLLEXPORT void contra_mul_fold_int(int_t *, int_t *, bool *);
DLLEXPORT void contra_div_apply_int(int_t *, int_t *, bool *);
DLLEXPORT void contra_div_fold_int(int_t *, int_t *, bool *);
DLLEXPORT void contra_min_apply_int(int_t *, int_t *, bool *);
DLLEXPORT void contra_min_fold_int(int_t *, int_t *, bool *);
DLLEXPORT void contra_div_apply_int(int_t *, int_t *, bool *);
DLLEXPORT void contra_div_fold_int(int_t *, int_t *, bool *);

} // extern

#endif // RTLIB_REDUCTION_HPP
