target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@__scratch_lds = linkonce_odr hidden addrspace(3) global [0 x i64] undef, align 8

define protected i64 addrspace(3)* @__get_scratch_lds() #0 {
  ret i64 addrspace(3)* getelementptr inbounds ([0 x i64], [0 x i64] addrspace(3)* @__scratch_lds, i64 0, i64 0)
}

attributes #0 = { alwaysinline norecurse nounwind readnone speculatable }
