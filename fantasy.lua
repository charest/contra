lib f = load( "mylib.h", "mylib.so" )

function partition(offsets, sizes, global_size, num_partitions)
  
  let local_size = global_size / num_partitions
  let remainder = global_size % num_partitions

  offsets[0] = 0
  for i=0,num_parts-1 do
    offsets[i+1] = offets[i] + local_size
  end
  offsets[num_parts-1] += remainder
  
  for i = 0, num_partitions-1 do
    local_sizes[i] = offsets[i+1] - offsets[i] 
  end

end

function identify( ids, offsets )

  let tid = task_id()

  let k = 0
  let num = offsets[tid+1] - offsets[tid]
  let num_ghost = length(ghost(ids)) / 2

  for i = num_ghost,1,-1 do
    let id = offsets[tid] - i
    ids.ghost[num_ghost - i] = {id, tid-1}
    ids[k] = id
    k = k + 1
  end

  for i = 0,num_ghost do
    let id = offsets[tid] + i
    ids.shared[i] = {id, {tid-1}}
    ids[k] = id
    k = k + 1
  end

  for i = num_ghost, num-num_ghost do
    ids[k] = offsets[tid] + i
    k = k + 1
  end

  for i = num-num_ghost+1, num do
    let id = offsets[tid] + i
    ids.shared = {id, {tid+1}}
    ids[k] = id
    k = k + 1
  end

  for i = 1,num_ghost do
    let id = offsets[tid+1] - i
    ids.ghost = {id, tid+1}
    ids[k] = id
    k = k + 1
  end

end

function init(arr, val)

  for i in owned(arr) do
    arr[i] = val
  end
  
end

function sum(a, b)
  return a+b
end

function print_result(res)
  print(res)
end


function main()

   const values = 100
   const parts = 4
   const ghost = 1


  let sizes : [i64, parts] = 0
  let offsets : [i64, parts+1] = 0
  partition( offsets, sizes, values, parts )

  let ghst_sizes : [i64, parts] = ghost
  let shrd_sizes : [i64, parts] = ghost
  let excl_sizes = sizes - ghost


  -- a, b, c = array(my_vals)
  
  indices ids : [i64, excl_sizes:shrd_sizes:ghst_sizes] = -1
  identify<num_partitions>( ids, offsets )

  let a, b, c : [f64, ids] = 0
  init<num_partitions>( a, 1 )

  init<num_partitions>( b, 2 )

  cf.sum<num_partitions>( a, b, c )
  let res : sum = cf.accumulate<num_partitions>( c )
  
  print_result(res)

end


start(main)

