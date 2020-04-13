Parallel Programming with Contra
================================

The language features you saw in the [basic tutorial](tutorial.md) are available in almost any other programming language.  The real advantage of Contra is how easily you can write programs which execute statements in parallel.  And not only that, but also take advantage of the many different computing platforms like Charm++, CUDA, HPX, Legion, MPI, OpenMP, etc.... 

```
task sum( f1 : i64, f2 : i64) -> i64
  return f1 + f2
end

task fibonacci(fib_num : i64) -> i64
  var res = 0
  if (fib_num == 0) then
    res = 0
  elif (fib_num == 1) then
    res = 1
  else
    var fib1 = fib_num-1
    var f1 = fibonacci(fib1)
    var fib2 = fib_num-2
    var f2 = fibonacci(fib2)
    res = sum(f1, f2)
  end
  return res
end

task print_result(i : i64, result : i64)
  print("Fibonacci(%d) = %d\n", i, result)
end

task top_level()
  var num_fibonacci = 7
  print("Computing the first %d Fibonacci numbers...\n", num_fibonacci)
  
  for i in 0 until num_fibonacci do
    #fib_results[i] = fibonacci(i)
    print_result(i, fibonacci(i))
  end
end

top_level()
```


```
function index_space(id : i64, ilocal : i64, iglobal : i64)
  print("Hello world from task %d, with local arg %d, and global arg %d!\n",
		id, ilocal, iglobal)
end


task top_level()
  var num_points = 10
  print("Running hello world redux for %d points...\n", num_points)

  foreach i in 0 until num_points do
    index_space(i, i+10, num_points)
  end
end

top_level()
```
