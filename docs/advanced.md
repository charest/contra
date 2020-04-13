Parallel Programming with Contra
================================

The language features you saw in the [basic tutorial](tutorial.md) are available in almost any other programming language.  The real advantage of Contra is how easily you can write programs which execute statements in parallel.  And not only that, but also take advantage of the many different computing paradigms like Charm++, CUDA, HPX, Legion, MPI, OpenMP, etc.... 

Task Parallelism
----------------

Let's revisit the [previous example](tutorial.md#variable-declarations) that computes the first few numbers in the Fibonacci series.
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
    var result = fibonacci(i)
    print("Fibonacci(%d) = %d\n", i, result)
  end
end

top_level()
```
The above starts to hint at the real power of Contra.  We have only really made one change to the code.  We have replaced the keyword `function` with `task`.  A *task* is similar to a *function* in that it is a block of statements that performs some action.  However, it implies that the result of the statements is not needed right away.  In other worlds, it states "go off and do some work for me."  Calling a task looks identical to calling a function.  However, you have to wait for a function to finish after it is called before going on to the next statement.  Calling a task *might* return immediately before any work has been performed.  We say *might* because this really depends on the parallel backend you are using.  It might even depend on what the compiler thinks is best.

So how do I know when my *task* is finished?  It's simple!  When you access the result of your function we guarantee that the *task* is complete.  In the above example,
```
var result = fibonacci(i)
```
says "go off and compute the *i*th number in the Fibonacci series."  But `result` may not yet contain the acutal result.  In the next line we call `print` with `result` as one of these arguments.  This implicitly says "I need the value now!" and Contra will fetch it for you.  

One important thing to know is that fetching the value might not be cheap.  It might be coming from a GPU or may require a bunch of processes to synchronize and communicate with each other.  So for performance sake, you should hold off accessing the result of a task for as long as possible.  Examples of accessing the result of a task include: passing it to a regular *function*, using it in a variable initializer, or an expression like an *if*-statement condition.

One cool feature of Contra is that we will try to preserve the "need-to-knowness" of a *task* result wherever we can.  An example of this is passing the result of a *task* to another *task*. So in the `fibonacci` task, when we pass `f1` and `f2` to `sum`, i.e.,
```
res = sum(f1, f2)
```
we never fetch or block anything.   This lets individual calls to `fibonacci` possibly execute in parallel with each other.  This is called *task-parallelism*.

One of the important design features of Contra, is to ensure that debugging your code is as easy as possible.  Thats why we minimize special annotations as much as possible.  There are no special designations for *task* results.  And you don't need to do anything when passing them to regular *functions*.  But fixing bugs with *tasks* can be a challenge because you might not have explicit control of when *tasks* execute.  So the `task` keyword can simply be repplaced with `function` and you will know exactly when the *task* executes.


Data Parallism
----------------

Lets look at another example.  The following code iterates over a counter and calls a function.
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
And the result is 
```
Running hello world redux for 10 points...
Hello world from task 0, with local arg 10, and global arg 10!
Hello world from task 1, with local arg 11, and global arg 10!
...
Hello world from task 8, with local arg 18, and global arg 10!
Hello world from task 9, with local arg 19, and global arg 10!
```
Another new concept is used here.  The `foreach` statement differs from a regular `for` statement in that it says "this loop can be executed in parallel."  All variables accessed inside the *foreach* loop that were defined prior to entering the loop are captured automatically for you.  And you can call *functions* or *tasks* from within.  

It would be nice if we could just automatically detect that the loop can be parallelized, but this is really hard to do! 
