Learn By Example
================

In this tutorial, many of the basic language features of Contra.  Once you finish this tutorial, see the more [advanced one](advanced.md) to learn about the powerful parallel programming features.

### Hello World

There is no predefined *main* function in Contra.  A program may be comprised of a simple
[*print*](reference.md#print) statement.  For example,
```
print("Hello World!\n")
```
simply prints
```
Hello World!
```


### Hello World Redefined

A slightly more complicated example might look like,
```
# This is a comment, and is ignored by the compiler
print("Hello World!\n")
print("Hello Again!\n")

# print a number
print("Printing the number one, %lld\n", 1)

# do some math
print("1 + 2 = \n")
1+2
```
outputs
```
Hello World!
Hello Again!
Printing the number one, 1
1 + 2 =
Ans = 3
```

### Functions

[Functions](reference.md#function) may be defined anywhere in the code, provided they are defined before they are used, and that their definition resides within the global scope.  The following program
```
print("Hello World!\n")                                                                           

function print_again()
  print("Hello World Again!\n")
end

print_again()
```
outputs
```
Hello World!
Hello World Again!
```

### Function Arguments and Return Specification

Functions may also have arguments and return a value
```
function addOne(x : i64) -> i64
  return x + 1
end

print("%lld+1 is %lld\n", 1, addOne(1))
```
outputs
```
1+1 is 2
```
The return type is specified to the right of the *->*.  Note that only one return specifier is allowed, and this must occur at the end of the function.  The argument is specified as `x : i64`, which idicates the function has one parameter whose variable name is *x* and type is *i64*.  Only two fundamental types are recognized right now, *i64* and *f64*, i.e. 64 bit integers and floating points.  Integers are signed.

### Control Flow

The next example demonstrates both [*if...then...else*](reference.md#ifthenelse) statements and [*for*](#reference.md#for-loops) loops.
```
function fizzbuzz(number : i64)
  if (number % 15 == 0) then
    print("fizzbuzz %d\n", number) 
  elif (number % 5 == 0) then
    print("buzz %d\n", number) 
  elif (number % 3 == 0) then
    print("fizz %d\n", number) 
  else
    print("%d\n", number) 
  end
end

function main()
  for i in 1 to 100 do
    fizzbuzz(i)
  end
end

main()
```
The *elif* and *else* components are optional.  There are several different ways to specify the range of the *for* loop.  They are explained in more detail in the [language reference](#reference.md#for-loops).


### Variable Declarations

Lets combine all the components that we have learned so far with a more complicated example.  The following code computes the first few numbers in the Fibonacci series. 
```
function sum( f1 : i64, f2 : i64) -> i64
  return f1 + f2
end

function fibonacci(fib_num : i64) -> i64
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

function main()
  var num_fibonacci = 7
  print("Computing the first %d Fibonacci numbers...\n", num_fibonacci)

  for i in 0 until num_fibonacci do
    var fib = fibonacci(i)
    print("Fibonacci(%d) = %d\n", i, fib)
  end
end

main()
```
outputs
```
Computing the first 7 Fibonacci numbers...
Fibonacci(0) = 0
Fibonacci(1) = 1
Fibonacci(2) = 1
Fibonacci(3) = 2
Fibonacci(4) = 3
Fibonacci(5) = 5
Fibonacci(6) = 8
```
Here we see a new feature of the language, [variable declarations](reference.md#variable-declarations).  These are the statements that begin with *var*.  In  many cases, the type is not requierd as it can be inferred from the initializer to the right of the *=* sign.  In the Contra programming language, the initializer is always required.  This prevents on from accidentally accessing uninitialized variables.

### Arrays

Arrays are the workhorse of many scientific codes.  We try to make their use as simple as possible and eliminate many of the common mistakes made with arrays.  In the following example, you will see the many different ways to initialize and index arrays.
```
function main()
  var m = 2
  var a : [i64; m] = 1
  print("Should be 1, res=%d\n", a[0])
  print("Should be 1, res=%d\n", a[1])

  var b : [f64] = [1, 2, 3.]
  print("Should be 1., res=%f\n", b[0])
  print("Should be 3., res=%f\n", b[2])

  var n = 3
  var c, d : [i64] = [1; n]
  print("Should be 1, res=%d\n", c[0])
  print("Should be 1, res=%d\n", d[2])
end

main()
```
outputs
```
Should be 1, res=1
Should be 1, res=1
Should be 1., res=1.000000
Should be 3., res=3.000000
Should be 1, res=1
Should be 1, res=1
```
As with scalar variables, an initializer is always required.  All arrays are dynamically sized, but they cannot be resized.  However, previously declared arrays may be assigned to new arrays of differing size.  The intention is to show that resizing an array is not *free* and should be limited.

Lets look at how arrays are passed to functions.  The next example
```
function change(x : [i64])
  print("x[0] is originaly 0, x=%d\n", x[0])
  print("x[1] is originaly 0, x=%d\n", x[1])
  x[0] = 1
  x[1] = 1
end

function main()
  var x : [i64] = [0; 2]
  change(x)
  print("x[0] should now be 1, x=%d\n", x[0])
  print("x[1] should now be 1, x=%d\n", x[1])
end

main()
```
outputs
```
x[0] is originaly 0, x=0
x[1] is originaly 0, x=0
x[0] should now be 1, x=1
x[1] should now be 1, x=1
```
