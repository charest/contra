Learn By Example
================

## Hello World

There is no predefined *main* function in Contra.  A program may be comprised of a simple
[*print*](reference.md#print) statement.  For example,
```
print("Hello World!\n")
```
simply prints
```
Hello World!
```


## Hello World Redefined

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

## Functions

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

## Function Arguments and Return Specification

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

## Control Flow

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
