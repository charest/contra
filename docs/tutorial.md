Learn By Example
================

There is no predefined *main* function in Contra.  A program may be comprised of a simple
[*print*](reference.md#print) statement.

```
print("Hello World!\n")
```


```
# This is a comment, and is ignored by the compiler
print("Hello World!\n")
print("Hello Again!\n")

# print a number
print("Printing the number one, %d\n", 1)

# do some math
print("1 + 2 = \n")
1+2
```

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
