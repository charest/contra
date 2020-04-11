Language Reference
==================

[Comments](#comments)

[Print](#print)



### Comments

A comment is preceded by the `#` character.  Everything after a `#` is ignored up to the end of the line.  For example,

     # This is a comment
     var i = 1 + 2 # this is also a comment

There are no special multi-line comments.  If you want to comment out an entire block of code, you must prefix each line with a `#` character.  That is

     # This is a multi-line
     # comment.
     var i = 1 + 2

### Functions

In Contra, a function is a group of statements that is given a name, and which can be called from some point of the program. The syntax to define a function is:

     function name ( parameter1, parameter2, ...) -> type 
       statements
       return statement
     end

Where:
- *type* is the type of the value returned by the function.
- *name* is the identifier by which the function can be called.
- *parameters* (as many as needed): Each parameter consists of an identifier followed by a colon and then a type, with each parameter being separated from the next by a comma. For example: `x : i64`.  These parameters are local to the function.
- *statements* is the function's body. It is a block of statements surrounded by the function signature and *end* that specify what the function actually does.

Only one *return* statement is allowed, and it must occur at the end of the function.  The return type specifier is optional.  A function that has no return type looks like

     function name ( parameter1, parameter2, ...)
       statements
     end

### Print

Writes the C string pointed by *format* to the standard output (stdout). If *format* includes format specifiers (subsequences beginning with %), the additional arguments following format are formatted and inserted in the resulting string replacing their respective specifiers.

     print( format, ... )

`print` secretly calls the `C` built-in `printf` function.  So *format* may contain any of the same format specificers.  See the manual for [*printf*](http://www.cplusplus.com/reference/cstdio/printf/).

Example:

     print("Decimals: %lld\n", 650000)
     print("Preceding with blanks: %10lld \n", 1977)
     print("Preceding with zeros: %010lld \n", 1977)
     print("floats: %4.2f %+.0e %E \n", 3.1416, 3.1416, 3.1416)
     print("%s \n", "A string")

Outputs:

     Decimals: 650000
     Preceding with blanks:       1977
     Preceding with zeros: 0000001977
     floats: 3.14 +3e+00 3.141600E+00
     A string

