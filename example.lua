-- Function equivalent to basename in POSIX systems
function basename(str)
	local name = string.gsub(str, "(.*/)(.*)", "%2")
	return name
end

-- Function equivalent to dirname in POSIX systems
function dirname(str)
	if str:match(".-/.-") then
		local name = string.gsub(str, "(.*/)(.*)", "%1")
		return name
	else
		return ''
	end
end

-- smarter concatenation if a or b is empty
function concatenate( a, b )
  if ( a == nil or a == '' ) then
    return b
  elseif ( b == nil or b == '' ) then
    return a
  else
    return a .. b
  end
end

-- Main program
local prefix = "ex> "

print(prefix .. "load ffi")
local ffi = require("ffi")

print(prefix .. "cdef")
ffi.cdef[[
void task( double * ptr, int n );
]]

print(prefix .. "load lib")
local cf = ffi.load( concatenate(arg[1], "./libcf.so") )

-- Initializing the array
print(prefix .. "ffi new")
local n = 10
local array = ffi.new("double[?]", n)


print(prefix .. "init loop")
for i= 0,n-1 do
   array[i] = i
   --print(array[i])
end

print(prefix .. "calling c")
cf.task(array, n)

