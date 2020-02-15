--- Function equivalent to basename in POSIX systems
function basename(str)
	local name = string.gsub(str, "(.*/)(.*)", "%2")
	return name
end
