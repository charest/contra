#ifndef CONTRA_FORMATTER_HPP
#define CONTRA_FORMATTER_HPP

#include <sstream>

namespace contra {

class Formatter
{
public:
    Formatter() {}
    ~Formatter() {}

    template <typename Type>
    Formatter & operator << (const Type & value)
    {
        stream_ << value;
        return *this;
    }

    std::string str() const         { return stream_.str(); }
    operator std::string () const   { return stream_.str(); }

    enum ConvertToString 
    {
        to_str
    };
    std::string operator >> (ConvertToString) { return stream_.str(); }

private:
    std::stringstream stream_;

    Formatter(const Formatter &);
    Formatter & operator = (Formatter &);
};

} // namespace

#endif // CONTRA_FORMATTER_HPP
