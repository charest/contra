#ifndef CONTRA_FORMATTER_HPP
#define CONTRA_FORMATTER_HPP

#include <sstream>
#include <vector>

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
    
    template<typename Type>
    Formatter & operator << (const std::vector<Type> & values)
    {
      if (!values.empty()) {
        for (unsigned i=0; i<values.size()-1; ++i) stream_ << values[i] << ", ";
        stream_ << values.back();
      }
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
