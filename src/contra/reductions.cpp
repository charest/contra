#include "reductions.hpp"


namespace contra {

std::map<std::string, ReductionType> SupportedReductions::Map = 
{
  {"+", ReductionType::Add},
  {"-", ReductionType::Sub},
  {"*", ReductionType::Mult},
  {"/", ReductionType::Div},
  {"min", ReductionType::Min},
  {"max", ReductionType::Max},
};

ReductionType SupportedReductions::getType(const std::string & Name)
{ 
  auto it = Map.find(Name);
  if (it != Map.end()) return it->second;
  else return ReductionType::User;
}

}


