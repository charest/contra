#include "precedence.hpp"

namespace contra {


//==============================================================================
// Make contra precedence
//==============================================================================
BinopPrecedence make_contra_precedence() {
  BinopPrecedence p;

  // Install standard binary operators.
  // 1 is lowest precedence.
  //Precedence_[tok_asgmt] = 2;
  p.add( tok_eq , 5 );
  p.add( tok_ne , 5 );
  p.add( tok_lt , 10);
  p.add( tok_le , 10);
  p.add( tok_gt , 10);
  p.add( tok_ge , 10);
  p.add( tok_add, 20);
  p.add( tok_sub, 20);
  p.add( tok_mul, 40);
  p.add( tok_div, 40);
  p.add( tok_mod, 40);
  // highest.
    
  return p;
}

} // namespace
