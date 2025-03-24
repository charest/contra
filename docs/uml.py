import graphviz

def uses(dot, parent, child):
    dot.edge(parent, child, style='dashed')
def inherits(dot, child, base, flip=False):
  if flip:
    dot.edge(child, base, arrowhead='empty')
  else:
    dot.edge(base, child, arrowtail='empty', dir='back')
def has(dot, parent, child, flip=False):
  if flip:
    dot.edge(child, parent, arrowhead='diamond')
  else:
    dot.edge(parent, child, arrowtail='diamond', dir='back')
def direct(dot, parent, child):
  dot.edge(parent, child)

###########################################################3
# Main

g = graphviz.Digraph('G', filename='contra.gv')
#g.attr(size='10,5.63!', ratio='fill')
#g.attr(splines='ortho')
#g.graph_attr['size'] = '10,5.63!'
#g.edge_attr['dir']='back'

uses(g, 'RecursiveAstVisiter', 'NodeAST')

has(g, 'Contra', 'BinopPrecedence')
has(g, 'Contra', 'Parser')
has(g, 'Contra', 'CodeGen')
has(g, 'Contra', 'Analyzer')
uses(g, 'Contra', 'LoopLifter')
#uses(g, 'Contra', 'FutureIdentifier')
#uses(g, 'Contra', 'LeafIdentifier')

has(g, 'Parser', 'Lexer')
direct(g, 'Parser', 'BinopPrecedence')
uses(g, 'Parser', 'NodeAST')
#uses(g, 'Parser', 'ExprAST')

inherits(g, 'Analyzer', 'RecursiveAstVisiter')
direct(g, 'Analyzer', 'BinopPrecedence')
uses(g, 'Analyzer', 'NodeAST')
#uses(g, 'Analyzer', 'ExprAST')

inherits(g, 'LoopLifter', 'RecursiveAstVisiter')
uses(g, 'LoopLifter', 'NodeAST')
#has(g, 'LoopLifter', 'FunctionAST')

#inherits(g, 'FutureIdentifier', 'RecursiveAstVisiter')
#uses(g, 'FutureIdentifier', 'ExprAST')
#has(g, 'FutureIdentifier', 'VariableDef')

#inherits(g, 'LeafIdentifier', 'RecursiveAstVisiter')
#uses(g, 'LeafIdentifier', 'ExprAST')

inherits(g, 'CodeGen', 'RecursiveAstVisiter')
#has(g, 'CodeGen', 'BuilderHelper')
has(g, 'CodeGen', 'JIT')
has(g, 'CodeGen', 'DeviceJIT')
#has(g, 'CodeGen', '{Var,Type,Func}Table')
#has(g, 'CodeGen', 'PrototypeAST')
has(g, 'CodeGen', 'AbstractTasker')
uses(g, 'CodeGen', 'NodeAST')

#has(g, 'AbstractTasker', 'BuilderHelper')
#has(g, 'AbstractTasker', 'Serializer')
#has(g, 'AbstractTasker', 'TaskInfo')

inherits(g, 'LegionTasker', 'AbstractTasker')
inherits(g, 'MpiTasker', 'AbstractTasker')
#uses(g, 'XxxTasker', 'TaskInfo')
#g.edge('AbstractTasker', 'CudaTasker', arrowhead='empty')
#g.edge('AbstractTasker', 'LegionTasker', arrowhead='empty')
#g.edge('AbstractTasker', 'MpiTasker', arrowhead='empty')
#g.edge('AbstractTasker', 'ROCmTasker', arrowhead='empty')
#g.edge('AbstractTasker', 'SerialTasker', arrowhead='empty')
#g.edge('AbstractTasker', 'ThreadsTasker', arrowhead='empty')

inherits(g, 'CudaJIT', 'DeviceJIT')
inherits(g, 'ROCmJIT', 'DeviceJIT')

#inherits(g, 'ExprAST', 'NodeAST')
uses(g, 'RecursiveAstVisiter', 'NodeAST')

g.render('contra_uml', format='png')
g.view()

###########################################################3
# legend

g = graphviz.Digraph('G', filename='legend.gv')
g.attr(rankdir='LR', sep='0', esep='0', layersep='0', nodesep='0', ranksep='1')
g.node_attr['shape']='plaintext'
g.node_attr['height']='0'
g.node_attr['width']='0'

g.node('Parent1', 'Usage')
g.node('Child1','')
uses(g, 'Parent1', 'Child1')

g.node('Derived', '')
g.node('Base', 'Inheritance')
inherits(g, 'Base', 'Derived', True)

g.node('Parent2', 'Composition')
g.node('Child2', '')
has(g, 'Child2', 'Parent2', True)

g.node('From', 'Association')
g.node('To', '')
direct(g, 'From', 'To')

g.render('contra_legend', format='png')
g.view()
