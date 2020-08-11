#ifndef CONTRA_UTILS_COMMUNICATOR_INTERFACE_HPP
#define CONTRA_UTILS_COMMUNICATOR_INTERFACE_HPP

namespace contra {

////////////////////////////////////////////////////////////////////////////////
/// Main virutal interface
////////////////////////////////////////////////////////////////////////////////
class Communicator
{
public:
  virtual ~Communicator() {}
  virtual Communicator& createCommunicator() = 0;
  virtual void init(int * argc, char ** argv[]) = 0;
  virtual void finalize() = 0;
};

////////////////////////////////////////////////////////////////////////////////
/// The empty communicator
////////////////////////////////////////////////////////////////////////////////
class NoCommunicator : public Communicator
{
public:

  static NoCommunicator & getInstance() {
    static NoCommunicator instance;
    return instance;
  }
  
  NoCommunicator(NoCommunicator const&) = delete;
  void operator=(NoCommunicator const&)   = delete;

  Communicator& createCommunicator() override {
    return NoCommunicator::getInstance();
  }

  void init(int * argc, char ** argv[]) override {}
  void finalize() override {}

private:
 
  NoCommunicator() = default;
};

////////////////////////////////////////////////////////////////////////////////
/// Static builder class
////////////////////////////////////////////////////////////////////////////////
class CommunicatorBuilder
{
  static Communicator * comm;

public:
  static Communicator& getInstance()
  {
    static Communicator& instance = comm ?
      comm->createCommunicator() : NoCommunicator::getInstance();
    return instance;
  }

  static void set(Communicator& mycomm)
  { comm = &mycomm; }
  
};

////////////////////////////////////////////////////////////////////////////////
/// Select the serial communicator
////////////////////////////////////////////////////////////////////////////////
template<typename T>
void commSelect()
{
  auto & Comm = T::getInstance();
  CommunicatorBuilder::set(Comm);
}

/// Initialize the comminicator
void commInit(int * argc, char ** argv[]);

/// Shut down the communicator
void commFinalize();


} // namespace


#endif // CONTRA_UTILS_COMMUNICATOR_INTERFACE_HPP
