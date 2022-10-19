#ifndef TS_CPP_UTILS_IFOO_HH_
#define TS_CPP_UTILS_IFOO_HH_

namespace torchserve {
class IFoo {
 public:
  virtual ~IFoo() = default;

  /*
  ** Pure method which will be overrided.
  */
  virtual int add(int x, int y) = 0;
};

}  // namespace torchserve
#endif  // TS_CPP_UTILS_IFOO_HH_