#ifndef TS_CPP_UTILS_FOO_HH_
#define TS_CPP_UTILS_FOO_HH_

#include "src/utils/ifoo.hh"

namespace torchserve {
class Foo : public IFoo {
 public:
  Foo() = default;
  ~Foo() override = default;

  int add(int x, int y) override;
};
}  // namespace torchserve
#endif  // TS_CPP_UTILS_FOO_HH_