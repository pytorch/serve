#include "src/utils/foo.hh"

namespace torchserve {
int Foo::add(int x, int y) { return x + y; }
}  // namespace torchserve

#if defined(__linux__) || defined(__APPLE__)
extern "C" {
torchserve::Foo *allocator() { return new torchserve::Foo(); }

void deleter(torchserve::Foo *p) { delete p; }
}
#endif
