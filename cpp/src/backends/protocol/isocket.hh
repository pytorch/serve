#ifndef TS_CPP_BACKENDS_PROTOCOL_ISOCKET_HH_
#define TS_CPP_BACKENDS_PROTOCOL_ISOCKET_HH_

#include <cstddef>

namespace torchserve {
class ISocket {
 public:
  virtual ~ISocket() {}
  virtual bool SendAll(size_t length, char *data) const = 0;
  virtual void RetrieveBuffer(size_t length, char *data) const = 0;
  virtual int RetrieveInt() const = 0;
  virtual bool RetrieveBool() const = 0;
};
}  // namespace torchserve
#endif  // TS_CPP_BACKENDS_PROTOCOL_ISOCKET_HH_
