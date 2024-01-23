#include "gmock/gmock.h"
#include "src/backends/protocol/otf_message.hh"
#include "src/utils/logging.hh"

namespace torchserve {
class MockSocket : public ISocket {
 public:
  MOCK_METHOD(bool, SendAll, (size_t, char*), (const, override));
  MOCK_METHOD(int, RetrieveInt, (), (const, override));
  MOCK_METHOD(bool, RetrieveBool, (), (const, override));
  MOCK_METHOD(void, RetrieveBuffer, (size_t, char*), (const, override));
};
}  // namespace torchserve