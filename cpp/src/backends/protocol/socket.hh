#ifndef TS_CPP_BACKENDS_PROTOCOL_SOCKET_HH_
#define TS_CPP_BACKENDS_PROTOCOL_SOCKET_HH_

#include <arpa/inet.h>
#include <sys/socket.h>

#include <string>

#include "isocket.hh"
#include "src/utils/logging.hh"
#include "src/utils/message.hh"

namespace torchserve {
#define BOOL_STD_SIZE 1
#define INT_STD_SIZE 4
#define LOAD_MSG 'L'
#define PREDICT_MSG 'I'

class Socket : public ISocket {
 public:
  Socket(int client_socket) : client_socket_(client_socket) {}
  Socket(const Socket &) = delete;
  ~Socket() override;
  bool SendAll(size_t length, char *data) const override;
  void RetrieveBuffer(size_t length, char *data) const override;
  int RetrieveInt() const override;
  bool RetrieveBool() const override;

 private:
  int client_socket_;
};
}  // namespace torchserve
#endif  // TS_CPP_BACKENDS_PROTOCOL_SOCKET_HH_