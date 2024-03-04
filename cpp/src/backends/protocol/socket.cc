#include "socket.hh"

#include <unistd.h>
#include <cstring>

namespace torchserve {
Socket::~Socket() {
  if (client_socket_ >= 0) {
    close(client_socket_);
  }
}

bool Socket::SendAll(size_t length, char* data) const {
  char* pkt = data;
  while (length > 0) {
    ssize_t pkt_size = send(client_socket_, pkt, length, 0);
    if (pkt_size < 0) {
      TS_LOGF(INFO, "Error sending data to socket. errno: ", errno);
      return false;
    }
    pkt += pkt_size;
    length -= pkt_size;
  }
  return true;
};

void Socket::RetrieveBuffer(size_t length, char* data) const {
  char* pkt = data;
  while (length > 0) {
    ssize_t pkt_size = recv(client_socket_, pkt, length, 0);
    if (pkt_size == 0) {
      TS_LOG(INFO, "Frontend disconnected.");
      close(client_socket_);
      exit(0);
    }
    if (pkt_size < 0) {
      TS_LOGF(FATAL, "Error recieving data from socket. errno: ", errno);
    }
    pkt += pkt_size;
    length -= pkt_size;
  }
};

int Socket::RetrieveInt() const {
  // TODO: check network - host byte-order is correct: ntohl() and htonl()
  // <arpa/inet.h>
  std::array<char, INT_STD_SIZE> int_buffer{};
  int value{};
  RetrieveBuffer(INT_STD_SIZE, int_buffer.data());
  std::memcpy(&value, int_buffer.data(), INT_STD_SIZE);
  return ntohl(value);
};

bool Socket::RetrieveBool() const {
  std::array<char, BOOL_STD_SIZE> bool_buffer{};
  bool value{};
  RetrieveBuffer(BOOL_STD_SIZE, bool_buffer.data());
  std::memcpy(&value, bool_buffer.data(), BOOL_STD_SIZE);
  return value;
};
}  // namespace torchserve
