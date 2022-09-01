#include "socket.hh"

namespace torchserve {
  Socket::~Socket() {
    if (client_socket_ >= 0) {
      close(client_socket_);
    }
  }

  bool Socket::SendAll(size_t length, char *data) const {
    char* pkt = data;
    while (length > 0) {
      ssize_t pkt_size = send(client_socket_, pkt, length, 0);
      if (pkt_size < 0) {
        return false;
      }
      pkt += pkt_size;
      length -= pkt_size;
    }
    return true;
  };

  void Socket::RetrieveBuffer(size_t length, char *data) const {
    char* pkt = data;
    while (length > 0) {
      ssize_t pkt_size = recv(client_socket_, pkt, length, 0);
      if (pkt_size == 0) {
        TS_LOG(INFO, "Frontend disconnected.");
        close(client_socket_);
        exit(0);
      }
      pkt += pkt_size;
      length -= pkt_size;
    }
  };

  int Socket::RetrieveInt() const {
    // TODO: check network - host byte-order is correct: ntohl() and htonl() <arpa/inet.h>
    char data[INT_STD_SIZE];
    int value;
    RetrieveBuffer(INT_STD_SIZE, data);
    std::memcpy(&value, data, INT_STD_SIZE);
    return ntohl(value);
  };

  bool Socket::RetrieveBool() const {
    char data[BOOL_STD_SIZE];
    bool value;
    RetrieveBuffer(BOOL_STD_SIZE, data);
    std::memcpy(&value, data, BOOL_STD_SIZE);
    return value;
  };
} //namespace torchserve