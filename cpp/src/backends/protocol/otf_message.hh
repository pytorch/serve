#ifndef TS_CPP_BACKENDS_PROTOCOL_OTF_MESSAGE_HH_
#define TS_CPP_BACKENDS_PROTOCOL_OTF_MESSAGE_HH_

#include <chrono>
#include <ctime>
#include <string>
#include <variant>

#include "src/backends/protocol/socket.hh"
#include "src/utils/message.hh"
#include "src/utils/logging.hh"

namespace torchserve {
  using StatusCode = int;

  #define LOG_CURRENT_TIMESTAMP() { \
    std::time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); \
    std::string timestr = std::ctime(&time); \
    return timestr; \
  }

  class OTFMessage {
    public:
    static void EncodeLoadModelResponse(std::unique_ptr<torchserve::LoadModelResponse> response, char* data);
    static bool SendLoadModelResponse(const ISocket& client_socket_, std::unique_ptr<torchserve::LoadModelResponse> response);
    static char RetrieveCmd(const ISocket& client_socket_);
    static std::shared_ptr<LoadModelRequest> RetrieveLoadMsg(const ISocket& client_socket_);
    // TODO: impl.
    static std::shared_ptr<torchserve::InferenceRequestBatch> RetrieveInferenceMsg(ISocket& client_socket_);
  };
} // namespace torchserve
#endif // TS_CPP_BACKENDS_PROTOCOL_OTF_MESSAGE_HH_
