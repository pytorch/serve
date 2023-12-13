#ifndef TS_CPP_BACKENDS_PROTOCOL_OTF_MESSAGE_HH_
#define TS_CPP_BACKENDS_PROTOCOL_OTF_MESSAGE_HH_

#include <chrono>
#include <ctime>
#include <optional>
#include <string>
#include <variant>

#include "src/backends/protocol/socket.hh"
#include "src/utils/logging.hh"
#include "src/utils/message.hh"

namespace torchserve {
using StatusCode = int;

#define LOG_CURRENT_TIMESTAMP()                              \
  {                                                          \
    std::time_t time = std::chrono::system_clock::to_time_t( \
        std::chrono::system_clock::now());                   \
    std::string timestr = std::ctime(&time);                 \
    return timestr;                                          \
  }

class OTFMessage {
 public:
  static void EncodeLoadModelResponse(
      std::unique_ptr<torchserve::LoadModelResponse> response,
      std::vector<char>& data_buffer);
  static bool SendLoadModelResponse(
      const ISocket& client_socket_,
      std::unique_ptr<torchserve::LoadModelResponse> response);
  static char RetrieveCmd(const ISocket& client_socket_);
  static std::shared_ptr<LoadModelRequest> RetrieveLoadMsg(
      const ISocket& client_socket_);
  static std::shared_ptr<torchserve::InferenceRequestBatch>
  RetrieveInferenceMsg(const ISocket& client_socket_);
  static bool SendInferenceResponse(
      const ISocket& client_socket_,
      std::shared_ptr<InferenceResponseBatch>& inference_response_batch);
  static void EncodeInferenceResponse(
      std::shared_ptr<InferenceResponseBatch>& inference_response_batch,
      std::vector<char>& data_buffer);

 private:
  static std::shared_ptr<InferenceRequest> RetrieveInferenceRequest(
      const ISocket& client_socket_);
  static void RetrieveInferenceRequestHeader(
      const ISocket& client_socket_,
      InferenceRequest::Headers& inference_request_headers, bool& is_valid);
  static void RetrieveInferenceRequestParameter(
      const ISocket& client_socket_,
      InferenceRequest::Headers& inference_request_headers,
      InferenceRequest::Parameters& inference_request_parameters,
      bool& is_valid);
  static std::shared_ptr<std::string> RetrieveStringBuffer(
      const ISocket& client_socket_, std::optional<int> length);
  static void AppendIntegerToCharVector(std::vector<char>& dest_vector,
                                        const int32_t& source_integer);
  static void AppendOTFStringToCharVector(std::vector<char>& dest_vector,
                                          const std::string& source_string);
};
}  // namespace torchserve
#endif  // TS_CPP_BACKENDS_PROTOCOL_OTF_MESSAGE_HH_
