#include "src/backends/protocol/otf_message.hh"

#include <cstring>

static const std::string CONTENT_TYPE_SUFFIX = ":contentType";
static constexpr char NULL_CHAR = '\0';

namespace torchserve {
char OTFMessage::RetrieveCmd(const ISocket& client_socket_) {
  char cmd{};
  client_socket_.RetrieveBuffer(1, &cmd);
  return cmd;
}

void OTFMessage::EncodeLoadModelResponse(
    std::unique_ptr<torchserve::LoadModelResponse> response,
    std::vector<char>& data_buffer) {
  // response code
  int32_t response_code = htonl(response->code);
  AppendIntegerToCharVector(data_buffer, response_code);
  // response message
  AppendOTFStringToCharVector(data_buffer, response->buf);
  // end of message
  int32_t end_of_response_code = htonl(-1);
  AppendIntegerToCharVector(data_buffer, end_of_response_code);
}

bool OTFMessage::SendLoadModelResponse(
    const ISocket& client_socket_,
    std::unique_ptr<torchserve::LoadModelResponse> response) {
  std::vector<char> data_buffer = {};
  torchserve::OTFMessage::EncodeLoadModelResponse(std::move(response),
                                                  data_buffer);
  return client_socket_.SendAll(data_buffer.size(), data_buffer.data());
}

std::shared_ptr<LoadModelRequest> OTFMessage::RetrieveLoadMsg(
    const ISocket& client_socket_) {
  /**
   * @brief
   * MSG Frame Format:
   * | cmd value |
   * | int model-name length | model-name value |
   * | int model-path length | model-path value |
   * | int batch-size length |
   * | int handler length | handler value |
   * | int gpu id |
   * | bool limitMaxImagePixels |
   */

  auto model_name = RetrieveStringBuffer(client_socket_, std::nullopt);
  auto model_dir = RetrieveStringBuffer(client_socket_, std::nullopt);
  auto batch_size = client_socket_.RetrieveInt();
  auto handler = RetrieveStringBuffer(client_socket_, std::nullopt);
  auto gpu_id = client_socket_.RetrieveInt();
  auto envelope = RetrieveStringBuffer(client_socket_, std::nullopt);
  auto limit_max_image_pixels = client_socket_.RetrieveBool();

  return std::make_shared<LoadModelRequest>(*model_dir, *model_name, gpu_id,
                                            *handler, *envelope, batch_size,
                                            limit_max_image_pixels);
}

std::shared_ptr<torchserve::InferenceRequestBatch>
OTFMessage::RetrieveInferenceMsg(const ISocket& client_socket_) {
  auto inference_requests =
      std::make_shared<InferenceRequestBatch>(InferenceRequestBatch{});

  while (true) {
    auto inference_request = RetrieveInferenceRequest(client_socket_);
    if (inference_request == nullptr) {
      break;
    }

    inference_requests->push_back(std::move(*inference_request));
  }

  return inference_requests;
}

std::shared_ptr<InferenceRequest> OTFMessage::RetrieveInferenceRequest(
    const ISocket& client_socket_) {
  // fetch request id
  int length = client_socket_.RetrieveInt();
  if (length == -1) {
    return nullptr;
  }

  auto request_id =
      RetrieveStringBuffer(client_socket_, std::make_optional(length));

  // fetch headers
  InferenceRequest::Headers headers{};
  bool is_valid = true;
  while (true) {
    RetrieveInferenceRequestHeader(client_socket_, headers, is_valid);
    if (!is_valid) {
      break;
    }
  }

  // use default data_type of bytes for now
  // TODO: handle data_type more broadly once backend support is added
  // This needs to be set based on some parameter from the frontend
  // And requires changes to the frontend and python backend
  headers[torchserve::PayloadType::kHEADER_NAME_BODY_TYPE] =
      torchserve::PayloadType::kDATA_TYPE_BYTES;

  // fetch parameters
  InferenceRequest::Parameters parameters{};
  is_valid = true;
  while (true) {
    RetrieveInferenceRequestParameter(client_socket_, headers, parameters,
                                      is_valid);
    if (!is_valid) {
      break;
    }
  }

  return std::make_shared<InferenceRequest>(*request_id, headers, parameters);
}

void OTFMessage::RetrieveInferenceRequestHeader(
    const ISocket& client_socket_,
    InferenceRequest::Headers& inference_request_headers, bool& is_valid) {
  int length = client_socket_.RetrieveInt();

  if (length == -1) {
    is_valid = false;
    return;
  }

  auto header_name =
      RetrieveStringBuffer(client_socket_, std::make_optional(length));
  auto header_value = RetrieveStringBuffer(client_socket_, std::nullopt);
  inference_request_headers[*header_name] = *header_value;
}

void OTFMessage::RetrieveInferenceRequestParameter(
    const ISocket& client_socket_,
    InferenceRequest::Headers& inference_request_headers,
    InferenceRequest::Parameters& inference_request_parameters,
    bool& is_valid) {
  auto length = client_socket_.RetrieveInt();
  if (length == -1) {
    is_valid = false;
    return;
  }

  auto parameter_name =
      RetrieveStringBuffer(client_socket_, std::make_optional(length));
  auto content_type = RetrieveStringBuffer(client_socket_, std::nullopt);

  length = client_socket_.RetrieveInt();
  std::vector<char> value(length);
  client_socket_.RetrieveBuffer(length, value.data());

  inference_request_parameters[*parameter_name] = value;
  inference_request_headers[*parameter_name + CONTENT_TYPE_SUFFIX] =
      *content_type;
}

bool OTFMessage::SendInferenceResponse(
    const ISocket& client_socket_,
    std::shared_ptr<InferenceResponseBatch>& inference_response_batch) {
  std::vector<char> data_buffer = {};
  OTFMessage::EncodeInferenceResponse(inference_response_batch, data_buffer);
  return client_socket_.SendAll(data_buffer.size(), data_buffer.data());
}

void OTFMessage::EncodeInferenceResponse(
    std::shared_ptr<InferenceResponseBatch>& inference_response_batch,
    std::vector<char>& data_buffer) {
  // frontend decoder -
  // https://github.com/pytorch/serve/blob/a4a553a1d77668310e74141f4efabdc7713d77f4/frontend/server/src/main/java/org/pytorch/serve/util/codec/ModelResponseDecoder.java#L20

  auto batch_response_status =
      std::make_pair(200, std::string("Prediction success"));
  for (auto const& [request_id, inference_response] :
       *inference_response_batch) {
    if (inference_response->code != 200) {
      batch_response_status = std::make_pair(
          inference_response->code,
          torchserve::Converter::VectorToStr(inference_response->msg));
      break;
    }
  }

  // status code
  int32_t code = htonl(batch_response_status.first);
  AppendIntegerToCharVector(data_buffer, code);

  // message
  AppendOTFStringToCharVector(data_buffer, batch_response_status.second);

  // for each response in the batch
  for (auto const& [request_id, inference_response] :
       *inference_response_batch) {
    // request id
    AppendOTFStringToCharVector(data_buffer, request_id);

    // content type - leaving it empty to be backward compatible. It will be
    // passed in headers if added there.
    int32_t message_size = htonl(0);
    AppendIntegerToCharVector(data_buffer, message_size);

    // status code
    int32_t status_code = htonl(inference_response->code);
    AppendIntegerToCharVector(data_buffer, status_code);

    // reason phrase - leaving it empty to be backward compatible. It will be
    // passed in headers if added there.
    int32_t reason_phrase_size = htonl(0);
    AppendIntegerToCharVector(data_buffer, reason_phrase_size);

    // headers
    int32_t headers_count = htonl(inference_response->headers.size());
    AppendIntegerToCharVector(data_buffer, headers_count);
    for (auto const& [header_name, header_value] :
         inference_response->headers) {
      AppendOTFStringToCharVector(data_buffer, header_name);
      AppendOTFStringToCharVector(data_buffer, header_value);
    }

    // response message
    int32_t msg_size = htonl(inference_response->msg.size());
    AppendIntegerToCharVector(data_buffer, msg_size);
    data_buffer.insert(data_buffer.end(), inference_response->msg.begin(),
                       inference_response->msg.end());
  }
  int32_t end_of_response_code = htonl(-1);
  AppendIntegerToCharVector(data_buffer, end_of_response_code);
}

std::shared_ptr<std::string> OTFMessage::RetrieveStringBuffer(
    const ISocket& client_socket_, std::optional<int> length_opt) {
  int length = length_opt ? length_opt.value() : client_socket_.RetrieveInt();
  auto string_data = std::make_shared<std::string>(length, NULL_CHAR);
  client_socket_.RetrieveBuffer(length, string_data->data());
  return string_data;
}

void OTFMessage::AppendIntegerToCharVector(std::vector<char>& dest_vector,
                                           const int32_t& source_integer) {
  dest_vector.resize(dest_vector.size() + sizeof(source_integer));
  memcpy(&dest_vector.back() + 1 - sizeof(source_integer), &source_integer,
         sizeof(source_integer));
}

void OTFMessage::AppendOTFStringToCharVector(std::vector<char>& dest_vector,
                                             const std::string& source_string) {
  int32_t source_string_size = htonl(source_string.size());
  AppendIntegerToCharVector(dest_vector, source_string_size);
  dest_vector.insert(dest_vector.end(), source_string.begin(),
                     source_string.end());
}

}  // namespace torchserve
