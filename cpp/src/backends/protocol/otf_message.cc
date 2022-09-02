#include "src/backends/protocol/otf_message.hh"

const std::string CONTENT_TYPE_SUFFIX = ":contentType";

namespace torchserve {
  void OTFMessage::EncodeLoadModelResponse(std::unique_ptr<torchserve::LoadModelResponse> response, char* data) {
    char* p = data;
    // Serialize response code
    int32_t s_code = htonl(response->code);
    memcpy(p, &s_code, sizeof(s_code));
    p += sizeof(s_code);
    // Serialize response message length
    int32_t resp_length = htonl(response->length);
    memcpy(p, &resp_length, sizeof(resp_length));
    p += sizeof(resp_length);
    // Serialize response message
    strcpy(p, response->buf.c_str());
    p += response->length;
    // Expectation from frontend deserializer is a -1
    // at the end of a LoadModelResponse
    int32_t no_predict = htonl(response->predictions);
    memcpy(p, &no_predict, sizeof(no_predict));
    p += sizeof(no_predict);
  }

  bool OTFMessage::SendLoadModelResponse(const ISocket& client_socket_, std::unique_ptr<torchserve::LoadModelResponse> response) {
    char *data = new char[sizeof(LoadModelResponse)];
    torchserve::OTFMessage::EncodeLoadModelResponse(std::move(response), data);
    bool status = client_socket_.SendAll(sizeof(LoadModelResponse), data);
    delete[] data;
    return status;
  }

  char OTFMessage::RetrieveCmd(const ISocket& client_socket_) {
    char* data = new char[1];
    client_socket_.RetrieveBuffer(1, data);
    return data[0];
  }

  std::shared_ptr<LoadModelRequest> OTFMessage::RetrieveLoadMsg(const ISocket& client_socket_) {
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
    int length;
    char* data;

    // Model Name
    length = client_socket_.RetrieveInt();
    data = new char[length];
    client_socket_.RetrieveBuffer(length, data);
    std::string model_name(data, length);
    delete[] data;

    // Model Path
    length = client_socket_.RetrieveInt();
    data = new char[length];
    client_socket_.RetrieveBuffer(length, data);
    std::string model_dir(data, length);
    delete[] data;

    // Batch Size
    auto batch_size = client_socket_.RetrieveInt();

    // Handler Name (Not used)
    length = client_socket_.RetrieveInt();
    data = new char[length];
    client_socket_.RetrieveBuffer(length, data);
    std::string handler(data, length);
    delete[] data;
    TS_LOGF(INFO, "Received handler in message, will be ignored: {}", handler);

    // GPU ID
    auto gpu_id = client_socket_.RetrieveInt();

    // Envelope
    length = client_socket_.RetrieveInt();
    data = new char[length];
    client_socket_.RetrieveBuffer(length, data);
    std::string envelope(data, length);
    delete[] data;

    // Limit max image pixels
    auto limit_max_image_pixels = client_socket_.RetrieveBool();

    TS_LOGF(DEBUG, "Model Name: {}", model_name);
    TS_LOGF(DEBUG, "Model dir: {}", model_dir);
    TS_LOGF(DEBUG, "Batch size: {}", batch_size);
    TS_LOGF(DEBUG, "Handler: {}", handler);
    TS_LOGF(DEBUG, "GPU_id: {}", gpu_id);
    TS_LOGF(DEBUG, "Envelope: {}", envelope);
    TS_LOGF(DEBUG, "Limit max image pixels: {}", limit_max_image_pixels);

    return std::make_shared<LoadModelRequest>(
      model_dir, model_name, gpu_id, handler,
      envelope, batch_size, limit_max_image_pixels);
  }

  std::shared_ptr<torchserve::InferenceRequestBatch> OTFMessage::RetrieveInferenceMsg(Socket conn) {
    std::shared_ptr<torchserve::InferenceRequestBatch> inference_requests(new InferenceRequestBatch);

    while (true)
    {
      auto inference_request = RetrieveInferenceRequest(conn);
      if (inference_request == nullptr) {
        break;
      }

      // TODO: use move?
      inference_requests->push_back(*inference_request);
    }

    return inference_requests;
  }

  std::shared_ptr<InferenceRequest> OTFMessage::RetrieveInferenceRequest(Socket conn) {
    int length;
    // fetch request id
    length = RetrieveInt(conn);
    if (length == -1) {
      return nullptr;
    }

    auto request_id = RetrieveStringBuffer(conn, length);

    // fetch headers
    InferenceRequest::Headers headers{};
    bool is_valid = true;
    while (true) {
      RetrieveInferenceRequestHeader(conn, headers, is_valid);
      if (!is_valid) {
        break;
      }
    }

    // fetch parameters
    InferenceRequest::Parameters parameters{};
    is_valid = true;
    while (true) {
      RetrieveInferenceRequestParameter(conn, headers, parameters, is_valid);
      if (!is_valid) {
        break;
      }
    }

    return std::make_shared<InferenceRequest>(*request_id, headers, parameters);
  }

  void OTFMessage::RetrieveInferenceRequestParameter(Socket conn, InferenceRequest::Headers& inference_request_headers,
                                                  InferenceRequest::Parameters& inference_request_parameters, bool& is_valid) {
    int length;
    char* data;

    length = RetrieveInt(conn);
    if (length == -1) {
      is_valid = false;
      return;
    }

    auto parameter_name = RetrieveStringBuffer(conn, length);

    length = RetrieveInt(conn);
    auto content_type = RetrieveStringBuffer(conn, length);

    length = RetrieveInt(conn);
    data = new char[length];
    RetrieveBuffer(conn, length, data);
    std::vector<char> value(data, data + length);
    delete[] data;

    inference_request_parameters[*parameter_name] = value;
    inference_request_headers[*parameter_name + CONTENT_TYPE_SUFFIX] = *content_type;
  }

  void OTFMessage::RetrieveInferenceRequestHeader(Socket conn, InferenceRequest::Headers& inference_request_headers, bool& is_valid) {
    int length;

    length = RetrieveInt(conn);

    if (length == -1) {
      is_valid = false;
      return;
    }

    auto header_name = RetrieveStringBuffer(conn, length);

    length = RetrieveInt(conn);
    auto header_value = RetrieveStringBuffer(conn, length);

    inference_request_headers[*header_name] = *header_value;
  }

  bool OTFMessage::SendInferenceResponse(Socket client_socket_, std::shared_ptr<InferenceResponseBatch> inference_response_batch) {
    std::vector<char> data_buffer = {};
    OTFMessage::EncodeInferenceResponse(inference_response_batch, data_buffer);
    return OTFMessage::SendAll(client_socket_, data_buffer.data(), data_buffer.size());
  }

  void OTFMessage::EncodeInferenceResponse(std::shared_ptr<InferenceResponseBatch> inference_response_batch, std::vector<char>& data_buffer) {
    // frontend decoder - https://github.com/pytorch/serve/blob/master/frontend/server/src/main/java/org/pytorch/serve/util/codec/ModelResponseDecoder.java#L20
    // status code
    // TODO: fetch failed responde code iterating through the batch if present
    int32_t code = htonl(200);
    AppendIntegerToCharVector(data_buffer, code);

    // message - leaving it empty for now.
    // TODO: model message for entire batch somehow to be backward compatible?
    int32_t message_size = htonl(0);
    AppendIntegerToCharVector(data_buffer, message_size);

    // for each response in the batch
    for(auto const& [request_id, inference_response] : *inference_response_batch) {
        // request id
        AppendOTFStringToCharVector(data_buffer, request_id);

        // content type - leaving it empty to be backward compatible. It will be passed in headers if added there.
        int32_t message_size = htonl(0);
        AppendIntegerToCharVector(data_buffer, message_size);

        // status code
        int32_t status_code = htonl(inference_response->code);
        AppendIntegerToCharVector(data_buffer, status_code);

        // reason phrase - leaving it empty to be backward compatible. It will be passed in headers if added there.
        int32_t reason_phrase_size = htonl(0);
        AppendIntegerToCharVector(data_buffer, reason_phrase_size);

        // headers
        int32_t headers_count = htonl(inference_response->headers.size());
        AppendIntegerToCharVector(data_buffer, headers_count);
        for(auto const& [header_name, header_value] : inference_response->headers) {
          AppendOTFStringToCharVector(data_buffer, header_name);
          AppendOTFStringToCharVector(data_buffer, header_value);
        }

        // response message
        int32_t msg_size = htonl(inference_response->msg.size());
        AppendIntegerToCharVector(data_buffer, msg_size);
        data_buffer.insert(data_buffer.end(), inference_response->msg.begin(), inference_response->msg.end());
    }
    int32_t end_of_response_code = htonl(-1);
    AppendIntegerToCharVector(data_buffer, end_of_response_code);
  }

  // TODO: pass length as optional argument
  std::shared_ptr<std::string> OTFMessage::RetrieveStringBuffer(Socket conn, int length) {
    char* data = new char[length];
    RetrieveBuffer(conn, length, data);
    auto string_data = std::make_shared<std::string>(data, length);
    delete[] data;
    return string_data;
  }

  void OTFMessage::AppendIntegerToCharVector(std::vector<char>& dest_vector, const int32_t& source_integer) {
    char* code_in_char = new char[sizeof(source_integer)];
    memcpy(code_in_char, &source_integer, sizeof(source_integer));
    dest_vector.insert(dest_vector.end(), code_in_char, code_in_char + sizeof(source_integer));
  }

  void OTFMessage::AppendOTFStringToCharVector(std::vector<char>& dest_vector, const std::string& source_string) {
    int32_t source_string_size = htonl(source_string.size());
    AppendIntegerToCharVector(dest_vector, source_string_size);
    dest_vector.insert(dest_vector.end(), source_string.begin(), source_string.end());
  }

} //namespace torchserve
