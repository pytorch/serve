#include "src/backends/protocol/otf_message.hh"

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
    int32_t no_predict = htonl(-1);
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
} //namespace torchserve
