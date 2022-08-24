#include <arpa/inet.h>
#include <sys/socket.h>

#include "src/backends/protocol/otf_message.hh"

namespace torchserve {
  bool OTFMessage::SendAll(Socket conn, char *data, size_t length) {
    char* pkt = data;
    while (length > 0) {
      ssize_t pkt_size = send(conn, pkt, length, 0);
      if (pkt_size < 0) {
        return false;
      }
      pkt += pkt_size;
      length -= pkt_size;
    }
    return true;
  }

  void OTFMessage::CreateLoadModelResponse(std::unique_ptr<torchserve::LoadModelResponse> response, char* data) {
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

  bool OTFMessage::SendLoadModelResponse(Socket client_socket_, std::unique_ptr<torchserve::LoadModelResponse> response) {
    char *data = new char[sizeof(LoadModelResponse)];
    torchserve::OTFMessage::CreateLoadModelResponse(std::move(response), data);
    if(!torchserve::OTFMessage::SendAll(client_socket_, data, sizeof(LoadModelResponse))) {
      return false;
    }
    delete[] data;
    return true;
  }

  char OTFMessage::RetrieveCmd(Socket conn) {
    char* data = new char[1];
    RetrieveBuffer(conn, 1, data);
    return data[0];
  }

  std::shared_ptr<LoadModelRequest> OTFMessage::RetrieveLoadMsg(Socket conn) {
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
    length = RetrieveInt(conn);
    data = new char[length];
    RetrieveBuffer(conn, length, data);
    std::string model_name(data, length);
    delete[] data;

    // Model Path
    length = RetrieveInt(conn);
    data = new char[length];
    RetrieveBuffer(conn, length, data);
    std::string model_dir(data, length);
    delete[] data;

    // Batch Size
    auto batch_size = RetrieveInt(conn);

    // Handler Name (Not used)
    length = RetrieveInt(conn);
    data = new char[length];
    RetrieveBuffer(conn, length, data);
    std::string handler(data, length);
    delete[] data;
    TS_LOGF(INFO, "Received handler in message, will be ignored: {}", handler);

    // GPU ID
    auto gpu_id = RetrieveInt(conn);

    // Envelope
    length = RetrieveInt(conn);
    data = new char[length];
    RetrieveBuffer(conn, length, data);
    std::string envelope(data, length);
    delete[] data;

    // Limit max image pixels
    auto limit_max_image_pixels = RetrieveBool(conn);

    TS_LOGF(INFO, "Model Name: {}", model_name);
    TS_LOGF(INFO, "Model dir: {}", model_dir);
    TS_LOGF(INFO, "Batch size: {}", batch_size);
    TS_LOGF(INFO, "Handler: {}", handler);
    TS_LOGF(INFO, "GPU_id: {}", gpu_id);
    TS_LOGF(INFO, "Envelope: {}", envelope);
    TS_LOGF(INFO, "Limit max image pixels: {}", limit_max_image_pixels);

    return std::make_shared<LoadModelRequest>(
      model_dir, model_name, gpu_id, handler, 
      envelope, batch_size, limit_max_image_pixels);
  }

  void OTFMessage::RetrieveBuffer(Socket conn, size_t length, char *data) {
    char* pkt = data;
    while (length > 0) {
      ssize_t pkt_size = recv(conn, pkt, length, 0);
      if (pkt_size == 0) {
        TS_LOG(INFO, "Frontend disconnected.");
        exit(0);
      }
      pkt += pkt_size;
      length -= pkt_size;
    }
  }

  int OTFMessage::RetrieveInt(Socket conn) {
    // TODO: check network - host byte-order is correct: ntohl() and htonl() <arpa/inet.h>
    char data[INT_STD_SIZE];
    int value;
    RetrieveBuffer(conn, INT_STD_SIZE, data);
    std::memcpy(&value, data, INT_STD_SIZE);
    return ntohl(value);
  }

  bool OTFMessage::RetrieveBool(Socket conn) {
    char data[BOOL_STD_SIZE];
    bool value;
    RetrieveBuffer(conn, BOOL_STD_SIZE, data);
    std::memcpy(&value, data, BOOL_STD_SIZE);
    return value;
  }
} //namespace torchserve
