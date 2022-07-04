#include <arpa/inet.h>
#include <sys/socket.h>
#include <glog/logging.h>

#include "src/backends/protocol/otf_message.hh"

namespace torchserve {
  byte_buffer OTFMessage::CreateLoadModelResponse(StatusCode code, const std::string& message) {
    LoadModelResponse response = {
      code, 
      static_cast<int>(message.length()), 
      message
    };
    std::byte msg[sizeof(LoadModelResponse)];
    std::memcpy(msg, &response, sizeof(LoadModelResponse));
    byte_buffer response_byte_buffer;
    std::copy(response_byte_buffer.begin(), response_byte_buffer.end(), msg);
    return response_byte_buffer;
  }

  std::pair<char, std::shared_ptr<void>> OTFMessage::RetrieveMsg(Socket conn) {
    char* data = new char[1];
    RetrieveBuffer(conn, 1, data);
    char cmd = data[0];
    std::shared_ptr<void> msg;
    if (cmd == LOAD_MSG) {
      msg = RetrieveLoadMsg(conn);
    } else if (cmd == PREDICT_MSG) {
      //TODO: call msg = RetrieveInferenceMsg(conn);
      std::time_t end_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
      LOG(INFO) << "Backend received inference at: " << std::ctime(&end_time);
    } else {
      LOG(ERROR) << "Invalid command: " << cmd;
    }
    return std::make_pair(cmd, msg);
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
    std::string model_path(data, length);
    delete[] data;

    // Batch Size
    auto batch_size = RetrieveInt(conn);

    // Handler Name (Not used)
    length = RetrieveInt(conn);
    data = new char[length];
    RetrieveBuffer(conn, length, data);
    std::string handler(data, length);
    delete[] data;
    LOG(INFO) << "Received handler in message, will be ignored: " << handler;

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

    LOG(INFO) << "Model Name: " << model_name;
    LOG(INFO) << "Model path: " << model_path;
    LOG(INFO) << "Batch size: " << batch_size;
    LOG(INFO) << "Handler: " << handler;
    LOG(INFO) << "GPU_id: " << gpu_id;
    LOG(INFO) << "Envelope: " << envelope;
    LOG(INFO) << "Limit max image pixels: " << limit_max_image_pixels;

    return std::make_shared<LoadModelRequest>(
      model_path, model_name, gpu_id, handler, 
      envelope, batch_size, limit_max_image_pixels);
  }

  void OTFMessage::RetrieveBuffer(Socket conn, size_t length, char *data) {
    char* pkt = data;
    while (length > 0) {
      ssize_t pkt_size = recv(conn, pkt, length, 0);
      if (pkt_size == 0) {
        LOG(INFO) << "Frontend disconnected.";
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