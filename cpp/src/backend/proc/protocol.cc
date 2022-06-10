#include "protocol.hh"

namespace torchserve {
std::pair<char, void*> RetrieveMsg(Socket conn) {
    char* data;
    RetrieveBuffer(conn, 1, data);
    char cmd = data[0];
    void* msg;
    if (cmd == LOAD_MSG)
        msg = RetrieveLoadMsg(conn);
    else if (cmd == PREDICT_MSG) {
        RetrieveInferenceMsg(conn);
        std::time_t end_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        LOG(INFO) << "Backend received inference at: " << std::ctime(&end_time);
    } else
        LOG(ERROR) << "Invalid command: " << cmd;
    return std::make_pair(cmd, msg);
}

byte_buffer CreateLoadModelResponse(StatusCode code, const std::string& message) {
    LoadModelResponse response = {
            code, static_cast<int>(message.length()), message
    };
    char msg[sizeof(LoadModelResponse)];
    //std::memcpy(msg, &response, sizeof(LoadModelResponse));
}

void RetrieveBuffer(Socket conn, size_t length, char *data) {
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

int RetrieveInt(Socket conn) {
    // TODO: check network - host byte-order is correct: ntohl() and htonl() <arpa/inet.h>
    char data[INT_STD_SIZE];
    int value;
    RetrieveBuffer(conn, INT_STD_SIZE, data);
    std::memcpy(&value, data, INT_STD_SIZE);
    return ntohl(value);
}

bool RetrieveBool(Socket conn) {
    char data[BOOL_STD_SIZE];
    bool value;
    RetrieveBuffer(conn, BOOL_STD_SIZE, data);
    std::memcpy(&value, data, BOOL_STD_SIZE);
    return value;
}

LoadModelRequest* RetrieveLoadMsg(Socket conn) {
    /*
     *  MSG Frame Format:
        | cmd value |
        | int model-name length | model-name value |
        | int model-path length | model-path value |
        | int batch-size length |
        | int handler length | handler value |
        | int gpu id |
        | bool limitMaxImagePixels |
     */
    int length;
    char* data;

    // Model Name
    length = torchserve::RetrieveInt(conn);
    data = new char[length];
    torchserve::RetrieveBuffer(conn, length, data);
    std::string model_name(data, length);
    delete[] data;

    // Model Path
    length = torchserve::RetrieveInt(conn);
    data = new char[length];
    torchserve::RetrieveBuffer(conn, length, data);
    std::string model_path(data, length);
    delete[] data;

    // Batch Size
    auto batch_size = torchserve::RetrieveInt(conn);

    // Handler Name (Not used)
    length = torchserve::RetrieveInt(conn);
    data = new char[length];
    torchserve::RetrieveBuffer(conn, length, data);
    std::string handler(data, length);
    delete[] data;
    LOG(INFO) << "Received handler in message, will be ignored: " << handler;

    // GPU ID
    auto gpu_id = torchserve::RetrieveInt(conn);

    // Envelope
    length = torchserve::RetrieveInt(conn);
    data = new char[length];
    torchserve::RetrieveBuffer(conn, length, data);
    std::string envelope(data, length);
    delete[] data;

    // Limit max image pixels
    auto limit_max_image_pixels = torchserve::RetrieveBool(conn);

    auto *load_request_model = new LoadModelRequest{model_path, model_name, gpu_id, handler, envelope, batch_size, limit_max_image_pixels};
    LOG(INFO) << "Model Name: " << load_request_model->model_name;
    LOG(INFO) << "Model path: " << load_request_model->model_path;
    LOG(INFO) << "Batch size: " << load_request_model->batch_size;
    LOG(INFO) << "Handler: " << load_request_model->handler;
    LOG(INFO) << "GPU_id: " << load_request_model->gpu_id;
    LOG(INFO) << "Envelope: " << load_request_model->envelope;
    LOG(INFO) << "Limit max image pixels: " << load_request_model->limit_max_image_pixels;
    LOG(INFO) << load_request_model;
    return load_request_model;
}

void RetrieveInferenceMsg(Socket conn) {

}
} //namespace torchserve