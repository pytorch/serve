#pragma once

#ifndef TS_CPP_BACKEND_PROTOCOL_H
#define TS_CPP_BACKEND_PROTOCOL_H

#include <arpa/inet.h>
#include <chrono>
#include <ctime>
#include <glog/logging.h>
#include <sys/socket.h>
#include <string>
#include <vector>

namespace torchserve {
using StatusCode = int;
using Socket = int;

typedef std::vector<uint8_t> byte_buffer;

//https://docs.python.org/3/library/struct.html#format-characters
#define BOOL_STD_SIZE 1
#define INT_STD_SIZE 4
#define LOAD_MSG 'L'
#define PREDICT_MSG 'I'

#define LOG_CURRENT_TIMESTAMP() { \
    std::time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); \
    std::string timestr = std::ctime(&time); \
    return timestr; \
}

struct LoadModelRequest {
    const std::string& model_path;	// /path/to/model/file
    const std::string& model_name;
    int gpu_id;						// None if CPU else gpu_id
    const std::string& handler;		// Expected to be null for cpp backend
    const std::string& envelope;	// name of wrapper/unwrapper of request data if provided
    int batch_size;
    bool limit_max_image_pixels;	// limit pillow image max_image_pixels
};

struct LoadModelResponse {
    int code;
    int length;
    const std::string buf;
};

std::pair<char, void*> RetrieveMsg(Socket conn);
//void EncodeResponseHeaders(void resp_hdr_map);
//void CreatePredictResponse(void ret, void req_id_map, const std::string& message, StatusCode code, void context);
byte_buffer CreateLoadModelResponse(StatusCode code, const std::string& message);
void RetrieveBuffer(Socket conn, size_t length, char *data);
int RetrieveInt(Socket conn);
bool RetrieveBool(Socket conn);
LoadModelRequest* RetrieveLoadMsg(Socket conn);
void RetrieveInferenceMsg(Socket conn);
//void RetrieveRequest(Socket conn);
//void RetrieveRequestHeader(Socket conn);
//void RetrieveInputData(Socket conn);
} //namespace torchserve


#endif //TS_CPP_BACKEND_PROTOCOL_H
