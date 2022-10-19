#ifndef TS_CPP_UTILS_MESSAGE_HH_
#define TS_CPP_UTILS_MESSAGE_HH_

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace torchserve {
class PayloadType {
 public:
  inline static const std::string kPARAMETER_NAME_DATA = "data";
  inline static const std::string kPARAMETER_NAME_BODY = "body";

  inline static const std::string kHEADER_NAME_DATA_TYPE = "data_dtype";
  inline static const std::string kHEADER_NAME_BODY_TYPE = "body_dtype";

  inline static const std::string kCONTENT_TYPE_JSON = "application/json";
  inline static const std::string kCONTENT_TYPE_TEXT = "text";
  inline static const std::string kDATA_TYPE_STRING = "string";
  inline static const std::string kDATA_TYPE_BYTES = "bytes";
};

class Converter {
 public:
  static std::vector<std::byte> StrToBytes(const std::string& str) {
    std::vector<std::byte> str_bytes;
    for (auto& ch : str) {
      str_bytes.emplace_back(std::byte(ch));
    }
    return str_bytes;
  };

  static std::vector<char> StrToVector(const std::string& str) {
    std::vector<char> vec_char(str.begin(), str.end());
    return vec_char;
  }

  static std::string VectorToStr(const std::vector<char>& vec_char) {
    std::string str(vec_char.begin(), vec_char.end());
    return str;
  }
};

// TODO: expand to support model instance, large model (ref: ModelConfig in
// config.hh)
struct LoadModelRequest {
  // /path/to/model/file
  const std::string model_dir;
  const std::string model_name;
  // Existing: -1 if CPU else gpu_id
  // TODO:
  // - support CPU core
  // - device type combine together
  int gpu_id;
  // Expected to be null for cpp backend
  const std::string handler;
  // name of wrapper/unwrapper of request data if provided
  const std::string envelope;
  int batch_size;
  // limit pillow image max_image_pixels
  bool limit_max_image_pixels;

  LoadModelRequest(const std::string& model_dir, const std::string& model_name,
                   int gpu_id, const std::string& handler,
                   const std::string& envelope, int batch_size,
                   bool limit_max_image_pixels)
      : model_dir{model_dir},
        model_name{model_name},
        gpu_id{gpu_id},
        handler{handler},
        envelope{envelope},
        batch_size{batch_size},
        limit_max_image_pixels{limit_max_image_pixels} {}

  bool operator==(const LoadModelRequest& other) {
    return this->model_dir == other.model_dir &&
           this->model_name == other.model_name &&
           this->gpu_id == other.gpu_id && this->handler == other.handler &&
           this->batch_size == other.batch_size &&
           this->limit_max_image_pixels == other.limit_max_image_pixels;
  }
};

struct LoadModelResponse {
  int code;
  const std::string buf;

  LoadModelResponse(int code, const std::string buf) : code(code), buf(buf){};
};

// Due to https://github.com/llvm/llvm-project/issues/54668,
// so ignore bugprone-exception-escape
// NOLINTBEGIN(bugprone-exception-escape)
struct InferenceRequest {
  /**
   * @brief
   * - all of the pairs <parameter_name, value_content_type> are also stored
   * in the Header
   * - key: header_name
   * - value: header_value
   */
  using Headers = std::map<std::string, std::string>;
  using Parameters = std::map<std::string, std::vector<char>>;

  std::string request_id;
  Headers headers;
  Parameters parameters;

  InferenceRequest(){};

  InferenceRequest(const std::string& request_id, const Headers& headers,
                   const Parameters& parameters)
      : request_id(request_id), headers(headers), parameters(parameters){};
  // NOLINTEND(bugprone-exception-escape)
};
// Ref: Ref: https://github.com/pytorch/serve/blob/master/ts/service.py#L36
using InferenceRequestBatch = std::vector<InferenceRequest>;

struct InferenceResponse {
  using Headers = std::map<std::string, std::string>;

  int code = 200;
  std::string request_id;
  // msg data_dtype must be added in headers
  Headers headers;
  std::vector<char> msg;

  InferenceResponse(const std::string& request_id) : request_id(request_id){};

  void SetResponse(int new_code, const std::string& new_header_key,
                   const std::string& new_header_val,
                   const std::string& new_msg) {
    code = new_code;
    headers[new_header_key] = new_header_val;
    msg = torchserve::Converter::StrToVector(new_msg);
  };

  void SetResponse(int new_code, const std::string& new_header_key,
                   const std::string& new_header_val,
                   const std::vector<char>& new_msg) {
    code = new_code;
    headers[new_header_key] = new_header_val;
    msg = new_msg;
  };
};
// Ref: https://github.com/pytorch/serve/blob/master/ts/service.py#L105
using InferenceResponseBatch =
    std::map<std::string, std::shared_ptr<InferenceResponse>>;
}  // namespace torchserve
#endif  // TS_CPP_UTILS_MESSAGE_HH_