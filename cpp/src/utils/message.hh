#ifndef TS_CPP_UTILS_MESSAGE_HH_
#define TS_CPP_UTILS_MESSAGE_HH_

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace torchserve {
  #define CONTENT_TYPE_JSON "application/json"
  #define CONTENT_TYPE_TEXT "text"

  #define DATA_TYPE_STRING "string"
  #define DATA_TYPE_BYTES "bytes"
  
  class Converter {
    public:
    static std::vector<std::byte> StrToBytes(const std::string& str) {
      std::vector<std::byte> str_bytes;
      for (auto& ch : str) {
        str_bytes.emplace_back(std::byte(ch));
      }
      return str_bytes;
    }
  };
  
  // TODO: expand to support model instance, large model (ref: ModelConfig in config.hh)
  struct LoadModelRequest {
    // /path/to/model/file
    const std::string model_path;	
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

    LoadModelRequest(
      const std::string& model_path,
      const std::string& model_name,
      int gpu_id,
      const std::string& handler,
      const std::string& envelope,
      int batch_size,
      bool limit_max_image_pixels
    ) : model_path{model_path}, model_name{model_name}, gpu_id{gpu_id}, 
    handler{handler}, envelope{envelope}, batch_size{batch_size}, 
    limit_max_image_pixels{limit_max_image_pixels} {}
  };

  struct LoadModelResponse {
    int code;
    int length;
    const std::string buf;

    LoadModelResponse(int code, int length, const std::string buf) : 
    code(code), length(length), buf(buf) {};
  };

  struct InferenceRequest {
    /**
     * @brief 
     * - all of the pairs <parameter_name, value_content_type> are also stored 
     * in the Header
     * - key: header_name
     * - value: header_value
     */
    using Headers = std::map<std::string, std::string>;
    using Parameters = std::map<std::string, std::vector<std::byte>>;
      
    std::string request_id;
    Headers headers;
    Parameters parameters;

    InferenceRequest() {};
  };
  // Ref: Ref: https://github.com/pytorch/serve/blob/master/ts/service.py#L36
  using InferenceRequestBatch = std::vector<InferenceRequest>;

  struct InferenceResponse {
    using Headers = std::map<std::string, std::string>;

    int code;
    std::string request_id;
    // msg data_dtype must be added in headers
    Headers headers;
    std::vector<std::byte> msg;

    InferenceResponse(const std::string& request_id) : request_id(request_id) {};

    void SetResponse(
      int new_code,
      const std::string& new_header_key,
      const std::string& new_header_val,
      const std::string& new_msg) {
      code = new_code;
      headers[new_header_key] = new_header_val;
      msg = torchserve::Converter::StrToBytes(new_msg);
    };
  };
  // Ref: https://github.com/pytorch/serve/blob/master/ts/service.py#L105
  using InferenceResponseBatch = std::map<std::string, std::shared_ptr<InferenceResponse>>;
} // namespace torchserve
#endif // TS_CPP_UTILS_MESSAGE_HH_