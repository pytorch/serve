
#ifndef TS_CPP_UTILS_MESSAGE_HH_
#define TS_CPP_UTILS_MESSAGE_HH_

#include <string>

namespace torchserve {
  // TODO: expand to support model instance, large model (ref: ModelConfig in config.hh)
  struct LoadModelRequest {
    const std::string& model_path;	// /path/to/model/file
    const std::string& model_name;
    int gpu_id;						// None if CPU else gpu_id
    const std::string& handler;		// Expected to be null for cpp backend
    const std::string& envelope;	// name of wrapper/unwrapper of request data if provided
    int batch_size;
    bool limit_max_image_pixels;	// limit pillow image max_image_pixels

    LoadModelRequest(
      const std::string& model_path,
      const std::string& model_name,
      int gpu_id,
      const std::string& handler,
      const std::string& envelope,
      int batch_size,
      bool limit_max_image_pixels
    ) : model_path(model_path), model_name(model_name), gpu_id(gpu_id), 
    handler(handler), envelope(envelope), batch_size(batch_size), 
    limit_max_image_pixels(limit_max_image_pixels) {}
  };

  struct LoadModelResponse {
    int code;
    int length;
    const std::string buf;

    LoadModelResponse(int code, int length, const std::string buf) : 
    code(code), length(length), buf(buf) {};
  };

  
  struct InferenceRequest {

  };

  struct InferenceResponse {
    // TODO: definition
    int code;
    int length;
    const std::string buf;
    InferenceResponse() {};
  };
} // namespace torchserve
#endif // TS_CPP_UTILS_MESSAGE_HH_