#include "src/backends/torch_scripted/handler/vision_handler.hh"

namespace torchserve {
  namespace torchscripted {
  std::vector<torch::jit::IValue> VisionHandler::Preprocess(
    /**
     * @brief 
     * Ref: https://github.com/pytorch/serve/blob/master/ts/torch_handler/vision_handler.py#L27
     */
    torchserve::InferenceRequestBatch batch) {
    std::vector<std::vector<std::byte>>> images;
    for (const auto& request : batch) {
      std::vector<std::byte>> image;
      images.emplace_back(image);
      auto data_it = request.parameters.find("data");
      auto dtype_it = request.headers.find("data_dtype");
      if (data_it == request.parameters.end()) {
        data_it = request.parameters.find("body");
        dtype_it = request.headers.find("body_dtype");
      }

      if (data_it == request.parameters.end() || 
        dtype_it == request.headers.end()) {
        LOG(ERROR) << "No input for request id:" << request.request_id;
      } 
      
      if (dtype_it->second == "String") {
        // case1: the image is a string of bytesarray
        try {
          auto b64decoded_str = folly::base64Decode(data_it->second);
          torchserve::Converter::StrToBytes(b64decoded_str, image);
        } catch (folly::base64_decode_error e) {
          LOG(ERROR) << "Failed to base64Decode for request id:" << request.request_id 
          << ", error: " << e.what();
        }
      } else if (dtype_it->second == "Bytes") {
        // case2: the image is sent as bytesarray


      } else if (dtype_it->second == "List") {
        // case3: the image is a list

      }

    }
  } // namespace torchscripted
} // namespace torchserve 