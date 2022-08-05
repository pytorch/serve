#include "src/backends/torch_scripted/handler/vision_handler.hh"

namespace torchserve {
  namespace torchscripted {
    std::vector<torch::jit::IValue> VisionHandler::Preprocess(
      const torchserve::InferenceRequestBatch& inference_request_batch) {
      std::vector<torch::Tensor> images;

      std::for_each(
        std::begin(inference_request_batch),
        std::emd(inference_request_batch),
        [](std::unique<InferenceRequest>& inference_request) {
          std::vector<torch::Tensor> images;
          auto input_image_it = inference_request->parameters.find("data");
          auto input_image_content_type_it = inference_request->headers.find("data");
          if (input_image_it == inference_request->parameters.end()) {
            input_image_it = inference_request->parameters.find("body");
            input_image_content_type_it = inference_request->headers.find("body");
          }

          if (input_image_content_type_it->second().starts_with(torchserve::CONTENT_TYPE_TEXT)) {
            
          } else {

          }

          if (input_image.index() == 0) {
            // case1: decode if image is a string of bytesarray

          } else if (input_image.index() == 2) {
            // case2: image is bytesarray
          } else if (input_image.index() == 3) {
            // case3: image is a list of float
          }

        }
      )

    }
  } // namespace torchscripted
} // namespace torchserve 