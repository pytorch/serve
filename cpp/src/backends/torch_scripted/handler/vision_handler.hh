#ifndef TS_CPP_BACKENDS_TORCH_SCRIPTED_HANDLER_VISION_HANDLER_HH_
#define TS_CPP_BACKENDS_TORCH_SCRIPTED_HANDLER_VISION_HANDLER_HH_

#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include "src/utils/message.hh"

namespace torchserve {
  namespace torchscripted {
    class VisionHandler : public BaseHandler {
      public:
      VisionHandler() {};
      ~VisionHandler() {};
      
      std::vector<torch::jit::IValue> Preprocess(
        const torchserve::InferenceRequestBatch& inference_request_batch) override;
      
      std::shared_ptr<torchserve::InferenceResponse> Postprocess(
        torch::Tensor data) override;

      virtual torch::Tensor Predict(
        std::shared_ptr<torch::jit::script::Module> model,
        std::vector<torch::jit::IValue> inputs) override;
    };
  } // torchscripted
} // namespace torchserve
#endif // TS_CPP_BACKENDS_TORCH_SCRIPTED_HANDLER_VISION_HANDLER_HH_