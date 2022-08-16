#ifndef TS_CPP_BACKENDS_TORCH_SCRIPTED_HANDLER_VISION_HANDLER_HH_
#define TS_CPP_BACKENDS_TORCH_SCRIPTED_HANDLER_VISION_HANDLER_HH_

#include <cstddef>
//#include <folly/base64.h>
#include <folly/json.h>
#include <sstream>
#include <torch/script.h>
#include <torch/serialize.h>
#include <torch/torch.h>
#include <vector>
#include "src/utils/message.hh"

namespace torchserve {
  namespace torchscripted {
    class VisionHandler : public BaseHandler {
      public:
      VisionHandler() {};
      ~VisionHandler() {};
      
      virtual std::vector<torch::jit::IValue> Preprocess(
        std::shared_ptr<torch::Device>& device,
        std::map<uint8_t, std::string>& idx_to_req_i,
        std::shared_ptr<torchserve::InferenceRequestBatch>& request_batch,
        std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) override;
      
      virtual torch::Tensor Predict(
        std::shared_ptr<torch::jit::script::Module> model, 
        std::vector<torch::jit::IValue>& inputs,
        std::shared_ptr<torch::Device>& device,
        std::map<uint8_t, std::string>& idx_to_req_id,
        std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) override;

      virtual void Postprocess(
        const torch::Tensor& data,
        std::map<uint8_t, std::string>& idx_to_req_i,
        std::shared_ptr<torchserve::InferenceResponseBatch>& response_batch) override;
    };
  } // torchscripted
} // namespace torchserve
#endif // TS_CPP_BACKENDS_TORCH_SCRIPTED_HANDLER_VISION_HANDLER_HH_