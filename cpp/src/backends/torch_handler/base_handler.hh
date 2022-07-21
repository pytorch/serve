#ifndef TS_CPP_BACKENDS_TORCH_HANDLER_BASE_HANDLER_HH_
#define TS_CPP_BACKENDS_TORCH_HANDLER_BASE_HANDLER_HH_

#include <torch/script.h>
#include <torch/torch.h>
#include <functional>

#include "src/backends/core/backend.hh"

namespace torchserve {
  /**
   * @brief 
   * TorchBaseHandler <=> BaseHandler:
   * https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py#L37
   * 
   * TorchBaseHandler is not responsible for loading model since it is derived from
   * TorchScritpedModelInstance.
   */
  class TorchBaseHandler {
    public:
    TorchBaseHandler() {};
    virtual ~TorchBaseHandler() {};

    void Initialize(std::shared_ptr<torchserve::ModelInstance> model_instance);

    virtual std::vector<torch::jit::IValue> Preprocess(
      std::unique_ptr<torchserve::InferenceRequest> inference_request) = 0;
    
    virtual std::shared_ptr<torchserve::InferenceResponse> Postprocess(
      torch::Tensor data) = 0;


    /**
     * @brief 
     * function Predict <=> entry point function handle
     * https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py#L205
     * @param inference_request 
     * @return std::shared_ptr<torchserve::InferenceResponse> 
     */
    std::shared_ptr<torchserve::InferenceResponse> Handle(
      std::unique_ptr<torchserve::InferenceRequest> inference_request,
      std::function<torch::Tensor(std::vector<torch::jit::IValue> inputs)> infer_func) {
        auto inputs = Preprocess(std::move(inference_request));
        auto output = infer_func(inputs);
        return Postprocess(output);
    };
  };
} // namespace torchserve
#endif // TS_CPP_BACKENDS_TORCH_HANDLER_BASE_HANDLER_HH_