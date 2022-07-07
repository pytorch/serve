#ifndef TS_CPP_BACKENDS_TORCH_SCRIPTED_TORCH_SCRIPTED_BACKEND_HH_
#define TS_CPP_BACKENDS_TORCH_SCRIPTED_TORCH_SCRIPTED_BACKEND_HH_

#include <memory>
#include <torch/script.h>
#include <torch/torch.h>

#include "src/backends/core/backend.hh"
#include "src/utils/message.hh"

namespace torchserve {
  class TorchScriptedBackend final : public Backend {
    public:
    TorchScriptedBackend() {};
    ~TorchScriptedBackend() {};

    std::pair<
    std::unique_ptr<torchserve::LoadModelResponse>, std::shared_ptr<torchserve::ModelInstance>> 
    LoadModelInternal(std::shared_ptr<torchserve::LoadModelRequest> load_model_request) override;

    private:
    torch::Device GetTorchDevice(
      std::shared_ptr<torchserve::LoadModelRequest> load_model_request);
  };

  class TorchScritpedModelInstance final : public ModelInstance {
    public:
    TorchScritpedModelInstance() {};
    ~TorchScritpedModelInstance() {};

    void Initialize(
      std::shared_ptr<torch::jit::script::Module> model, 
      std::shared_ptr<torchserve::LoadModelRequest> load_model_request);

    std::shared_ptr<torchserve::InferenceResponse> Predict(
      std::unique_ptr<torchserve::InferenceRequest> inference_request) override {
      // TODO; impl.
      return std::make_shared<torchserve::InferenceResponse>();
    };

    private:
    std::shared_ptr<torch::jit::script::Module> model_;
  };
} // namespace torchserve
#endif // TS_CPP_BACKENDS_TORCH_SCRIPTED_TORCH_SCRIPTED_BACKEND_HH_