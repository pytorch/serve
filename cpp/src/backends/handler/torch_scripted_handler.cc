#include "src/backends/handler/torch_scripted_handler.hh"

#include <memory>

#include "src/utils/message.hh"
#include "src/utils/metrics/registry.hh"

namespace torchserve {
std::pair<std::shared_ptr<void>, std::shared_ptr<torch::Device>>
TorchScriptHandler::LoadModel(
    std::shared_ptr<torchserve::LoadModelRequest>& load_model_request) {
  try {
    auto device = GetTorchDevice(load_model_request);
    std::shared_ptr<void> module(
        std::make_shared<torch::jit::Module>(torch::jit::load(
            // TODO: windows
            fmt::format("{}/{}", load_model_request->model_dir,
                        manifest_->GetModel().serialized_file),
            *device)));
    return std::make_pair(module, device);
  } catch (const c10::Error& e) {
    TS_LOGF(ERROR, "loading the model: {}, device id: {}, error: {}",
            load_model_request->model_name, load_model_request->gpu_id,
            e.msg());
    throw e;
  } catch (const std::runtime_error& e) {
    TS_LOGF(ERROR, "loading the model: {}, device id: {}, error: {}",
            load_model_request->model_name, load_model_request->gpu_id,
            e.what());
    throw e;
  }
}
}  // namespace torchserve
