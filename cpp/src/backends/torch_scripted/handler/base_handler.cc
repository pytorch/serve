#include "src/backends/torch_scripted/handler/base_handler.hh"

namespace torchserve {
  namespace torchscripted {
    std::pair<std::shared_ptr<torch::jit::script::Module>, std::shared_ptr<torch::Device>> 
    BaseHandler::LoadModel(
      std::shared_ptr<torchserve::LoadModelRequest>& load_model_request) {
      try {
        auto device = GetTorchDevice(load_model_request);
        auto module = std::make_shared<torch::jit::script::Module>(torch::jit::load(
          // TODO: windows
          fmt::format("{}/{}", 
          load_model_request->model_dir, 
          manifest->GetModel().serialized_file),
          device));
        return std::make_pair(module, device);
        /*
        if (load_model_request->gpu_id != -1) {
          module = std::make_shared<torch::jit::script::Module>(
            torch::jit::load(
              // TODO: windows
              fmt::format("{}/{}", 
              load_model_request->model_dir, 
              manifest->GetModel().serialized_file),
              GetTorchDevice(load_model_request)));
        } else {
          module = std::make_shared<torch::jit::script::Module>(
            torch::jit::load(fmt::format("{}/{}", 
              load_model_request->model_dir, 
              manifest->GetModel().serialized_file)));
        }
        */
      } catch (const c10::Error& e) {
        LOG(ERROR) << "loading the model: " 
        << load_model_request->model_name 
        << ", device id:" 
        << load_model_request->gpu_id
        << ", error: " << e.msg();
        throw e;
      }
    }

    std::shared_ptr<torch::Device> BaseHandler::GetTorchDevice(
      std::shared_ptr<torchserve::LoadModelRequest> load_model_request) {
      /**
       * @brief 
       * TODO: extend LoadModelRequest to support 
       * - device type: CPU, GPU or others
       */
      if (load_model_request->gpu_id < 0) {
        return std::make_shared<torch::Device>(torch::kCPU);
      } 

      return std::make_shared<torch::Device>(torch::kCUDA, load_model_request->gpu_id);
    }
  } // namespace torchscripted
} // namespace torchserve

#if defined(__linux__) || defined(__APPLE__)
extern "C" {
  torchserve::torchscripted::BaseHandler *allocatorBaseHandler() {
    return new torchserve::torchscripted::BaseHandler();
  }
  
  void deleterBaseHandler(torchserve::torchscripted::BaseHandler *p) {
    delete p;
  }
}
#endif