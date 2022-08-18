#include "src/backends/core/backend.hh"

namespace torchserve {
  std::pair<std::unique_ptr<torchserve::LoadModelResponse>, std::shared_ptr<torchserve::ModelInstance>> 
  Backend::LoadModel(std::shared_ptr<torchserve::LoadModelRequest> load_model_request) {
    /**
     * TODO: 
     * in multi-thread, this function is called by workers.
     * - check the model instance status in LoadModel
     * - status_NOT_INIT: call LoadModelInternal and register the new model instance
     * - status_INIT: wait for notification
     * - status_READY: return the model instance if it is already.
     * 
     * Common steps:
     * https://github.com/pytorch/serve/blob/master/ts/model_loader.py#L62
     */
    
    return LoadModelInternal(std::move(load_model_request));
  }

  std::string Backend::BuildModelInstanceId(
    std::shared_ptr<torchserve::LoadModelRequest> load_model_request) {
    std::string device_type("cpu");
    if (load_model_request->gpu_id >= 0) {
      device_type = "gpu";
    }
    return fmt::format(
      "{}:{}:{}:{}", 
      load_model_request->model_name,
      device_type,
      load_model_request->gpu_id,
      model_instance_count_.fetch_add(1)
    );
  }


  torchserve::Backend::ModelInstanceStatus Backend::GetModelInstanceStatus(const std::string& model_instance_id) {
    auto status = model_instance_status_.find(model_instance_id);
    if (status == model_instance_status_.end()) {
      return torchserve::Backend::ModelInstanceStatus::NOT_INIT;
    }
    return status->second;
  }

  std::shared_ptr<torchserve::ModelInstance> Backend::GetModelInstance(const std::string& model_instance_id) {
    auto model_instance = model_instance_table_.find(model_instance_id);
    if (model_instance == model_instance_table_.end()) {
      return nullptr;
    }
    return model_instance->second;
  }
} // namespace torchserve
