#include "src/backends/core/backend.hh"

namespace torchserve {
  std::unique_ptr<torchserve::LoadModelResponse> Backend::LoadModel(
    std::shared_ptr<torchserve::LoadModelRequest> load_model_request) {
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
      "{}:{}:{}", 
      device_type,
      load_model_request->gpu_id,
      model_instance_count_.fetch_add(1)
    );
  }

  void Backend::SetModelInstanceInfo(
    const std::string& model_instance_id,
    ModelInstanceStatus new_status,
    std::shared_ptr<torchserve::ModelInstance> new_model_instance) {
    model_instance_table_[model_instance_id].status = new_status;
    model_instance_table_[model_instance_id].model_instance = std::move(new_model_instance);
  }


  torchserve::Backend::ModelInstanceStatus Backend::GetModelInstanceStatus(const std::string& model_instance_id) {
    auto model_instance_info = model_instance_table_.find(model_instance_id);
    if (model_instance_info == model_instance_table_.end()) {
      return torchserve::Backend::ModelInstanceStatus::NOT_INIT;
    }
    return model_instance_info->second.status;
  }

  std::shared_ptr<torchserve::ModelInstance> Backend::GetModelInstance(const std::string& model_instance_id) {
    auto model_instance_info = model_instance_table_.find(model_instance_id);
    if (model_instance_info == model_instance_table_.end()) {
      return std::shared_ptr<torchserve::ModelInstance>(nullptr);
    }
    return model_instance_info->second.model_instance;
  }

  std::shared_ptr<torchserve::ModelInstance> Backend::GetModelInstance() {
    if (ready_model_instance_ids_.empty()) {
      return std::shared_ptr<torchserve::ModelInstance>(nullptr);
    }
   
    auto model_instance_info = model_instance_table_.find(ready_model_instance_ids_[Random()]);
    return model_instance_info->second.model_instance;
  }

  std::size_t Backend::Random() {
    auto size = ready_model_instance_ids_.size();
    if (size == 1) {
      return 0;
    } else {
      std::uniform_int_distribution<> uint_distribution_(0, size - 1);
      return uint_distribution_(random_generator_);
    }
  }
} // namespace torchserve
