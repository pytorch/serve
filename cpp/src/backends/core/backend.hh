#pragma once

#include <atomic>
#include <filesystem>
#include <memory>
#include <queue>
#include <random>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "model_instance.hh"
#include "src/utils/config.hh"
#include "src/utils/dl_loader.hh"
#include "src/utils/message.hh"

namespace torchserve {
/**
 * @brief TorchServe Backend Interface
 * Backend <=> ModelLoader:
 * serve/blob/master/ts/model_loader.py#L28
 *
 * Note:
 * Any framework should implement its own backend which includes:
 * 1. Implement class Backend
 * 2. Implement class ModelInstance
 * 3. function std::shared_ptr<Backend> CreateBackend()
 *
 * The core idea:
 * - A backend has multiple model instances.
 * - A worker is associated with one model instance.
 * - A model instance is one model loaded on CPU or GPU.
 */
class Backend {
 public:
  enum ModelInstanceStatus { NOT_INIT, INIT, READY, FAILED };

  // NOLINTBEGIN(cppcoreguidelines-pro-type-member-init)
  struct ModelInstanceInfo {
    ModelInstanceStatus status;
    std::shared_ptr<ModelInstance> model_instance;
  };
  // NOLINTEND(cppcoreguidelines-pro-type-member-init)

  Backend();
  virtual ~Backend();

  bool Initialize(const std::string &model_dir);

  ModelInstanceStatus GetModelInstanceStatus(
      const std::string &model_instance_id);

  std::shared_ptr<torchserve::ModelInstance> GetModelInstance(
      const std::string &model_instance_id);
  std::shared_ptr<torchserve::ModelInstance> GetModelInstance();

  void SetModelInstanceInfo(const std::string &model_instance_id,
                            ModelInstanceStatus new_status,
                            std::shared_ptr<ModelInstance> new_model_instance);

  std::unique_ptr<torchserve::LoadModelResponse> LoadModel(
      std::shared_ptr<torchserve::LoadModelRequest> load_model_request);

 protected:
  std::string BuildModelInstanceId(
      std::shared_ptr<torchserve::LoadModelRequest> load_model_request);

  void LoadHandler(const std::string &model_dir);

  std::unique_ptr<torchserve::LoadModelResponse> LoadModelInternal(
      std::shared_ptr<torchserve::LoadModelRequest> load_model_request);

  std::shared_ptr<torchserve::Manifest> manifest_;

  // key: model_instance_id
  // value: model_instance_info
  std::map<std::string, ModelInstanceInfo> model_instance_table_;

  std::vector<std::string> ready_model_instance_ids_;

  std::atomic_uint16_t model_instance_count_ = 0;

  std::unique_ptr<DLLoader<BaseHandler>> dl_loader_;
  std::shared_ptr<BaseHandler> handler_;

  std::size_t Random();
  std::mt19937 random_generator_;
};
}  // namespace torchserve
