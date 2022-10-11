#ifndef TS_CPP_BACKENDS_CORE_BACKEND_HH_
#define TS_CPP_BACKENDS_CORE_BACKEND_HH_

#include <atomic>
#include <filesystem>
#include <memory>
#include <queue>
#include <random>
#include <stdexcept>
//#include <stdlib.h>
//#include <time.h>
#include <tuple>
#include <utility>


#include "src/utils/config.hh"
#include "src/utils/message.hh"

namespace torchserve {
  /**
   * 
   * @brief TorchServe ModelInstance Interface
   * ModelInstance <=> Service:
   *  https://github.com/pytorch/serve/blob/master/ts/service.py#L21
   */
  class ModelInstance {
    public:
    ModelInstance(const std::string& instance_id) : instance_id_(instance_id) {};
    virtual ~ModelInstance() {};

    virtual std::shared_ptr<torchserve::InferenceResponseBatch> Predict(
      std::shared_ptr<torchserve::InferenceRequestBatch> batch) = 0;

    const std::string& GetInstanceId() {
      return instance_id_;
    };

    protected:
    // instance_id naming convention:
    // device_type + ":" + device_id (or object id)
    std::string instance_id_;
  };

  /**
   * @brief TorchServe Backend Interface
   * Backend <=> ModelLoader:
   * https://github.com/pytorch/serve/blob/master/ts/model_loader.py#L28
   * 
   * Note:
   * Any framework should implement its own backend which includes:
   * 1. Implement class Backend
   * 2. Implement class ModelInstance
   * 3. function std::shared_ptr<Backend> CreateBackend()
   * 
   * The core idea:
   * - A framework has its own backend in a model server.
   * - A backend has multiple model instances.
   * - A worker is associated with one model instance.
   * - A model instance is one model loaded on CPU or GPU.
   */
  class Backend {
    public:
    enum ModelInstanceStatus {
      NOT_INIT,
      INIT,
      READY,
      FAILED
    };

    struct ModelInstanceInfo {
      ModelInstanceStatus status;
      std::shared_ptr<torchserve::ModelInstance> model_instance;
    };

    Backend() = default;
    virtual ~Backend() = default;

    virtual bool Initialize(const std::string& model_dir) {
      random_generator_.seed(time(0));
      manifest_ = std::make_shared<torchserve::Manifest>();
      // TODO: windows
      return manifest_->Initialize(fmt::format("{}/MAR-INF/MANIFEST.json", model_dir));
    };

    std::unique_ptr<torchserve::LoadModelResponse> LoadModel(
      std::shared_ptr<torchserve::LoadModelRequest> load_model_request);

    virtual 
    std::unique_ptr<torchserve::LoadModelResponse>LoadModelInternal(
      std::shared_ptr<torchserve::LoadModelRequest> load_model_request) = 0;

    ModelInstanceStatus GetModelInstanceStatus(const std::string& model_instance_id);

    std::shared_ptr<torchserve::ModelInstance> GetModelInstance(const std::string& model_instance_id);
    std::shared_ptr<torchserve::ModelInstance> GetModelInstance();

    void SetModelInstanceInfo(
      const std::string& model_instance_id,
      ModelInstanceStatus new_status,
      std::shared_ptr<torchserve::ModelInstance> new_model_instance);

    protected:
    std::string BuildModelInstanceId(std::shared_ptr<torchserve::LoadModelRequest> load_model_request);

    std::shared_ptr<torchserve::Manifest> manifest_;

    // key: model_instance_id
    // value: model_instance_info 
    std::map<std::string, ModelInstanceInfo> model_instance_table_;

    std::vector<std::string> ready_model_instance_ids_;

    std::atomic_uint16_t model_instance_count_ = 0;

    
    private:
    std::size_t Random();

    std::mt19937 random_generator_;
  };

  class ModelWorker {
    public:
    ModelWorker() {};
    ~ModelWorker();
  };
}  // namespace torchserve
#endif // TS_CPP_BACKENDS_CORE_BACKEND_HH_