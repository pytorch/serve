#ifndef TS_CPP_UTILS_CONFIG_HH_
#define TS_CPP_UTILS_CONFIG_HH_

#include <map>
#include <string>
#include <vector>
#include <stdexcept>

#include "src/utils/model_archive.hh"

namespace torchserve {
  // TODO: limit to lower case "cpu", or "gpu"
  using DeviceType = std::string;

  struct ModelConfig {
    // The model alias name.
    std::string model_name;
    // The path of the model file.
    std::string url;
    // The number of a model's copies on CPU or GPU.
    uint8_t num_instances;
    // The minimum number of workers distributed on this model's instances.
    uint16_t min_workers;
    // The maximum number of workers distributed on this model's instances.
    uint16_t max_workers;
    // The maximum size of queue holding incoming requests for the model.
    uint32_t max_queue_size;
    // The maximum etry attempts for a model incase of a failure.
    uint16_t max_retries;
    // The maximum batch size in ms that a model is expected to handle.
    uint16_t batch_size;
    // The maximum batch delay time TorchServe waits to receive batch_size number
    // of requests.
    uint32_t max_batch_delay_msec;
    // The timeout in second of a model's response.
    uint32_t response_timeout_sec;
    // The device type  where a model instance is loaded.
    DeviceType device_type;
    // Large model only has one model instance with a list of device ids (eg. gpu id).
    // Normal model has multiple model instances, each model instance has one device id.
    std::vector<short> device_ids;
    // The runtime type.
    torchserve::Manifest::RuntimeType runtime_type;
    // The model's manifest.
    Manifest manifest;
  };

  /**
   * TODO: Load Backend Framework Configuration into this object during initialization.
   * Sample Yaml Config:
   * backends:
   *   frameworks:
   *     torch_scripted:
   *       runtime_type: [LSP, LST]
   *       version: 1.11
   *     torch_deploy:
   *       runtime_type: [LDP, LDT]
   *       version: 1.11
   */
  class BackendFrameworkConfig {
    public:
    struct Framework {
      std::string name;
      std::string version;
      Framework(const std::string& name, const std::string& version) : 
      name(name), version(version) {}
    };

    static BackendFrameworkConfig GetInstance() {
      static BackendFrameworkConfig instance;
      return instance; 
    };

    private:
    /**
     * @brief Get the Libso object
     * TODO: TorchServe cpp backend shared lib naming convention:
     * - torchserve/backends/{framework_name}/{version}/lib{framework_name}_{device_type}_backend.so
     * eg. torchserve/backends/torch_scripted/1.11/libts_backends_torch_scripted.so
     * 
     * - framework_name follows the name in src/backends/{framework_name}/
     * eg. src/backends/torch_scripted
     * @param runtime_type 
     * @param device_type 
     * @return std::string 
     */
    const std::string GetBackendLibPath(
      const torchserve::Manifest::RuntimeType& runtime_type, 
      DeviceType device_type,
      const std::string& version);

    std::map<std::string, std::shared_ptr<torchserve::BackendFrameworkConfig::Framework>> 
    runtime_to_framework_map_;

    BackendFrameworkConfig();
  }; 
}  // namespace torchserve
#endif // TS_CPP_UTILS_CONFIG_HH_