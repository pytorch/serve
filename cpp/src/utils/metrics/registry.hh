#ifndef TS_CPP_UTILS_METRICS_REGISTRY_HH_
#define TS_CPP_UTILS_METRICS_REGISTRY_HH_

#include <stdexcept>

#include "src/utils/metrics/log_metrics_cache.hh"
#include "src/utils/metrics/yaml_config.hh"

namespace torchserve {
class MetricsRegistry {
 public:
  static void Initialize(const std::string& metrics_config_file_path,
                         const MetricsContext& metrics_context);
  static std::shared_ptr<MetricsCache>& GetMetricsCacheInstance();

 private:
  static std::shared_ptr<MetricsCache> metrics_cache;
};
}  // namespace torchserve

#endif  // TS_CPP_UTILS_METRICS_REGISTRY_HH_
