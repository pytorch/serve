#include "src/utils/metrics/registry.hh"

namespace torchserve {
std::shared_ptr<MetricsCache> MetricsRegistry::metrics_cache = nullptr;

void MetricsRegistry::Initialize(const std::string& metrics_config_file_path,
                                 const MetricsContext& metrics_context) {
  try {
    std::shared_ptr<MetricsConfigurationHandler> metrics_config_handler =
        std::make_shared<YAMLMetricsConfigurationHandler>();
    metrics_config_handler->LoadConfiguration(metrics_config_file_path,
                                              metrics_context);
    metrics_cache = std::make_shared<LogMetricsCache>();
    metrics_cache->Initialize(*metrics_config_handler);
  } catch (...) {
    metrics_cache = nullptr;
    throw;
  }
}

std::shared_ptr<MetricsCache>& MetricsRegistry::GetMetricsCacheInstance() {
  if (metrics_cache == nullptr) {
    throw std::runtime_error(
        "Metrics cache not initialized in metrics registry");
  }

  return metrics_cache;
}
}  // namespace torchserve
