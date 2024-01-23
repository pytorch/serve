#include "src/utils/metrics/log_metrics_cache.hh"

#include <stdexcept>

namespace torchserve {
TSLogMetric& LogMetricsCache::GetMetric(const MetricType& type,
                                        const std::string& name) {
  auto metric_type_match = cache.find(type);
  if (metric_type_match == cache.end()) {
    std::string error_message =
        "Metric with MetricType enum: " + std::to_string(type) +
        " not found in cache";
    throw std::invalid_argument(error_message);
  }

  auto metric_name_match = metric_type_match->second.find(name);
  if (metric_name_match == metric_type_match->second.end()) {
    std::string error_message =
        "Metric with MetricType enum: " + std::to_string(type) +
        " and name: " + name + " not found in cache";
    throw std::invalid_argument(error_message);
  }

  return metric_name_match->second;
}

void LogMetricsCache::Initialize(
    const MetricsConfigurationHandler& config_handler) {
  Clear();

  for (const auto& config : config_handler.GetModelMetrics()) {
    AddMetric(config.type, config.name, config.unit, config.dimension_names);
  }

  for (const auto& config : config_handler.GetTsMetrics()) {
    AddMetric(config.type, config.name, config.unit, config.dimension_names);
  }
}

TSLogMetric& LogMetricsCache::AddMetric(
    const MetricType& type, const std::string& name, const std::string& unit,
    const std::vector<std::string>& dimension_names) {
  auto metric_type_match = cache.find(type);
  if (metric_type_match == cache.end()) {
    metric_type_match =
        cache.insert({type, std::unordered_map<std::string, TSLogMetric>{}})
            .first;
  }

  return metric_type_match->second
      .insert({name, TSLogMetric(type, name, unit, dimension_names)})
      .first->second;
}

void LogMetricsCache::Clear() { cache.clear(); }
}  // namespace torchserve
