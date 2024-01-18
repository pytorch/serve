#ifndef TS_CPP_UTILS_METRICS_LOG_METRICS_CACHE_HH_
#define TS_CPP_UTILS_METRICS_LOG_METRICS_CACHE_HH_

#include <string>
#include <unordered_map>
#include <vector>

#include "src/utils/metrics/cache.hh"
#include "src/utils/metrics/config.hh"
#include "src/utils/metrics/log_metric.hh"

namespace torchserve {
class LogMetricsCache : public MetricsCache {
 public:
  ~LogMetricsCache() override {}
  void Initialize(const MetricsConfigurationHandler& config_handler) override;
  TSLogMetric& GetMetric(const MetricType& type,
                         const std::string& name) override;

 protected:
  TSLogMetric& AddMetric(
      const MetricType& type, const std::string& name, const std::string& unit,
      const std::vector<std::string>& dimension_names) override;

 private:
  std::unordered_map<MetricType, std::unordered_map<std::string, TSLogMetric>>
      cache;
  void Clear();
};
}  // namespace torchserve

#endif  // TS_CPP_UTILS_METRICS_LOG_METRICS_CACHE_HH_
