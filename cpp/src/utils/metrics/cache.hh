#ifndef TS_CPP_UTILS_METRICS_CACHE_HH_
#define TS_CPP_UTILS_METRICS_CACHE_HH_

#include <string>
#include <vector>

#include "src/utils/metrics/config.hh"
#include "src/utils/metrics/metric.hh"

namespace torchserve {
class MetricsCache {
 public:
  virtual ~MetricsCache() {}
  virtual void Initialize(
      const MetricsConfigurationHandler& config_handler) = 0;
  virtual IMetric& GetMetric(const MetricType& type,
                             const std::string& name) = 0;

 protected:
  virtual IMetric& AddMetric(
      const MetricType& type, const std::string& name, const std::string& unit,
      const std::vector<std::string>& dimension_names) = 0;
};
}  // namespace torchserve

#endif  // TS_CPP_UTILS_METRICS_CACHE_HH_
