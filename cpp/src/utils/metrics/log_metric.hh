#ifndef TS_CPP_UTILS_METRICS_LOG_METRIC_HH_
#define TS_CPP_UTILS_METRICS_LOG_METRIC_HH_

#include <string>
#include <vector>

#include "src/utils/metrics/metric.hh"

namespace torchserve {
class TSLogMetric : public IMetric {
 public:
  TSLogMetric(const MetricType& type, const std::string& name,
              const std::string& unit,
              const std::vector<std::string>& dimension_names)
      : IMetric(type, name, unit, dimension_names) {}
  ~TSLogMetric() override {}
  void AddOrUpdate(const std::vector<std::string>& dimension_values,
                   const double& value) override;
  void AddOrUpdate(const std::vector<std::string>& dimension_values,
                   const std::string& request_id, const double& value) override;

 private:
  std::string BuildDimensionsString(
      const std::vector<std::string>& dimension_values);
  void ValidateDimensionValues(
      const std::vector<std::string>& dimension_values);
  void ValidateMetricValue(const double& value);
  void Emit(const std::vector<std::string>& dimension_values,
            const std::string& request_id, const double& value);
};
}  // namespace torchserve

#endif  // TS_CPP_UTILS_METRICS_LOG_METRIC_HH_
