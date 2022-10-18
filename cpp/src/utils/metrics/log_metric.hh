#ifndef TS_CPP_UTILS_METRICS_LOG_METRIC_HH_
#define TS_CPP_UTILS_METRICS_LOG_METRIC_HH_

#include <string>
#include <set>
#include <map>

#include "src/utils/metrics/metric.hh"

namespace torchserve {
    class TSLogMetric : public IMetric {
        public:
        TSLogMetric(const MetricType& type, const std::string& name, const std::string& unit,
                    const std::set<std::string>& dimension_names) :
                    IMetric(type, name, unit, dimension_names) {}
        void AddOrUpdate(const std::map<std::string, std::string>& dimension_values, const double& value);
        void AddOrUpdate(const std::map<std::string, std::string>& dimension_values,
                         const std::string& request_id, const double& value);

        private:
        void ValidateDimensionValues(const std::map<std::string, std::string>& dimension_values);
        void ValidateMetricValue(const double& value);
        void Emit(const std::map<std::string, std::string>& dimension_values, const double& value,
                  const std::string& request_id);
    };
} // namespace torchserve

#endif // TS_CPP_UTILS_METRICS_LOG_METRIC_HH_
