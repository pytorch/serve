#ifndef TS_CPP_UTILS_METRICS_LOG_METRIC_HH_
#define TS_CPP_UTILS_METRICS_LOG_METRIC_HH_

#include <string>
#include <tuple>
#include <vector>
#include <map>

#include "src/utils/metrics/metric.hh"
#include "src/utils/metrics/emitter.hh"

namespace torchserve {
    // Value entry for log metric consisting of <metric_value, timestamp, request_id>
    typedef std::tuple<double, std::uint64_t, std::string> LogMetricValue;

    class LogMetric : public Metric, public Emitter {
        public:
        LogMetric(const std::string& name, const std::string& unit,
                  const std::map<std::string, std::string>& dimensions, const MetricType& type);
        void Update(const double& value, const std::string& request_id = "");
        void Reset();
        void Emit();

        private:
        const std::string name;
        const std::string unit;
        const std::map<std::string, std::string> dimensions;
        const MetricType type;
        std::vector<LogMetricValue> values;
    };
} // namespace torchserve

#endif // TS_CPP_UTILS_METRICS_LOG_METRIC_HH_
