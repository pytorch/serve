#ifndef TS_CPP_UTILS_METRICS_METRIC_CACHE_HH_
#define TS_CPP_UTILS_METRICS_METRIC_CACHE_HH_

#include <string>
#include <map>

#include "src/utils/metrics/metric.hh"

namespace torchserve {
    class MetricCache {
        public:
        virtual Metric AddMetric(const std::string& name, const std::string& unit,
                                 const std::map<std::string, std::string>& dimensions, const MetricType& type) = 0;
        virtual Metric GetMetric(const std::string& name, const std::map<std::string, std::string>& dimensions) = 0;
        virtual void Flush() = 0;
    };
} // namespace torchserve

#endif // TS_CPP_UTILS_METRICS_METRIC_CACHE_HH_
