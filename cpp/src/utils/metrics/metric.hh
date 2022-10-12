#ifndef TS_CPP_UTILS_METRICS_METRIC_HH_
#define TS_CPP_UTILS_METRICS_METRIC_HH_

#include <string>

namespace torchserve {
    enum MetricType {
        COUNTER,
        GAUGE,
        HISTOGRAM
    };

    class Metric {
        public:
        virtual void Update(const double& value, const std::string& request_id = "") = 0;
        virtual void Reset() = 0;
    };
} // namespace torchserve

#endif // TS_CPP_UTILS_METRICS_METRIC_HH_
