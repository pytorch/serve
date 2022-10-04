#ifndef TS_CPP_UTILS_METRICS_METRIC_HH_
#define TS_CPP_UTILS_METRICS_METRIC_HH_

#include <string>
#include <vector>

#include "src/utils/metrics/dimension.hh"

namespace torchserve {
    enum MetricType {
        COUNTER,
        GAUGE,
        HISTOGRAM
    };

    class MetricValue {
        public:
        MetricValue(const double& value, const uint64_t& timestamp, const std::string& request_id);
        double GetValue() const;
        uint64_t GetTimestamp() const;
        const std::string GetRequestId() const;

        private:
        const double value;
        const uint64_t timestamp;
        const std::string request_id;
    };

    class Metric {
        public:
        Metric(const std::string& name, const std::string& unit,
               const MetricType& type, const std::vector<Dimension>& dimensions);
        const std::string GetName() const;
        const std::string GetUnit() const;
        MetricType GetType() const;
        const std::vector<Dimension> GetDimensions() const;
        const std::vector<MetricValue> GetValues() const;
        void Update(const double& value, const std::string& request_id = "");
        void Reset();

        private:
        const std::string name;
        const std::string unit;
        const MetricType type;
        const std::vector<Dimension> dimensions;
        std::vector<MetricValue> values;
    };
} // namespace torchserve

#endif // TS_CPP_UTILS_METRICS_METRIC_HH_
