#ifndef TS_CPP_BACKENDS_METRICS_METRIC_HH_
#define TS_CPP_BACKENDS_METRICS_METRIC_HH_

#include <string>
#include <vector>

#include "src/backends/metrics/dimension.hh"

namespace torchserve {
    enum MetricType {
        COUNTER,
        GAUGE,
        HISTOGRAM
    };

    template <typename ValueType>
    class Metric {
        public:
        Metric(const std::string& name, const MetricType& type, const ValueType& value,
               const std::string& unit, const std::vector<Dimension>& dimensions,
               const std::string& request_id = "", const bool& is_updated = false);
        void Update(const ValueType& value);
        bool IsUpdated() const;
        void Reset();
        std::string ToString() const;

        private:
        const std::string name;
        const MetricType type;
        ValueType value;
        const std::string unit;
        const std::vector<Dimension> dimensions;
        const std::string request_id;
        bool is_updated;
    };

    extern template class Metric<int>;
    extern template class Metric<double>;
} // namespace torchserve

#endif // TS_CPP_BACKENDS_METRICS_METRIC_HH_
