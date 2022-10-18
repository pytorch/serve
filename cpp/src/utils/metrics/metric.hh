#ifndef TS_CPP_UTILS_METRICS_METRIC_HH_
#define TS_CPP_UTILS_METRICS_METRIC_HH_

#include <string>
#include <set>
#include <map>

#include "src/utils/metrics/units.hh"

namespace torchserve {
    enum MetricType {
        COUNTER,
        GAUGE,
        HISTOGRAM
    };

    class IMetric {
        public:
        IMetric(const MetricType& type, const std::string& name, const std::string& unit,
                const std::set<std::string>& dimension_names) :
                type{type}, name{name}, unit{Units::GetUnitMapping(unit)}, dimension_names{dimension_names} {}
        virtual void AddOrUpdate(const std::map<std::string, std::string>& dimension_values, const double& value) = 0;

        protected:
        const MetricType type;
        const std::string name;
        const std::string unit;
        const std::set<std::string> dimension_names;
    };
} // namespace torchserve

#endif // TS_CPP_UTILS_METRICS_METRIC_HH_
