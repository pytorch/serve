#include <chrono>

#include "src/utils/metrics/units.hh"
#include "src/utils/metrics/metric.hh"

namespace torchserve {
    MetricValue::MetricValue(const double& value, const uint64_t& timestamp, const std::string& request_id) :
                             value{value}, timestamp{timestamp}, request_id{request_id} {}

    double MetricValue::GetValue() const {
        return value;
    }

    uint64_t MetricValue::GetTimestamp() const {
        return timestamp;
    }

    const std::string MetricValue::GetRequestId() const {
        return request_id;
    }

    Metric::Metric(const std::string& name, const std::string& unit,
                   const MetricType& type, const std::vector<Dimension>& dimensions) :
                   name{name}, unit{Units::GetUnitMapping(unit)}, type{type}, dimensions{dimensions} {}

    const std::string Metric::GetName() const {
        return name;
    }

    const std::string Metric::GetUnit() const {
        return unit;
    }

    MetricType Metric::GetType() const {
        return type;
    }

    const std::vector<Dimension> Metric::GetDimensions() const {
        return dimensions;
    }

    const std::vector<MetricValue> Metric::GetValues() const {
        return values;
    }

    void Metric::Update(const double& value, const std::string& request_id) {
        if(type == MetricType::COUNTER && value < 0.0) {
            return;
        }

        uint64_t time_since_epoch_in_seconds = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        values.push_back(MetricValue(value, time_since_epoch_in_seconds, request_id));
    }

    void Metric::Reset() {
        values.clear();
    }
} // namespace torchserve
