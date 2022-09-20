#include <chrono>
#include <boost/asio.hpp>

#include "src/backends/metrics/metric.hh"
#include "src/backends/metrics/units.hh"

namespace torchserve {
    template <typename ValueType>
    Metric<ValueType>::Metric(
            const std::string& name, const MetricType& type, const ValueType& value,
            const std::string& unit, const std::vector<Dimension>& dimensions,
            const std::string& request_id, const bool& is_updated) :
            name{name}, type{type}, value{value}, unit{Units::GetUnitMapping(unit)},
            dimensions{dimensions}, request_id{request_id}, is_updated{is_updated} {}

    template <typename ValueType>
    void Metric<ValueType>::Update(const ValueType& value) {
        ValueType previous_value = this->value;

        if(type == MetricType::COUNTER) {
            this->value += value;
        }
        else {
            this->value = value;
        }

        is_updated = this->value != previous_value;
    }

    template <typename ValueType>
    bool Metric<ValueType>::IsUpdated() const {
        return is_updated;
    }

    template <typename ValueType>
    void Metric<ValueType>::Reset() {
        is_updated = false;
    }

    template <typename ValueType>
    std::string Metric<ValueType>::ToString() const {
        std::string dimensions_string = dimensions.empty() ? "" : dimensions.at(0).ToString();
        for(unsigned int index=1; index<dimensions.size(); index++) {
            dimensions_string += "," + dimensions.at(index).ToString();
        }
        std::string hostname = ::boost::asio::ip::host_name();
        uint64_t time_since_epoch_in_seconds = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

        std::string output_string = name + "." + unit + ":" + std::to_string(value) + "|#" + dimensions_string +
                                    "|#hostname:" + hostname + "," + std::to_string(time_since_epoch_in_seconds);
        if(!request_id.empty()) {
            output_string += "," + request_id;
        }

        return output_string;
    }

    template class Metric<int>;
    template class Metric<double>;
} // namespace torchserve
