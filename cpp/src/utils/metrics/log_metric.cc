#include <chrono>
#include <boost/asio.hpp>

#include "src/utils/metrics/units.hh"
#include "src/utils/logging.hh"
#include "src/utils/metrics/log_metric.hh"

namespace torchserve {
    LogMetric::LogMetric(const std::string& name, const std::string& unit,
                         const std::map<std::string, std::string>& dimensions, const MetricType& type) :
                         name{name}, unit{Units::GetUnitMapping(unit)}, dimensions{dimensions}, type{type} {}

    void LogMetric::Update(const double& value, const std::string& request_id) {
        if(type == MetricType::COUNTER && value < 0.0) {
            return;
        }

        uint64_t time_since_epoch_in_seconds = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        values.push_back(std::make_tuple(value, time_since_epoch_in_seconds, request_id));
    }

    void LogMetric::Reset() {
        values.clear();
    }

    void LogMetric::Emit() {
        std::string dimensions_string = "";
        for(auto iter = dimensions.begin(); iter != dimensions.end(); iter++) {
            dimensions_string += iter->first + ":" + iter->second;
            if (std::next(iter) != dimensions.end()) {
                dimensions_string += ",";
            }
        }
        std::string hostname = ::boost::asio::ip::host_name();

        for(const auto& value_entry : values) {
            double value;
            std::uint64_t timestamp;
            std::string request_id;
            std::tie(value, timestamp, request_id) = value_entry;

            if (request_id.empty()) {
                TS_LOGF(INFO, "[METRICS]{}.{}:{}|#{}|#hostname:{},{}",
                        name, unit, value, dimensions_string, hostname, timestamp);
            }
            else {
                TS_LOGF(INFO, "[METRICS]{}.{}:{}|#{}|#hostname:{},{},{}",
                        name, unit, value, dimensions_string, hostname, timestamp, request_id);
            }
        }
    }
} // namespace torchserve
