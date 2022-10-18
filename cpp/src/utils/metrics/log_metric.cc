#include <chrono>
#include <boost/asio.hpp>
#include <stdexcept>

#include "src/utils/logging.hh"
#include "src/utils/metrics/log_metric.hh"

namespace torchserve {
    void TSLogMetric::AddOrUpdate(const std::map<std::string, std::string>& dimension_values, const double& value) {
        AddOrUpdate(dimension_values, "", value);
    }

    void TSLogMetric::AddOrUpdate(const std::map<std::string, std::string>& dimension_values,
                                  const std::string& request_id, const double& value) {
        try {
            ValidateDimensionValues(dimension_values);
            ValidateMetricValue(value);
        }
        catch (const std::invalid_argument& exception) {
            std::string dimension_values_string = "";
            for(auto iter = dimension_values.begin(); iter != dimension_values.end(); iter++) {
                dimension_values_string += iter->first + ":" + iter->second;
                if (std::next(iter) != dimension_values.end()) {
                    dimension_values_string += ",";
                }
            }

            TS_LOGF(ERROR, "Failed to update metric with name: {} and dimension values: {} with value: {}. {}",
                    name, dimension_values_string, value, exception.what());
            return;
        }

        Emit(dimension_values, value, request_id);
    }

    void TSLogMetric::ValidateDimensionValues(const std::map<std::string, std::string>& dimension_values) {
        std::set<std::string> input_dimension_names = {};
        for(auto iter = dimension_values.begin(); iter != dimension_values.end(); iter++) {
            input_dimension_names.insert(iter->first);
            if(iter->second.empty()) {
                std::string error_message = "Dimension value corresponding to dimension name "
                                            + iter->first + "is empty";
                throw std::invalid_argument(error_message);
            }
        }

        if(input_dimension_names != dimension_names) {
            std::string dimension_names_string = "";
            for(auto iter = dimension_names.begin(); iter != dimension_names.end(); iter++) {
                dimension_names_string += *iter;
                if (std::next(iter) != dimension_names.end()) {
                    dimension_names_string += ", ";
                }
            }
            std::string error_message = "Provide dimension values corresponding to dimension names: "
                                        + dimension_names_string;
            throw std::invalid_argument(error_message);
        }
    }

    void TSLogMetric::ValidateMetricValue(const double& value) {
        if(type == MetricType::COUNTER && value < 0.0) {
            throw std::invalid_argument("Counter metric update value cannot be negative");
        }
    }

    void TSLogMetric::Emit(const std::map<std::string, std::string>& dimension_values,
                           const double& value, const std::string& request_id) {
        std::string dimension_values_string = "";
        for(auto iter = dimension_values.begin(); iter != dimension_values.end(); iter++) {
            dimension_values_string += iter->first + ":" + iter->second;
            if (std::next(iter) != dimension_values.end()) {
                dimension_values_string += ",";
            }
        }
        std::string hostname = ::boost::asio::ip::host_name();
        std::uint64_t timestamp = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

        if (request_id.empty()) {
            TS_LOGF(INFO, "[METRICS]{}.{}:{}|#{}|#hostname:{},{}",
                    name, unit, value, dimension_values_string, hostname, timestamp);
        }
        else {
            TS_LOGF(INFO, "[METRICS]{}.{}:{}|#{}|#hostname:{},{},{}",
                    name, unit, value, dimension_values_string, hostname, timestamp, request_id);
        }
    }
} // namespace torchserve
