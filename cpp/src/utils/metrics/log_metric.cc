#include "src/utils/metrics/log_metric.hh"

#include <algorithm>
#include <boost/asio.hpp>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/utils/logging.hh"

namespace torchserve {
void TSLogMetric::AddOrUpdate(const std::vector<std::string>& dimension_values,
                              const double& value) {
  AddOrUpdate(dimension_values, "", value);
}

void TSLogMetric::AddOrUpdate(const std::vector<std::string>& dimension_values,
                              const std::string& request_id,
                              const double& value) {
  try {
    ValidateDimensionValues(dimension_values);
    ValidateMetricValue(value);
  } catch (const std::invalid_argument& exception) {
    std::string error_message =
        "Failed to update metric with name: " + name +
        " and dimensions: " + BuildDimensionsString(dimension_values) +
        " with value: " + std::to_string(value) + ". " + exception.what();
    throw std::invalid_argument(error_message);
  }

  Emit(dimension_values, request_id, value);
}

std::string TSLogMetric::BuildDimensionsString(
    const std::vector<std::string>& dimension_values) {
  std::string dimensions_string = "";
  for (auto name_iter = dimension_names.begin(),
            value_iter = dimension_values.begin();
       name_iter != dimension_names.end() &&
       value_iter != dimension_values.end();
       name_iter++, value_iter++) {
    dimensions_string += *name_iter + ":" + *value_iter;
    if (std::next(name_iter) != dimension_names.end() &&
        std::next(value_iter) != dimension_values.end()) {
      dimensions_string += ",";
    }
  }

  return dimensions_string;
}

void TSLogMetric::ValidateDimensionValues(
    const std::vector<std::string>& dimension_values) {
  if (dimension_values.size() != dimension_names.size() ||
      std::find(dimension_values.begin(), dimension_values.end(), "") !=
          dimension_values.end()) {
    std::string dimension_names_string = "";
    for (auto iter = dimension_names.begin(); iter != dimension_names.end();
         iter++) {
      dimension_names_string += *iter;
      if (std::next(iter) != dimension_names.end()) {
        dimension_names_string += ", ";
      }
    }
    std::string error_message =
        "Dimension values not provided corresponding to dimension names: " +
        dimension_names_string;
    throw std::invalid_argument(error_message);
  }
}

void TSLogMetric::ValidateMetricValue(const double& value) {
  if (type == MetricType::COUNTER && value < 0.0) {
    throw std::invalid_argument(
        "Counter metric update value cannot be negative");
  }
}

void TSLogMetric::Emit(const std::vector<std::string>& dimension_values,
                       const std::string& request_id, const double& value) {
  std::string hostname = ::boost::asio::ip::host_name();
  std::uint64_t timestamp =
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();

  if (request_id.empty()) {
    TS_LOGF(INFO, "[METRICS]{}.{}:{}|#{}|#hostname:{},{}", name, unit, value,
            BuildDimensionsString(dimension_values), hostname, timestamp);
  } else {
    TS_LOGF(INFO, "[METRICS]{}.{}:{}|#{}|#hostname:{},{},{}", name, unit, value,
            BuildDimensionsString(dimension_values), hostname, timestamp,
            request_id);
  }
}
}  // namespace torchserve
