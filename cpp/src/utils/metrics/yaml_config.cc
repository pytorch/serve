#include "src/utils/metrics/yaml_config.hh"

#include <exception>
#include <set>

#include "src/utils/logging.hh"

namespace torchserve {
void YAMLMetricsConfigurationHandler::LoadConfiguration(
    const std::string& metrics_config_file_path,
    const MetricsContext& metrics_context) {
  ClearConfiguration();

  try {
    YAML::Node config_node = YAML::LoadFile(metrics_config_file_path);
    ParseMode(config_node);
    ParseDimensionNames(config_node);
    ParseModelMetrics(config_node);
    if (metrics_context == MetricsContext::FRONTEND) {
      ParseTsMetrics(config_node);
    }
    ValidateMetricNames();
  } catch (YAML::ParserException& e) {
    ClearConfiguration();
    std::string error_message =
        "Failed to parse metrics YAML configuration file: " +
        metrics_config_file_path + ". " + e.what();
    throw std::invalid_argument(error_message);
  } catch (YAML::RepresentationException& e) {
    ClearConfiguration();
    std::string error_message =
        "Failed to interpret metrics YAML configuration file: " +
        metrics_config_file_path + ". " + e.what();
    throw std::invalid_argument(error_message);
  } catch (YAML::Exception& e) {
    ClearConfiguration();
    std::string error_message =
        "Failed to load metrics YAML configuration file: " +
        metrics_config_file_path + ". " + e.what();
    throw std::invalid_argument(error_message);
  }
}

const MetricsMode& YAMLMetricsConfigurationHandler::GetMode() const {
  return mode;
}

const std::set<std::string>&
YAMLMetricsConfigurationHandler::GetDimensionNames() const {
  return dimension_names;
}

const std::vector<MetricConfiguration>&
YAMLMetricsConfigurationHandler::GetModelMetrics() const {
  return model_metrics;
}

const std::vector<MetricConfiguration>&
YAMLMetricsConfigurationHandler::GetTsMetrics() const {
  return ts_metrics;
}

void YAMLMetricsConfigurationHandler::ClearConfiguration() {
  mode = MetricsMode::LOG;
  dimension_names.clear();
  model_metrics.clear();
  ts_metrics.clear();
}

void YAMLMetricsConfigurationHandler::ParseMode(
    const YAML::Node& document_node) {
  if (document_node["mode"] &&
      document_node["mode"].as<std::string>() == "prometheus") {
    mode = MetricsMode::PROMETHEUS;
  } else {
    mode = MetricsMode::LOG;
  }
}

void YAMLMetricsConfigurationHandler::ParseDimensionNames(
    const YAML::Node& document_node) {
  if (!document_node["dimensions"]) {
    return;
  }

  const std::vector<std::string> dimension_names_list =
      document_node["dimensions"].as<std::vector<std::string>>();

  for (const auto& name : dimension_names_list) {
    if (dimension_names.insert(name).second == false) {
      throw YAML::Exception(YAML::Mark::null_mark(),
                            "Dimension names defined under central "
                            "\"dimensions\" key must be unique");
    }
    if (name.empty()) {
      throw YAML::Exception(
          YAML::Mark::null_mark(),
          "Dimension names defined under \"dimensions\" key cannot be emtpy");
    }
  }
}

void YAMLMetricsConfigurationHandler::ParseModelMetrics(
    const YAML::Node& document_node) {
  if (document_node["model_metrics"]) {
    ParseMetricTypes(document_node["model_metrics"], model_metrics);
  }
}

void YAMLMetricsConfigurationHandler::ParseTsMetrics(
    const YAML::Node& document_node) {
  if (document_node["ts_metrics"]) {
    ParseMetricTypes(document_node["ts_metrics"], ts_metrics);
  }
}

void YAMLMetricsConfigurationHandler::ParseMetricTypes(
    const YAML::Node& metric_types_node,
    std::vector<MetricConfiguration>& metrics_config_store) {
  if (metric_types_node["counter"]) {
    ParseMetrics(metric_types_node["counter"], MetricType::COUNTER,
                 metrics_config_store);
  }

  if (metric_types_node["gauge"]) {
    ParseMetrics(metric_types_node["gauge"], MetricType::GAUGE,
                 metrics_config_store);
  }

  if (metric_types_node["histogram"]) {
    ParseMetrics(metric_types_node["histogram"], MetricType::HISTOGRAM,
                 metrics_config_store);
  }
}

void YAMLMetricsConfigurationHandler::ParseMetrics(
    const YAML::Node& metrics_list_node, const MetricType& metric_type,
    std::vector<MetricConfiguration>& metrics_config_store) {
  std::vector<MetricConfiguration> metrics_config =
      metrics_list_node.as<std::vector<MetricConfiguration>>();
  for (auto& config : metrics_config) {
    config.type = metric_type;
    ValidateMetricConfiguration(config);
  }

  metrics_config_store.insert(metrics_config_store.end(),
                              metrics_config.begin(), metrics_config.end());
}

void YAMLMetricsConfigurationHandler::ValidateMetricConfiguration(
    const MetricConfiguration& metric_config) {
  std::set<std::string> metric_dimensions_set{};

  for (const auto& name : metric_config.dimension_names) {
    if (dimension_names.find(name) == dimension_names.end()) {
      std::string error_message =
          "Dimension \"" + name +
          "\" associated with metric: " + metric_config.name +
          " not defined under central \"dimensions\" key in configuration";
      throw YAML::Exception(YAML::Mark::null_mark(), error_message);
    }

    if (metric_dimensions_set.insert(name).second == false) {
      std::string error_message =
          "Dimensions for metric: " + metric_config.name + " must be unique";
      throw YAML::Exception(YAML::Mark::null_mark(), error_message);
    }
  }
}

void YAMLMetricsConfigurationHandler::ValidateMetricNames() {
  std::unordered_map<MetricType, std::set<std::string>> metric_names{};

  for (const auto& config : model_metrics) {
    if (metric_names[config.type].insert(config.name).second == false) {
      std::string error_message =
          "Metrics of a given type must have unique names. Duplicate metric "
          "name found: " +
          config.name;
      throw YAML::Exception(YAML::Mark::null_mark(), error_message);
    }
  }

  for (const auto& config : ts_metrics) {
    if (metric_names[config.type].insert(config.name).second == false) {
      std::string error_message =
          "Metrics of a given type must have unique names. Duplicate metric "
          "name found: " +
          config.name;
      throw YAML::Exception(YAML::Mark::null_mark(), error_message);
    }
  }
}
}  // namespace torchserve

namespace YAML {
template <>
struct convert<torchserve::MetricConfiguration> {
  static bool decode(const Node& metric_config_node,
                     torchserve::MetricConfiguration& metric_config) {
    if (!metric_config_node["name"] || !metric_config_node["unit"] ||
        !metric_config_node["dimensions"]) {
      TS_LOG(ERROR,
             "Configuration for a metric must consist of \"name\", "
             "\"unit\" and \"dimensions\"");
      return false;
    }

    metric_config.name = metric_config_node["name"].as<std::string>();
    metric_config.unit = metric_config_node["unit"].as<std::string>();
    metric_config.dimension_names =
        metric_config_node["dimensions"].as<std::vector<std::string>>();

    if (metric_config.name.empty()) {
      TS_LOG(ERROR,
             "Configuration for a metric must consist of a non-empty "
             "\"name\"");
      return false;
    }

    return true;
  }
};
}  // namespace YAML
