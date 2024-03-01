#ifndef TS_CPP_UTILS_METRICS_YAML_CONFIG_HH_
#define TS_CPP_UTILS_METRICS_YAML_CONFIG_HH_

#include <yaml-cpp/yaml.h>

#include <set>
#include <string>
#include <vector>

#include "src/utils/metrics/config.hh"
#include "src/utils/metrics/metric.hh"

namespace torchserve {
class YAMLMetricsConfigurationHandler : public MetricsConfigurationHandler {
 public:
  YAMLMetricsConfigurationHandler()
      : mode{MetricsMode::LOG},
        dimension_names{},
        model_metrics{},
        ts_metrics{} {}
  ~YAMLMetricsConfigurationHandler() override {}
  void LoadConfiguration(const std::string& metrics_config_file_path,
                         const MetricsContext& metrics_context) override;
  const MetricsMode& GetMode() const override;
  const std::set<std::string>& GetDimensionNames() const override;
  const std::vector<MetricConfiguration>& GetModelMetrics() const override;
  const std::vector<MetricConfiguration>& GetTsMetrics() const override;

 private:
  MetricsMode mode;
  std::set<std::string> dimension_names;
  std::vector<MetricConfiguration> model_metrics;
  std::vector<MetricConfiguration> ts_metrics;

  void ClearConfiguration();
  void ParseMode(const YAML::Node& document_node);
  void ParseDimensionNames(const YAML::Node& document_node);
  void ParseModelMetrics(const YAML::Node& document_node);
  void ParseTsMetrics(const YAML::Node& document_node);
  void ParseMetricTypes(const YAML::Node& metric_types_node,
                        std::vector<MetricConfiguration>& metrics_config_store);
  void ParseMetrics(const YAML::Node& metrics_list_node,
                    const MetricType& metric_type,
                    std::vector<MetricConfiguration>& metrics_config_store);
  void ValidateMetricConfiguration(const MetricConfiguration& metric_config);
  void ValidateMetricNames();
};
}  // namespace torchserve

#endif  // TS_CPP_UTILS_METRICS_YAML_CONFIG_HH_
