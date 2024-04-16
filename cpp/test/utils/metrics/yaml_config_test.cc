#include "src/utils/metrics/yaml_config.hh"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <set>
#include <string>

namespace torchserve {
TEST(YAMLConfigTest, TestLoadValidConfigFrontendContext) {
  YAMLMetricsConfigurationHandler config_handler =
      YAMLMetricsConfigurationHandler();
  ASSERT_EQ(config_handler.GetMode(), MetricsMode::LOG);
  ASSERT_TRUE(config_handler.GetDimensionNames().empty());
  ASSERT_TRUE(config_handler.GetTsMetrics().empty());
  ASSERT_TRUE(config_handler.GetModelMetrics().empty());

  const std::string config_file_path =
      "resources/metrics/valid_config.yaml";
  config_handler.LoadConfiguration(config_file_path, MetricsContext::FRONTEND);

  ASSERT_EQ(config_handler.GetMode(), MetricsMode::PROMETHEUS);

  std::set<std::string> expected_dimension_names = {"model_name", "host_name",
                                                    "level"};
  ASSERT_EQ(config_handler.GetDimensionNames(), expected_dimension_names);

  std::vector<MetricConfiguration> expected_model_metrics = {
      MetricConfiguration{MetricType::COUNTER,
                          "CounterModelMetricExample",
                          "count",
                          {"model_name", "host_name"}},
      MetricConfiguration{MetricType::COUNTER,
                          "AnotherCounterModelMetricExample",
                          "count",
                          {"model_name", "level"}},
      MetricConfiguration{MetricType::GAUGE,
                          "GaugeModelMetricExample",
                          "count",
                          {"model_name", "level"}},
      MetricConfiguration{MetricType::HISTOGRAM,
                          "HistogramModelMetricExample",
                          "ms",
                          {"model_name", "level"}}};
  ASSERT_EQ(config_handler.GetModelMetrics(), expected_model_metrics);

  std::vector<MetricConfiguration> expected_ts_metrics = {
      MetricConfiguration{MetricType::COUNTER,
                          "CounterTsMetricExample",
                          "count",
                          {"model_name", "host_name"}},
      MetricConfiguration{MetricType::GAUGE,
                          "GaugeTsMetricExample",
                          "count",
                          {"model_name", "host_name"}},
      MetricConfiguration{MetricType::HISTOGRAM,
                          "HistogramTsMetricExample",
                          "ms",
                          {"model_name", "host_name"}}};
  ASSERT_EQ(config_handler.GetTsMetrics(), expected_ts_metrics);
}

TEST(YAMLConfigTest, TestLoadValidConfigBackendContext) {
  YAMLMetricsConfigurationHandler config_handler =
      YAMLMetricsConfigurationHandler();
  ASSERT_EQ(config_handler.GetMode(), MetricsMode::LOG);
  ASSERT_TRUE(config_handler.GetDimensionNames().empty());
  ASSERT_TRUE(config_handler.GetTsMetrics().empty());
  ASSERT_TRUE(config_handler.GetModelMetrics().empty());

  const std::string config_file_path =
      "resources/metrics/valid_config.yaml";
  config_handler.LoadConfiguration(config_file_path, MetricsContext::BACKEND);

  std::set<std::string> expected_dimension_names = {"model_name", "host_name",
                                                    "level"};
  ASSERT_EQ(config_handler.GetDimensionNames(), expected_dimension_names);

  std::vector<MetricConfiguration> expected_model_metrics = {
      MetricConfiguration{MetricType::COUNTER,
                          "CounterModelMetricExample",
                          "count",
                          {"model_name", "host_name"}},
      MetricConfiguration{MetricType::COUNTER,
                          "AnotherCounterModelMetricExample",
                          "count",
                          {"model_name", "level"}},
      MetricConfiguration{MetricType::GAUGE,
                          "GaugeModelMetricExample",
                          "count",
                          {"model_name", "level"}},
      MetricConfiguration{MetricType::HISTOGRAM,
                          "HistogramModelMetricExample",
                          "ms",
                          {"model_name", "level"}}};
  ASSERT_EQ(config_handler.GetModelMetrics(), expected_model_metrics);

  ASSERT_TRUE(config_handler.GetTsMetrics().empty());
}

TEST(YAMLConfigTest, TestLoadMinimalValidConfig) {
  YAMLMetricsConfigurationHandler config_handler =
      YAMLMetricsConfigurationHandler();
  const std::string config_file_path =
      "resources/metrics/minimal_valid_config.yaml";
  config_handler.LoadConfiguration(config_file_path, MetricsContext::FRONTEND);

  ASSERT_EQ(config_handler.GetMode(), MetricsMode::LOG);
  ASSERT_TRUE(config_handler.GetDimensionNames().empty());
  ASSERT_TRUE(config_handler.GetTsMetrics().empty());

  std::vector<MetricConfiguration> expected_model_metrics = {
      MetricConfiguration{
          MetricType::HISTOGRAM, "HistogramModelMetricExample", "ms", {}}};
  ASSERT_EQ(config_handler.GetModelMetrics(), expected_model_metrics);
}

TEST(YAMLConfigTest, TestLoadInvalidConfigWithDuplicateDimension) {
  YAMLMetricsConfigurationHandler config_handler =
      YAMLMetricsConfigurationHandler();
  const std::string config_file_path =
      "resources/metrics/invalid_config_duplicate_dimension.yaml";
  ASSERT_THROW(config_handler.LoadConfiguration(config_file_path,
                                                MetricsContext::FRONTEND),
               std::invalid_argument);

  ASSERT_EQ(config_handler.GetMode(), MetricsMode::LOG);
  ASSERT_TRUE(config_handler.GetDimensionNames().empty());
  ASSERT_TRUE(config_handler.GetTsMetrics().empty());
  ASSERT_TRUE(config_handler.GetModelMetrics().empty());
}

TEST(YAMLConfigTest, TestLoadInvalidConfigWithEmptyDimension) {
  YAMLMetricsConfigurationHandler config_handler =
      YAMLMetricsConfigurationHandler();
  const std::string config_file_path =
      "resources/metrics/invalid_config_empty_dimension.yaml";
  ASSERT_THROW(config_handler.LoadConfiguration(config_file_path,
                                                MetricsContext::FRONTEND),
               std::invalid_argument);

  ASSERT_EQ(config_handler.GetMode(), MetricsMode::LOG);
  ASSERT_TRUE(config_handler.GetDimensionNames().empty());
  ASSERT_TRUE(config_handler.GetTsMetrics().empty());
  ASSERT_TRUE(config_handler.GetModelMetrics().empty());
}

TEST(YAMLConfigTest, TestLoadInvalidConfigWithUndefinedDimension) {
  YAMLMetricsConfigurationHandler config_handler =
      YAMLMetricsConfigurationHandler();
  const std::string config_file_path =
      "resources/metrics/invalid_config_undefined_dimension.yaml";
  ASSERT_THROW(config_handler.LoadConfiguration(config_file_path,
                                                MetricsContext::FRONTEND),
               std::invalid_argument);

  ASSERT_EQ(config_handler.GetMode(), MetricsMode::LOG);
  ASSERT_TRUE(config_handler.GetDimensionNames().empty());
  ASSERT_TRUE(config_handler.GetTsMetrics().empty());
  ASSERT_TRUE(config_handler.GetModelMetrics().empty());
}

TEST(YAMLConfigTest, TestLoadInvalidConfigWithDuplicateMetricDimension) {
  YAMLMetricsConfigurationHandler config_handler =
      YAMLMetricsConfigurationHandler();
  const std::string config_file_path =
      "resources/metrics/"
      "invalid_config_duplicate_metric_dimension.yaml";
  ASSERT_THROW(config_handler.LoadConfiguration(config_file_path,
                                                MetricsContext::FRONTEND),
               std::invalid_argument);

  ASSERT_EQ(config_handler.GetMode(), MetricsMode::LOG);
  ASSERT_TRUE(config_handler.GetDimensionNames().empty());
  ASSERT_TRUE(config_handler.GetTsMetrics().empty());
  ASSERT_TRUE(config_handler.GetModelMetrics().empty());
}

TEST(YAMLConfigTest, TestLoadInvalidConfigWithMissingMetricName) {
  YAMLMetricsConfigurationHandler config_handler =
      YAMLMetricsConfigurationHandler();
  const std::string config_file_path =
      "resources/metrics/"
      "invalid_config_missing_metric_name.yaml";
  ASSERT_THROW(config_handler.LoadConfiguration(config_file_path,
                                                MetricsContext::FRONTEND),
               std::invalid_argument);

  ASSERT_EQ(config_handler.GetMode(), MetricsMode::LOG);
  ASSERT_TRUE(config_handler.GetDimensionNames().empty());
  ASSERT_TRUE(config_handler.GetTsMetrics().empty());
  ASSERT_TRUE(config_handler.GetModelMetrics().empty());
}

TEST(YAMLConfigTest, TestLoadInvalidConfigWithEmptyMetricName) {
  YAMLMetricsConfigurationHandler config_handler =
      YAMLMetricsConfigurationHandler();
  const std::string config_file_path =
      "resources/metrics/"
      "invalid_config_empty_metric_name.yaml";
  ASSERT_THROW(config_handler.LoadConfiguration(config_file_path,
                                                MetricsContext::FRONTEND),
               std::invalid_argument);

  ASSERT_EQ(config_handler.GetMode(), MetricsMode::LOG);
  ASSERT_TRUE(config_handler.GetDimensionNames().empty());
  ASSERT_TRUE(config_handler.GetTsMetrics().empty());
  ASSERT_TRUE(config_handler.GetModelMetrics().empty());
}

TEST(YAMLConfigTest, TestLoadInvalidConfigWithDuplicateMetricName) {
  YAMLMetricsConfigurationHandler config_handler =
      YAMLMetricsConfigurationHandler();
  const std::string config_file_path =
      "resources/metrics/"
      "invalid_config_duplicate_metric_name.yaml";
  ASSERT_THROW(config_handler.LoadConfiguration(config_file_path,
                                                MetricsContext::FRONTEND),
               std::invalid_argument);

  ASSERT_EQ(config_handler.GetMode(), MetricsMode::LOG);
  ASSERT_TRUE(config_handler.GetDimensionNames().empty());
  ASSERT_TRUE(config_handler.GetTsMetrics().empty());
  ASSERT_TRUE(config_handler.GetModelMetrics().empty());
}
}  // namespace torchserve
