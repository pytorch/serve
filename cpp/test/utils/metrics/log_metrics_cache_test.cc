#include "src/utils/metrics/log_metrics_cache.hh"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdexcept>

#include "src/utils/metrics/yaml_config.hh"

namespace torchserve {
class TSLogMetricsCacheTest : public ::testing::Test {
 protected:
  const std::string config_file_path{
      "resources/metrics/valid_config.yaml"};
  MetricsConfigurationHandler* config_handler{nullptr};
  LogMetricsCache* cache{nullptr};

  void SetUp() override {
    config_handler = new YAMLMetricsConfigurationHandler();
    cache = new LogMetricsCache();
  }

  void TearDown() override {
    delete cache;
    delete config_handler;
  }
};

TEST_F(TSLogMetricsCacheTest, TestInitialize) {
  cache->Initialize(*config_handler);
  config_handler->LoadConfiguration(config_file_path, MetricsContext::BACKEND);
  cache->Initialize(*config_handler);
  config_handler->LoadConfiguration(config_file_path, MetricsContext::FRONTEND);
  cache->Initialize(*config_handler);
}

TEST_F(TSLogMetricsCacheTest, TestGetMetric) {
  config_handler->LoadConfiguration(config_file_path, MetricsContext::FRONTEND);
  cache->Initialize(*config_handler);

  ASSERT_THAT(cache->GetMetric(MetricType::COUNTER, "CounterTsMetricExample"),
              ::testing::A<TSLogMetric>());
  ASSERT_THAT(cache->GetMetric(MetricType::GAUGE, "GaugeTsMetricExample"),
              ::testing::A<TSLogMetric>());
  ASSERT_THAT(
      cache->GetMetric(MetricType::HISTOGRAM, "HistogramTsMetricExample"),
      ::testing::A<TSLogMetric>());
  ASSERT_THAT(
      cache->GetMetric(MetricType::COUNTER, "CounterModelMetricExample"),
      ::testing::A<TSLogMetric>());
  ASSERT_THAT(
      cache->GetMetric(MetricType::COUNTER, "AnotherCounterModelMetricExample"),
      ::testing::A<TSLogMetric>());
  ASSERT_THAT(cache->GetMetric(MetricType::GAUGE, "GaugeModelMetricExample"),
              ::testing::A<TSLogMetric>());
  ASSERT_THAT(
      cache->GetMetric(MetricType::HISTOGRAM, "HistogramModelMetricExample"),
      ::testing::A<TSLogMetric>());

  auto& metric = cache->GetMetric(MetricType::GAUGE, "GaugeTsMetricExample");
  metric.AddOrUpdate(std::vector<std::string>{"model_name", "host_name"}, 1.5);

  ASSERT_THROW(cache->GetMetric(MetricType::GAUGE, "InvalidMetricName"),
               std::invalid_argument);
}
}  // namespace torchserve
