#include "src/utils/metrics/log_metric.hh"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

#include "src/utils/logging.hh"

namespace torchserve {

class TestLogger : public Logger {
    public:
    TestLogger(){};
    virtual ~TestLogger(){};
    void flush() { return Logger::logger->flush(); }
};

class TSLogMetricTest : public ::testing::Test {
 protected:
  const std::string logfile_path{"utils/metrics/metrics_test.log"};
  const std::string logger_config_path_str{
      "resources/metrics/log_to_file.yaml"};
  const std::string metric_name{"test_metric"};
  const std::vector<std::string> metric_dimension_names{"level", "model_name"};
  const std::vector<std::string> metric_dimension_values{"model", "test_model"};
  const std::string metric_request_id{"test_request_id"};
  const std::regex metric_log_regex{
      std::regex("\\s*([\\w\\s]+)\\.([\\w\\s]+):([0-9\\-,.e]+)\\|#([^|]*)\\|#"
                 "hostname:([^,]+),([^,]+)(,(.*))?")};

  void SetUp() override {
    std::filesystem::remove(logfile_path);
    torchserve::Logger::InitLogger(logger_config_path_str);
  }

  void TearDown() override {
    if (!std::filesystem::remove(logfile_path)) {
      std::cout << "Failed to delete test metric log file" << logfile_path
                << std::endl;
    };
    torchserve::Logger::InitDefaultLogger();
  }

  const std::vector<std::string> GetMetricLogs() {
    std::vector<std::string> metric_logs;
    std::ifstream logfile(logfile_path);
    if (!logfile.is_open()) {
      return metric_logs;
    }

    std::string log_line;
    while (std::getline(logfile, log_line)) {
      if (!log_line.empty()) {
        metric_logs.push_back(log_line);
      }
    }
    logfile.close();

    return metric_logs;
  }

  const std::vector<double> GetMetricValuesFromLogs() {
    std::vector<double> metric_values;
    const std::vector<std::string> metric_logs = GetMetricLogs();
    for (const auto& log : metric_logs) {
      std::smatch match;
      if (std::regex_search(log, match, metric_log_regex)) {
        metric_values.push_back(std::stod(match[3]));
      }
    }

    return metric_values;
  }
};

TEST_F(TSLogMetricTest, TestCounterMetric) {
  TSLogMetric test_metric(MetricType::COUNTER, metric_name, "ms",
                          metric_dimension_names);
  ASSERT_TRUE(GetMetricValuesFromLogs().empty());

  test_metric.AddOrUpdate(metric_dimension_values, 1.0);
  ASSERT_THROW(test_metric.AddOrUpdate(metric_dimension_values, -2.0),
               std::invalid_argument);
  ASSERT_THROW(
      test_metric.AddOrUpdate(metric_dimension_values, metric_request_id, -2.5),
      std::invalid_argument);
  test_metric.AddOrUpdate(metric_dimension_values, metric_request_id, 3.5);
  const std::vector<double> expected_metric_values{1.0, 3.5};
  TestLogger logger;
  logger.flush();
  ASSERT_EQ(GetMetricValuesFromLogs(), expected_metric_values);
}

TEST_F(TSLogMetricTest, TestGaugeMetric) {
  TSLogMetric test_metric(MetricType::GAUGE, metric_name, "count",
                          metric_dimension_names);
  ASSERT_TRUE(GetMetricValuesFromLogs().empty());

  test_metric.AddOrUpdate(metric_dimension_values, 1.0);
  test_metric.AddOrUpdate(metric_dimension_values, -2.0);
  test_metric.AddOrUpdate(metric_dimension_values, metric_request_id, 3.5);
  test_metric.AddOrUpdate(metric_dimension_values, metric_request_id, -4.0);
  const std::vector<double> expected_metric_values{1.0, -2.0, 3.5, -4.0};

  TestLogger logger;
  logger.flush();

  ASSERT_EQ(GetMetricValuesFromLogs(), expected_metric_values);
}

TEST_F(TSLogMetricTest, TestHistogramMetric) {
  TSLogMetric test_metric(MetricType::HISTOGRAM, metric_name, "count",
                          metric_dimension_names);
  ASSERT_TRUE(GetMetricValuesFromLogs().empty());

  test_metric.AddOrUpdate(metric_dimension_values, 1.0);
  test_metric.AddOrUpdate(metric_dimension_values, -2.0);
  test_metric.AddOrUpdate(metric_dimension_values, metric_request_id, 3.5);
  test_metric.AddOrUpdate(metric_dimension_values, metric_request_id, -4.0);
  const std::vector<double> expected_metric_values{1.0, -2.0, 3.5, -4.0};

  TestLogger logger;
  logger.flush();

  ASSERT_EQ(GetMetricValuesFromLogs(), expected_metric_values);
}

TEST_F(TSLogMetricTest, TestTSLogMetricEmitWithRequestId) {
  TSLogMetric test_metric(MetricType::COUNTER, metric_name, "ms",
                          metric_dimension_names);
  test_metric.AddOrUpdate(metric_dimension_values, metric_request_id, 1.5);
  TestLogger logger;
  logger.flush();
  const std::vector<std::string> metric_logs = GetMetricLogs();
  ASSERT_EQ(metric_logs.size(), 1);

  std::smatch match;
  if (!std::regex_search(metric_logs[0], match, metric_log_regex)) {
    FAIL();
  }
  ASSERT_EQ(match[1], "test_metric");
  ASSERT_EQ(match[2], "Milliseconds");
  ASSERT_EQ(match[3], "1.5");
  ASSERT_EQ(match[4], "level:model,model_name:test_model");
  ASSERT_EQ(match[8], "test_request_id");
}

TEST_F(TSLogMetricTest, TestTSLogMetricEmitWithoutRequestId) {
  TSLogMetric test_metric(MetricType::COUNTER, metric_name, "ms",
                          metric_dimension_names);
  test_metric.AddOrUpdate(metric_dimension_values, 1.5);
  TestLogger logger;
  logger.flush();
  const std::vector<std::string> metric_logs = GetMetricLogs();
  ASSERT_EQ(metric_logs.size(), 1);

  std::smatch match;
  if (!std::regex_search(metric_logs[0], match, metric_log_regex)) {
    FAIL();
  }
  ASSERT_EQ(match[1], "test_metric");
  ASSERT_EQ(match[2], "Milliseconds");
  ASSERT_EQ(match[3], "1.5");
  ASSERT_EQ(match[4], "level:model,model_name:test_model");
  ASSERT_EQ(match[8], "");
}

TEST_F(TSLogMetricTest, TestTSLogMetricEmitWithIncorrectDimensionData) {
  TSLogMetric test_metric(MetricType::COUNTER, metric_name, "ms",
                          metric_dimension_names);
  ASSERT_THROW(test_metric.AddOrUpdate(std::vector<std::string>{"model"}, 1.5),
               std::invalid_argument);
  ASSERT_THROW(
      test_metric.AddOrUpdate(std::vector<std::string>{"model", ""}, 1.5),
      std::invalid_argument);
  ASSERT_THROW(
      test_metric.AddOrUpdate(
          std::vector<std::string>{"model", "test_model", "extra_dim"}, 1.5),
      std::invalid_argument);
  ASSERT_EQ(GetMetricLogs().size(), 0);
  ASSERT_TRUE(GetMetricValuesFromLogs().empty());
}
}  // namespace torchserve
