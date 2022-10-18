#include <string>
#include <set>
#include <map>
#include <filesystem>
#include <fstream>
#include <regex>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "src/utils/logging.hh"
#include "src/utils/metrics/log_metric.hh"

namespace torchserve {
    class TSLogMetricTest : public ::testing::Test {
        protected:
        const std::string logfile_path {"test/resources/logging/test.log"};
        const std::string logger_config_path_str {"test/resources/logging/log_to_file.config"};
        const std::string metric_name {"test_metric"};
        const std::set<std::string> metric_dimension_names {"model_name", "level"};
        const std::map<std::string, std::string> metric_dimension_values {
            {"model_name", "test_model"},
            {"level", "model"}
        };
        const std::string metric_request_id {"test_request_id"};
        const std::regex metric_log_regex {std::regex(
                "^.*\\[METRICS\\].*\\..*\\:(\\-?([0-9]+\\.)?[0-9]+)\\|#.*\\|#hostname\\:.*,[0-9]+(,.*)?$")};

        void SetUp() {
            torchserve::Logger::InitLogger(logger_config_path_str);
        }

        void TearDown() {
            if (!std::filesystem::remove(logfile_path)) {
                std::cout << "Failed to delete test metric log file" << logfile_path << std::endl;
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
            while(std::getline(logfile, log_line)) {
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
                if(std::regex_match(log.begin(), log.end(), match, metric_log_regex)) {
                    metric_values.push_back(std::stod(match[1]));
                }
            }

            return metric_values;
        }
    };

    TEST_F(TSLogMetricTest, TestCounterMetric) {
        TSLogMetric test_metric(MetricType::COUNTER, metric_name, "ms", metric_dimension_names);
        ASSERT_TRUE(GetMetricValuesFromLogs().empty());

        test_metric.AddOrUpdate(metric_dimension_values, 1.0);
        test_metric.AddOrUpdate(metric_dimension_values, -2.0);
        test_metric.AddOrUpdate(metric_dimension_values, metric_request_id, -2.5);
        test_metric.AddOrUpdate(metric_dimension_values, metric_request_id, 3.5);
        const std::vector<double> expected_metric_values {1.0, 3.5};
        ASSERT_EQ(GetMetricValuesFromLogs(), expected_metric_values);
    }

    TEST_F(TSLogMetricTest, TestGaugeMetric) {
        TSLogMetric test_metric(MetricType::GAUGE, metric_name, "count", metric_dimension_names);
        ASSERT_TRUE(GetMetricValuesFromLogs().empty());

        test_metric.AddOrUpdate(metric_dimension_values, 1.0);
        test_metric.AddOrUpdate(metric_dimension_values, -2.0);
        test_metric.AddOrUpdate(metric_dimension_values, metric_request_id, 3.5);
        test_metric.AddOrUpdate(metric_dimension_values, metric_request_id, -4.0);
        const std::vector<double> expected_metric_values {1.0, -2.0, 3.5, -4.0};
        ASSERT_EQ(GetMetricValuesFromLogs(), expected_metric_values);
    }

    TEST_F(TSLogMetricTest, TestHistogramMetric) {
        TSLogMetric test_metric(MetricType::HISTOGRAM, metric_name, "count", metric_dimension_names);
        ASSERT_TRUE(GetMetricValuesFromLogs().empty());

        test_metric.AddOrUpdate(metric_dimension_values, 1.0);
        test_metric.AddOrUpdate(metric_dimension_values, -2.0);
        test_metric.AddOrUpdate(metric_dimension_values, metric_request_id, 3.5);
        test_metric.AddOrUpdate(metric_dimension_values, metric_request_id, -4.0);
        const std::vector<double> expected_metric_values {1.0, -2.0, 3.5, -4.0};
        ASSERT_EQ(GetMetricValuesFromLogs(), expected_metric_values);
    }

    TEST_F(TSLogMetricTest, TestTSLogMetricEmitWithRequestId) {
        TSLogMetric test_metric(MetricType::COUNTER, metric_name, "ms", metric_dimension_names);
        test_metric.AddOrUpdate(metric_dimension_values, metric_request_id, 1.5);
        const std::vector<std::string> metric_logs = GetMetricLogs();
        const std::string expected_metric_log_pattern =
                "^.*\\[METRICS\\]test_metric\\.Milliseconds\\:1\\.5\\|#level\\:model,model_name\\:test_model\\"
                "|#hostname\\:.*,[0-9]+,test_request_id?$";
        ASSERT_EQ(metric_logs.size(), 1);
        ASSERT_THAT(metric_logs[0], ::testing::MatchesRegex(expected_metric_log_pattern));
    }

    TEST_F(TSLogMetricTest, TestTSLogMetricEmitWithoutRequestId) {
        TSLogMetric test_metric(MetricType::COUNTER, metric_name, "ms", metric_dimension_names);
        test_metric.AddOrUpdate(metric_dimension_values, 1.5);
        const std::vector<std::string> metric_logs = GetMetricLogs();
        const std::string expected_metric_log_pattern =
                "^.*\\[METRICS\\]test_metric\\.Milliseconds\\:1\\.5\\|#level\\:model,model_name\\:test_model\\"
                "|#hostname\\:.*,[0-9]+$";
        ASSERT_EQ(metric_logs.size(), 1);
        ASSERT_THAT(metric_logs[0], ::testing::MatchesRegex(expected_metric_log_pattern));
    }

    TEST_F(TSLogMetricTest, TestTSLogMetricEmitWithIncorrectDimensionData) {
        TSLogMetric test_metric(MetricType::COUNTER, metric_name, "ms", metric_dimension_names);
        test_metric.AddOrUpdate(std::map<std::string, std::string> {{"level", "model"}}, 1.5);
        test_metric.AddOrUpdate(std::map<std::string, std::string> {{"level", ""}, {"model_name", ""}}, 1.5);
        ASSERT_EQ(GetMetricLogs().size(), 2);
        ASSERT_TRUE(GetMetricValuesFromLogs().empty());
    }
} // namespace torchserve
