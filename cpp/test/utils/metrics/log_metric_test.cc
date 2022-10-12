#include <string>
#include <map>
#include <filesystem>
#include <fstream>
#include <regex>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "src/utils/logging.hh"
#include "src/utils/metrics/log_metric.hh"

namespace torchserve {
    class LogMetricTest : public ::testing::Test {
        protected:
        const std::string logfile_path {"test/resources/logging/test.log"};
        const std::string logger_config_path_str {"test/resources/logging/log_to_file.config"};
        const std::string metric_name {"test_metric"};
        const std::map<std::string, std::string> metric_dimensions {
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

    TEST_F(LogMetricTest, TestCounterMetric) {
        LogMetric test_metric(
                LogMetricTest::metric_name, "ms", LogMetricTest::metric_dimensions,
                MetricType::COUNTER);
        test_metric.Emit();
        ASSERT_TRUE(GetMetricValuesFromLogs().empty());

        test_metric.Update(1.0);
        test_metric.Reset();
        test_metric.Emit();
        ASSERT_TRUE(GetMetricValuesFromLogs().empty());

        test_metric.Update(1.0);
        test_metric.Update(-2.0);
        test_metric.Update(-2.5, metric_request_id);
        test_metric.Update(3.5, metric_request_id);
        test_metric.Emit();
        const std::vector<double> expected_metric_values {1.0, 3.5};
        ASSERT_EQ(GetMetricValuesFromLogs(), expected_metric_values);
    }

    TEST_F(LogMetricTest, TestGaugeMetric) {
        LogMetric test_metric(
                LogMetricTest::metric_name, "count", LogMetricTest::metric_dimensions,
                MetricType::GAUGE);
        test_metric.Emit();
        ASSERT_TRUE(GetMetricValuesFromLogs().empty());

        test_metric.Update(1.0);
        test_metric.Reset();
        test_metric.Emit();
        ASSERT_TRUE(GetMetricValuesFromLogs().empty());

        test_metric.Update(1.0);
        test_metric.Update(-2.0);
        test_metric.Update(3.5, metric_request_id);
        test_metric.Update(-4.0, metric_request_id);
        test_metric.Emit();
        const std::vector<double> expected_metric_values {1.0, -2.0, 3.5, -4.0};
        ASSERT_EQ(GetMetricValuesFromLogs(), expected_metric_values);
    }

    TEST_F(LogMetricTest, TestHistogramMetric) {
        LogMetric test_metric(
                LogMetricTest::metric_name, "count", LogMetricTest::metric_dimensions,
                MetricType::HISTOGRAM);
        test_metric.Emit();
        ASSERT_TRUE(GetMetricValuesFromLogs().empty());

        test_metric.Update(1.0);
        test_metric.Reset();
        test_metric.Emit();
        ASSERT_TRUE(GetMetricValuesFromLogs().empty());

        test_metric.Update(1.0);
        test_metric.Update(-2.0);
        test_metric.Update(3.5, metric_request_id);
        test_metric.Update(-4.0, metric_request_id);
        test_metric.Emit();
        const std::vector<double> expected_metric_values {1.0, -2.0, 3.5, -4.0};
        ASSERT_EQ(GetMetricValuesFromLogs(), expected_metric_values);
    }

    TEST_F(LogMetricTest, TestLogMetricEmitWithRequestId) {
        LogMetric test_metric(metric_name, "ms", LogMetricTest::metric_dimensions, MetricType::COUNTER);
        test_metric.Update(1.5, metric_request_id);
        test_metric.Emit();
        const std::vector<std::string> metric_logs = GetMetricLogs();
        const std::string expected_metric_log_pattern =
                "^.*\\[METRICS\\]test_metric\\.Milliseconds\\:1\\.5\\|#level\\:model,model_name\\:test_model\\"
                "|#hostname\\:.*,[0-9]+,test_request_id?$";
        ASSERT_EQ(metric_logs.size(), 1);
        ASSERT_THAT(metric_logs[0], ::testing::MatchesRegex(expected_metric_log_pattern));
    }

    TEST_F(LogMetricTest, TestLogMetricEmitWithoutRequestId) {
        LogMetric test_metric(metric_name, "ms", LogMetricTest::metric_dimensions, MetricType::COUNTER);
        test_metric.Update(1.5);
        test_metric.Emit();
        const std::vector<std::string> metric_logs = GetMetricLogs();
        const std::string expected_metric_log_pattern =
                "^.*\\[METRICS\\]test_metric\\.Milliseconds\\:1\\.5\\|#level\\:model,model_name\\:test_model\\"
                "|#hostname\\:.*,[0-9]+$";
        ASSERT_EQ(metric_logs.size(), 1);
        ASSERT_THAT(metric_logs[0], ::testing::MatchesRegex(expected_metric_log_pattern));
    }
} // namespace torchserve
