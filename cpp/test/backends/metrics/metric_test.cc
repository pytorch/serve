#include <string>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <regex>

#include "src/backends/metrics/metric.hh"

namespace torchserve {
    class MetricTest : public ::testing::Test {
        protected:
        static const std::string metric_name;
        static const std::vector<Dimension> metric_dimensions;
        static const std::string metric_request_id;
        static const std::regex metric_string_regex;

        static std::string extract_metric_value(const std::string& metric_string);
    };

    const std::string MetricTest::metric_name {"test_metric"};
    const std::vector<Dimension> MetricTest::metric_dimensions {
        Dimension("level", "model"),
        Dimension("model_name", "test_model")
    };
    const std::string MetricTest::metric_request_id {"test_request_id"};
    const std::regex MetricTest::metric_string_regex {
        std::regex("^.*\\..*\\:(([0-9]+\\.)?[0-9]+)\\|#.*\\|#hostname\\:.*,[0-9]+(,.*)?$")};

    std::string MetricTest::extract_metric_value(const std::string& metric_string) {
        std::smatch match;
        if(std::regex_match(metric_string.begin(), metric_string.end(), match, metric_string_regex)) {
            return match[1];
        }
        return "";
    }

    TEST_F(MetricTest, TestCounterMetric) {
        Metric<int> test_metric(
                MetricTest::metric_name, MetricType::COUNTER, 0, "ms",
                MetricTest::metric_dimensions, MetricTest::metric_request_id, false);
        ASSERT_FALSE(test_metric.IsUpdated());
        ASSERT_EQ(MetricTest::extract_metric_value(test_metric.ToString()), std::to_string(0));

        test_metric.Update(1);
        ASSERT_TRUE(test_metric.IsUpdated());
        ASSERT_EQ(MetricTest::extract_metric_value(test_metric.ToString()), std::to_string(1));

        test_metric.Update(2);
        ASSERT_TRUE(test_metric.IsUpdated());
        ASSERT_EQ(MetricTest::extract_metric_value(test_metric.ToString()), std::to_string(3));

        test_metric.Reset();
        ASSERT_FALSE(test_metric.IsUpdated());
    }

    TEST_F(MetricTest, TestGaugeMetric) {
        Metric<int> test_metric(
            MetricTest::metric_name, MetricType::GAUGE, 0, "ms",
            MetricTest::metric_dimensions, MetricTest::metric_request_id, false);
        ASSERT_FALSE(test_metric.IsUpdated());
        ASSERT_EQ(MetricTest::extract_metric_value(test_metric.ToString()), std::to_string(0));

        test_metric.Update(1);
        ASSERT_TRUE(test_metric.IsUpdated());
        ASSERT_EQ(MetricTest::extract_metric_value(test_metric.ToString()), std::to_string(1));

        test_metric.Update(2);
        ASSERT_TRUE(test_metric.IsUpdated());
        ASSERT_EQ(MetricTest::extract_metric_value(test_metric.ToString()), std::to_string(2));

        test_metric.Reset();
        ASSERT_FALSE(test_metric.IsUpdated());
    }

    TEST_F(MetricTest, TestMetricWithDoubleValue) {
        Metric<double> test_metric(
                MetricTest::metric_name, MetricType::COUNTER, 0.0, "ms",
                MetricTest::metric_dimensions, MetricTest::metric_request_id, false);
        ASSERT_FALSE(test_metric.IsUpdated());
        ASSERT_EQ(MetricTest::extract_metric_value(test_metric.ToString()), std::to_string(0.0));

        test_metric.Update(1.1);
        ASSERT_TRUE(test_metric.IsUpdated());
        ASSERT_EQ(MetricTest::extract_metric_value(test_metric.ToString()), std::to_string(1.1));

        test_metric.Update(2.1);
        ASSERT_TRUE(test_metric.IsUpdated());
        ASSERT_EQ(MetricTest::extract_metric_value(test_metric.ToString()), std::to_string(3.2));

        test_metric.Reset();
        ASSERT_FALSE(test_metric.IsUpdated());
    }

    TEST_F(MetricTest, TestMetricWithoutRequestID) {
        Metric<int> test_metric(MetricTest::metric_name, MetricType::COUNTER, 0, "ms", MetricTest::metric_dimensions);
        ASSERT_FALSE(test_metric.IsUpdated());
        ASSERT_EQ(MetricTest::extract_metric_value(test_metric.ToString()), std::to_string(0));

        test_metric.Update(1);
        ASSERT_TRUE(test_metric.IsUpdated());
        ASSERT_EQ(MetricTest::extract_metric_value(test_metric.ToString()), std::to_string(1));

        test_metric.Update(2);
        ASSERT_TRUE(test_metric.IsUpdated());
        ASSERT_EQ(MetricTest::extract_metric_value(test_metric.ToString()), std::to_string(3));

        test_metric.Reset();
        ASSERT_FALSE(test_metric.IsUpdated());
    }

    TEST_F(MetricTest, TestMetricToStringWithRequestID) {
        Metric<int> test_metric(
                MetricTest::metric_name, MetricType::COUNTER, 0, "ms",
                MetricTest::metric_dimensions, MetricTest::metric_request_id, false);
        const std::string expected_metric_string_regex {
            "^test_metric\\.Milliseconds\\:0\\|#level\\:model,model_name\\:test_model\\|#hostname\\:.*,[0-9]+,test_request_id$"};
        ASSERT_THAT(test_metric.ToString(), ::testing::MatchesRegex(expected_metric_string_regex));
    }

    TEST_F(MetricTest, TestMetricToStringWithoutRequestID) {
        Metric<int> test_metric(MetricTest::metric_name, MetricType::COUNTER, 0, "ms", MetricTest::metric_dimensions);
        const std::string expected_metric_string_regex {
            "^test_metric\\.Milliseconds\\:0\\|#level\\:model,model_name\\:test_model\\|#hostname\\:.*,[0-9]+$"};
        ASSERT_THAT(test_metric.ToString(), ::testing::MatchesRegex(expected_metric_string_regex));
    }
} // namespace torchserve
