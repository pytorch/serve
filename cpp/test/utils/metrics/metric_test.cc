#include <string>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "src/utils/metrics/metric.hh"

namespace torchserve {
    class MetricTest : public ::testing::Test {
        protected:
        static const std::string metric_name;
        static const std::vector<Dimension> metric_dimensions;
        static const std::string metric_request_id;
    };

    const std::string MetricTest::metric_name {"test_metric"};
    const std::vector<Dimension> MetricTest::metric_dimensions {
        Dimension("level", "model"),
        Dimension("model_name", "test_model")
    };
    const std::string MetricTest::metric_request_id {"test_request_id"};

    TEST_F(MetricTest, TestMetricValue) {
        MetricValue test_metric_value = MetricValue(1.0, 1664844858, MetricTest::metric_request_id);
        ASSERT_EQ(test_metric_value.GetValue(), 1.0);
        ASSERT_EQ(test_metric_value.GetTimestamp(), 1664844858);
        ASSERT_EQ(test_metric_value.GetRequestId(), MetricTest::metric_request_id);
    }

    TEST_F(MetricTest, TestMetric) {
        Metric test_metric(MetricTest::metric_name, "ms", MetricType::COUNTER, MetricTest::metric_dimensions);
        ASSERT_EQ(test_metric.GetName(), MetricTest::metric_name);
        ASSERT_EQ(test_metric.GetUnit(), "Milliseconds");
        ASSERT_EQ(test_metric.GetType(), MetricType::COUNTER);
        ASSERT_EQ(test_metric.GetDimensions().size(), 2);
        ASSERT_EQ(test_metric.GetDimensions().at(0).GetName(), MetricTest::metric_dimensions.at(0).GetName());
        ASSERT_EQ(test_metric.GetDimensions().at(0).GetValue(), MetricTest::metric_dimensions.at(0).GetValue());
        ASSERT_EQ(test_metric.GetDimensions().at(1).GetName(), MetricTest::metric_dimensions.at(1).GetName());
        ASSERT_EQ(test_metric.GetDimensions().at(1).GetValue(), MetricTest::metric_dimensions.at(1).GetValue());
        ASSERT_TRUE(test_metric.GetValues().empty());
    }

    TEST_F(MetricTest, TestCounterMetric) {
        Metric test_metric(MetricTest::metric_name, "ms", MetricType::COUNTER, MetricTest::metric_dimensions);
        ASSERT_TRUE(test_metric.GetValues().empty());

        test_metric.Update(1.0);
        ASSERT_EQ(test_metric.GetValues().size(), 1);
        ASSERT_EQ(test_metric.GetValues().at(0).GetValue(), 1.0);
        ASSERT_EQ(test_metric.GetValues().at(0).GetRequestId(), "");

        test_metric.Update(2.0, MetricTest::metric_request_id);
        ASSERT_EQ(test_metric.GetValues().size(), 2);
        ASSERT_EQ(test_metric.GetValues().at(0).GetValue(), 1.0);
        ASSERT_EQ(test_metric.GetValues().at(0).GetRequestId(), "");
        ASSERT_EQ(test_metric.GetValues().at(1).GetValue(), 2.0);
        ASSERT_EQ(test_metric.GetValues().at(1).GetRequestId(), MetricTest::metric_request_id);

        test_metric.Update(-1.0);
        ASSERT_EQ(test_metric.GetValues().size(), 2);
        ASSERT_EQ(test_metric.GetValues().at(0).GetValue(), 1.0);
        ASSERT_EQ(test_metric.GetValues().at(0).GetRequestId(), "");
        ASSERT_EQ(test_metric.GetValues().at(1).GetValue(), 2.0);
        ASSERT_EQ(test_metric.GetValues().at(1).GetRequestId(), MetricTest::metric_request_id);

        test_metric.Reset();
        ASSERT_TRUE(test_metric.GetValues().empty());
    }

    TEST_F(MetricTest, TestGaugeMetric) {
        Metric test_metric(MetricTest::metric_name, "count", MetricType::GAUGE, MetricTest::metric_dimensions);
        ASSERT_TRUE(test_metric.GetValues().empty());

        test_metric.Update(1.0);
        ASSERT_EQ(test_metric.GetValues().size(), 1);
        ASSERT_EQ(test_metric.GetValues().at(0).GetValue(), 1.0);
        ASSERT_EQ(test_metric.GetValues().at(0).GetRequestId(), "");

        test_metric.Update(-2.0, MetricTest::metric_request_id);
        ASSERT_EQ(test_metric.GetValues().size(), 2);
        ASSERT_EQ(test_metric.GetValues().at(0).GetValue(), 1.0);
        ASSERT_EQ(test_metric.GetValues().at(0).GetRequestId(), "");
        ASSERT_EQ(test_metric.GetValues().at(1).GetValue(), -2.0);
        ASSERT_EQ(test_metric.GetValues().at(1).GetRequestId(), MetricTest::metric_request_id);

        test_metric.Reset();
        ASSERT_TRUE(test_metric.GetValues().empty());
    }

    TEST_F(MetricTest, TestHistogramMetric) {
        Metric test_metric(MetricTest::metric_name, "count", MetricType::HISTOGRAM, MetricTest::metric_dimensions);
        ASSERT_TRUE(test_metric.GetValues().empty());

        test_metric.Update(1.0);
        ASSERT_EQ(test_metric.GetValues().size(), 1);
        ASSERT_EQ(test_metric.GetValues().at(0).GetValue(), 1.0);
        ASSERT_EQ(test_metric.GetValues().at(0).GetRequestId(), "");

        test_metric.Update(-2.0, MetricTest::metric_request_id);
        ASSERT_EQ(test_metric.GetValues().size(), 2);
        ASSERT_EQ(test_metric.GetValues().at(0).GetValue(), 1.0);
        ASSERT_EQ(test_metric.GetValues().at(0).GetRequestId(), "");
        ASSERT_EQ(test_metric.GetValues().at(1).GetValue(), -2.0);
        ASSERT_EQ(test_metric.GetValues().at(1).GetRequestId(), MetricTest::metric_request_id);

        test_metric.Reset();
        ASSERT_TRUE(test_metric.GetValues().empty());
    }
} // namespace torchserve
