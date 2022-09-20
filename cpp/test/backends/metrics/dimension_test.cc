#include <string>
#include <gtest/gtest.h>

#include "src/backends/metrics/dimension.hh"

namespace torchserve {
    class DimensionTest : public ::testing::Test {
        protected:
        static const std::string dimension_name;
        static const std::string dimension_value;
        static const std::string dimension_string;
        Dimension* dimension;

        void SetUp() override {
            dimension = new Dimension(dimension_name, dimension_value);
        }

        void TearDown() override {
            delete dimension;
        }
    };

    const std::string DimensionTest::dimension_name {"name"};
    const std::string DimensionTest::dimension_value {"value"};
    const std::string DimensionTest::dimension_string {"name:value"};

    TEST_F(DimensionTest, TestToString) {
        ASSERT_EQ(dimension->ToString(), DimensionTest::dimension_string);
    }
} // namespace torchserve
