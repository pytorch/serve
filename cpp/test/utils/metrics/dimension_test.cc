#include <string>
#include <gtest/gtest.h>

#include "src/utils/metrics/dimension.hh"

namespace torchserve {
    class DimensionTest : public ::testing::Test {
        protected:
        static const std::string dimension_name;
        static const std::string dimension_value;
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

    TEST_F(DimensionTest, TestGetName) {
        ASSERT_EQ(dimension->GetName(), DimensionTest::dimension_name);
    }

    TEST_F(DimensionTest, TestGetValue) {
        ASSERT_EQ(dimension->GetValue(), DimensionTest::dimension_value);
    }
} // namespace torchserve
