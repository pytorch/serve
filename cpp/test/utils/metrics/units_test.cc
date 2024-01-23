#include "src/utils/metrics/units.hh"

#include <gtest/gtest.h>

#include <string>

namespace torchserve {
TEST(UnitsTest, TestGetExistingUnitMapping) {
  ASSERT_EQ(Units::GetUnitMapping("MB"), "Megabytes");
}

TEST(UnitsTest, TestGetNonExistentUnitMapping) {
  ASSERT_EQ(Units::GetUnitMapping("test_unit"), "test_unit");
}

TEST(UnitsTest, TestGetEmptyUnitMapping) {
  ASSERT_EQ(Units::GetUnitMapping(""), "unit");
}

}  // namespace torchserve
