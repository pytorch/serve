#include "src/utils/metrics/registry.hh"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdexcept>

namespace torchserve {
TEST(RegistryTest, TestValidConfigFile) {
  MetricsRegistry::Initialize("resources/metrics/valid_config.yaml",
                              MetricsContext::BACKEND);
  ASSERT_THAT(MetricsRegistry::GetMetricsCacheInstance(),
              ::testing::A<std::shared_ptr<MetricsCache>>());
}

TEST(RegistryTest, TestInvalidConfigFile) {
  ASSERT_THROW(
      MetricsRegistry::Initialize(
          "resources/metrics/invalid_config_duplicate_dimension.yaml",
          MetricsContext::BACKEND),
      std::invalid_argument);
  ASSERT_THROW(MetricsRegistry::GetMetricsCacheInstance(), std::runtime_error);
}

TEST(RegistryTest, TestReInitialize) {
  MetricsRegistry::Initialize("resources/metrics/valid_config.yaml",
                              MetricsContext::BACKEND);
  ASSERT_THAT(MetricsRegistry::GetMetricsCacheInstance(),
              ::testing::A<std::shared_ptr<MetricsCache>>());

  MetricsRegistry::Initialize("resources/metrics/default_config.yaml",
                              MetricsContext::BACKEND);
  ASSERT_THAT(MetricsRegistry::GetMetricsCacheInstance(),
              ::testing::A<std::shared_ptr<MetricsCache>>());
}
}  // namespace torchserve
