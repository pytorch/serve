#include <gtest/gtest.h>

#include "src/utils/json.hh"

namespace torchserve {

TEST(JsonTest, TestParsingAndGetValue) {
  std::string json_file = "test/resources/test.json";
  auto data = Json::ParseJsonFile(json_file);

  EXPECT_TRUE(data.GetValueAsString("string").compare("test") == 0);

  EXPECT_TRUE(data.GetValueAsInt("int") == 42);

  auto data2 = data.GetValue("json");

  EXPECT_TRUE(data2.GetValueAsString("string").compare("test2") == 0);

  EXPECT_TRUE(data.HasKey("array"));

  EXPECT_TRUE(data.GetValue("array").GetValueAsString(0).compare("element1") == 0);
  EXPECT_TRUE(data.GetValue("array").GetValueAsString(1).compare("element2") == 0);
}

}  // namespace torchserve
