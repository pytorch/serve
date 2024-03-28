#include <gtest/gtest.h>

#include "src/utils/json.hh"

namespace torchserve {

TEST(JsonTest, TestParsingAndGetValue) {
  std::string json_file = "resources/test.json";
  auto data = Json::ParseJsonFile(json_file);

  EXPECT_TRUE(data.GetValue("string").AsString().compare("test") == 0);

  EXPECT_TRUE(data.GetValue("int").AsInt() == 42);

  auto data2 = data.GetValue("json");

  EXPECT_TRUE(data2.GetValue("string").AsString().compare("test2") == 0);

  EXPECT_TRUE(data.HasKey("array"));

  EXPECT_TRUE(data.GetValue("array").GetValue(0).AsString().compare("element1") == 0);
  EXPECT_TRUE(data.GetValue("array").GetValue(1).AsString().compare("element2") == 0);
}

}  // namespace torchserve
