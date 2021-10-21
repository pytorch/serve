#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace torchserve {
  using RequestBatch = std::vector<Request>;

struct Request {
  struct InputParameter {
    std::string name;
    std::vector<std::byte> value;
  };

  std::string requsetId;
  std::unordered_map<std::string, std::string> headers;
  std::vector<InputParameter> inputParameters;
};
}  // namespace torchserve