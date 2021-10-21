#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace torchserve {
  using ResponseBatch = std::vector<std::shared_ptr<Request>>;

struct Response {
  std::string requsetId;
  int16_t statusCode;
  std::string statusPhase;
  std::unordered_map<std::string, std::string> headers;
  std::vector<std::byte> value;
};
}  // namespace torchserve