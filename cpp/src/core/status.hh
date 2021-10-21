#pragma once

#include <string>

namespace torchserve {
class Status {
 public:
  int code;
  std::string msg;
};
}  // namespace torchserve