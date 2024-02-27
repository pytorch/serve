#ifndef TS_CPP_UTILS_FILE_SYSTEM_HH_
#define TS_CPP_UTILS_FILE_SYSTEM_HH_

#include <fmt/format.h>

#include <fstream>
#include <stdexcept>
#include <string>

namespace torchserve {
class FileSystem {
 public:
  static std::unique_ptr<std::istream> GetStream(const std::string& path);
};
}  // namespace torchserve
#endif  // TS_CPP_UTILS_FILE_SYSTEM_HH_
