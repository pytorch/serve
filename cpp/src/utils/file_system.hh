#ifndef TS_CPP_UTILS_FILE_SYSTEM_HH_
#define TS_CPP_UTILS_FILE_SYSTEM_HH_

#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

namespace torchserve {
class FileSystem {
 public:
  static std::unique_ptr<std::istream> GetStream(const std::string& path);
  static std::string LoadBytesFromFile(const std::string& path);
};
}  // namespace torchserve
#endif  // TS_CPP_UTILS_FILE_SYSTEM_HH_
