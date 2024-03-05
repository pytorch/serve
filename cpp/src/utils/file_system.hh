#ifndef TS_CPP_UTILS_FILE_SYSTEM_HH_
#define TS_CPP_UTILS_FILE_SYSTEM_HH_

#include <folly/dynamic.h>
#include <fstream>
#include <stdexcept>
#include <string>

namespace torchserve {
class FileSystem {
 public:
  static std::unique_ptr<std::istream> GetStream(const std::string& path);
  static std::string LoadBytesFromFile(const std::string& path);
  static std::unique_ptr<folly::dynamic> LoadJsonFile(const std::string& file_path);
  static const folly::dynamic& GetJsonValue(std::unique_ptr<folly::dynamic>& json, const std::string& key);
};
}  // namespace torchserve
#endif  // TS_CPP_UTILS_FILE_SYSTEM_HH_
