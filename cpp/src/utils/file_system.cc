#include "src/utils/file_system.hh"
#include "src/utils/logging.hh"

#include <folly/FileUtil.h>
#include <folly/json.h>

namespace torchserve {
std::unique_ptr<std::istream> FileSystem::GetStream(
    const std::string& file_path) {
  auto file_stream = std::make_unique<std::ifstream>(file_path);
  if (!file_stream) {
    throw std::invalid_argument(
        fmt::format("Invalid file path: {}", file_path));
  }
  return file_stream;
}

std::string FileSystem::LoadBytesFromFile(const std::string& path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    TS_LOGF(ERROR, "Cannot open tokenizer file {}", path);
    throw;
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

std::unique_ptr<folly::dynamic> FileSystem::LoadJsonFile(const std::string& file_path) {
  std::string content;
  if (!folly::readFile(file_path.c_str(), content)) {
    TS_LOGF(ERROR, "{} not found", file_path);
    throw;
  }
  return std::make_unique<folly::dynamic>(folly::parseJson(content));
}

const folly::dynamic& FileSystem::GetJsonValue(std::unique_ptr<folly::dynamic>& json, const std::string& key) {
  if (json->find(key) != json->items().end()) {
    return (*json)[key];
  } else {
    TS_LOG(ERROR, "Required field {} not found in JSON.", key);
    throw ;
  }
}
}  // namespace torchserve
