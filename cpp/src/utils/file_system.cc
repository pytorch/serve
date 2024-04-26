#include "src/utils/file_system.hh"
#include "src/utils/logging.hh"

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
}  // namespace torchserve
