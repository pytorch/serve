#include "src/utils/file_system.hh"

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
}  // namespace torchserve