#include "src/utils/metrics/units.hh"

namespace torchserve {
const std::unordered_map<std::string, std::string> Units::unit_mapping{
    {"ms", "Milliseconds"}, {"s", "Seconds"},    {"percent", "Percent"},
    {"count", "Count"},     {"MB", "Megabytes"}, {"GB", "Gigabytes"},
    {"kB", "Kilobytes"},    {"B", "Bytes"},      {"", "unit"}};

const std::string Units::GetUnitMapping(const std::string& unit) {
  auto mapping = Units::unit_mapping.find(unit);
  return mapping == Units::unit_mapping.end() ? unit : mapping->second;
}
}  // namespace torchserve
