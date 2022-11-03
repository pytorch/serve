#ifndef TS_CPP_UTILS_METRICS_UNITS_HH_
#define TS_CPP_UTILS_METRICS_UNITS_HH_

#include <string>
#include <unordered_map>

namespace torchserve {
class Units {
 public:
  static const std::string GetUnitMapping(const std::string& unit);

 private:
  static const std::unordered_map<std::string, std::string> unit_mapping;
};
}  // namespace torchserve

#endif  // TS_CPP_UTILS_METRICS_UNITS_HH_
