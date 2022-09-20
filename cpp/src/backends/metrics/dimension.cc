#include "src/backends/metrics/dimension.hh"

namespace torchserve {
    Dimension::Dimension(const std::string& name, const std::string& value) :
        name{name}, value{value} {}

    std::string Dimension::ToString() const {
        return name + ":" + value;
    }

} // namespace torchserve
