#include "src/utils/metrics/dimension.hh"

namespace torchserve {
    Dimension::Dimension(const std::string& name, const std::string& value) :
                         name{name}, value{value} {}

    const std::string Dimension::GetName() const {
        return name;
    }

    const std::string Dimension::GetValue() const {
        return value;
    }
} // namespace torchserve
