#ifndef TS_CPP_BACKENDS_METRICS_DIMENSION_HH_
#define TS_CPP_BACKENDS_METRICS_DIMENSION_HH_

#include <string>

namespace torchserve {
    class Dimension {
        public:
        Dimension(const std::string& name, const std::string& value);
        const std::string GetName() const;
        const std::string GetValue() const;

        private:
        const std::string name;
        const std::string value;
    };
} // namespace torchserve

#endif // TS_CPP_BACKENDS_METRICS_DIMENSION_HH_
