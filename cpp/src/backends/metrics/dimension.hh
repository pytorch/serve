#ifndef TS_CPP_BACKENDS_METRICS_DIMENSION_HH_
#define TS_CPP_BACKENDS_METRICS_DIMENSION_HH_

#include <string>

namespace torchserve {
    class Dimension {
        public:
        Dimension(const std::string& name, const std::string& value);
        std::string ToString() const;

        private:
        const std::string name;
        const std::string value;
    };
} // namespace torchserve

#endif // TS_CPP_BACKENDS_METRICS_DIMENSION_HH_
