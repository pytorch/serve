#ifndef TS_CPP_UTILS_METRICS_EMITTER_HH_
#define TS_CPP_UTILS_METRICS_EMITTER_HH_

namespace torchserve {
    class Emitter {
        public:
        virtual void Emit() = 0;
    };
} // namespace torchserve

#endif // TS_CPP_UTILS_METRICS_EMITTER_HH_
