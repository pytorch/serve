#ifndef TS_CPP_UTILS_LOGGING_HH_
#define TS_CPP_UTILS_LOGGING_HH_

#include <memory>
#include <string>
#include <vector>

namespace torchserve {
  struct LoggerConfig {
    std::string logger_name;
    std::vector<std::string> sink_paths;
    std::string log_prefix_format;
    bool async;
  };

  // TorchServe Logger API
  class Logger {
    public:
    Logger();
    virtual ~Logger() = 0;

    virtual void Trace(const std::string& message) = 0;
    virtual void Debug(const std::string& message) = 0;
    virtual void Info(const std::string& message) = 0;
    virtual void Warn(const std::string& message) = 0;
    virtual void Error(const std::string& message) = 0;
    virtual void Fatal(const std::string& message) = 0;
  };

  // TorchServe LoggerFactory API
  class ILoggerFactory {
    public:
    virtual std::shared_ptr<torchserve::Logger> GetLogger(const std::string& logger_name) = 0;
  };
} // namespace torchserve
#endif // TS_CPP_UTILS_LOGGING_HH_