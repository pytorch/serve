#ifndef TS_CPP_UTILS_LOGGING_HH_
#define TS_CPP_UTILS_LOGGING_HH_

#include <folly/init/Init.h>
#include <folly/logging/FileHandlerFactory.h>
#include <folly/logging/Init.h>
#include <folly/logging/LogConfigParser.h>
#include <folly/logging/LoggerDB.h>
#include <folly/logging/StreamHandlerFactory.h>
#include <folly/logging/xlog.h>

#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define TS_LOG(level, ...)                                                 \
  TS_LOG_IMPL(::torchserve::LogLevel::level, __FILE__, __LINE__, __func__, \
              ##__VA_ARGS__)

#define TS_LOGF(level, fmt, arg1, ...)                                      \
  TS_LOGF_IMPL(::torchserve::LogLevel::level, __FILE__, __LINE__, __func__, \
               fmt, arg1, ##__VA_ARGS__)

#define TS_LOG_IMPL(level, filename, line, function_name, ...) \
  ::torchserve::Logger::Log(level, filename, line, function_name, ##__VA_ARGS__)

#define TS_LOGF_IMPL(level, filename, line, function_name, fmt, arg1, ...)   \
  ::torchserve::Logger::Log(level, filename, line, function_name, fmt, arg1, \
                            ##__VA_ARGS__)

namespace torchserve {
enum LogLevel { TRACE = 0, DEBUG, INFO, WARN, ERROR, FATAL };

// TorchServe Logger API
class Logger {
 public:
  Logger();
  virtual ~Logger() = 0;
  static void InitDefaultLogger();
  static void InitLogger(const std::string &logger_config_path);
  // TODO: Add Initializer for file based config parsing
  // static void InitLogger(const std::filesystem::path& logconfig_file_path);

  template <typename... Args>
  static void Log(LogLevel level, const char *filename, unsigned int linenumber,
                  const char *function_name, Args &&... args) {
    FB_LOG_RAW(Logger::GetInstance(), ConvertTSLogLevelToFollyLogLevel(level),
               filename, linenumber, function_name,
               std::forward<Args>(args)...);
  }

  template <typename Arg, typename... Args>
  static void Log(LogLevel level, const char *filename, unsigned int linenumber,
                  const char *function_name, const char *fmt, Arg &&arg,
                  Args &&... args) {
    FB_LOGF_RAW(Logger::GetInstance(), ConvertTSLogLevelToFollyLogLevel(level),
                filename, linenumber, function_name, fmt,
                std::forward<Arg>(arg), std::forward<Args>(args)...);
  }

 private:
  static folly::LogLevel ConvertTSLogLevelToFollyLogLevel(LogLevel);
  static folly::Logger GetInstance() {
    static folly::Logger event_logger_(folly::Logger("torchserve"));
    return event_logger_;
  };
};
}  // namespace torchserve
#endif  // TS_CPP_UTILS_LOGGING_HH_