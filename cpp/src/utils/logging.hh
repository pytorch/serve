#pragma once

#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include "spdlog/spdlog.h"

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

namespace torchserve
{
  enum LogLevel
  {
    TRACE = 0,
    DEBUG,
    INFO,
    WARN,
    ERROR,
    FATAL
  };

  spdlog::level::level_enum ConvertTsToSpdLogLevel(LogLevel log_level);

  // TorchServe Logger API
  class Logger
  {
  public:
    Logger();
    virtual ~Logger() = 0;
    static void InitDefaultLogger();
    static void InitLogger(const std::string &logger_config_path);
    // TODO: Add Initializer for file based config parsing
    // static void InitLogger(const std::filesystem::path& logconfig_file_path);

    template <typename... Args>
    static void Log(LogLevel level, const char *filename, unsigned int linenumber,
                            const char *function_name, Args &&...args)
    {
      if(!Logger::logger)
        return;

      std::string msg = fmt::format("{}:{}] {}", filename, linenumber, function_name);
      Logger::logger->log(ConvertTsToSpdLogLevel(level),
        msg + " {}", std::forward<Args>(args)...);
    }

    template <typename Arg, typename... Args>
    static void Log(LogLevel level, const char *filename, unsigned int linenumber,
                            const char *function_name, const char *fmt, Arg &&arg,
                            Args &&...args)
    {
      if(!Logger::logger)
        return;

      std::string msg = fmt::format("{}:{}] {} {}", filename, linenumber, function_name, fmt);
      Logger::logger->log(ConvertTsToSpdLogLevel(level),
        msg, std::forward<Arg>(arg), std::forward<Args>(args)...);
    }
  protected:
    static std::shared_ptr<spdlog::logger> logger;
  };

} // namespace torchserve
