#include "logging.hh"

namespace torchserve {
  void Logger::InitDefaultLogger() {
    auto config = folly::parseLogConfig("INFO:consoleLogger;consoleLogger=stream:stream=stdout,async=true");
    folly::LoggerDB::get().resetConfig(config);
  }
  void Logger::InitLogger(const std::string& logger_config) {
    auto config = folly::parseLogConfig(logger_config);
    folly::LoggerDB::get().registerHandlerFactory(std::make_unique<folly::FileHandlerFactory>());
    folly::LoggerDB::get().resetConfig(config);
  }

  folly::LogLevel Logger::ConvertTSLogLevelToFollyLogLevel(LogLevel log_level) {
    switch (log_level) {
      case LogLevel::TRACE:
      case LogLevel::DEBUG:
        return folly::LogLevel::DBG;
      case LogLevel::INFO:
        return folly::LogLevel::INFO;
      case LogLevel::WARN:
        return folly::LogLevel::WARN;
      case LogLevel::ERROR:
        return folly::LogLevel::ERR;
      case LogLevel::FATAL:
        return folly::LogLevel::FATAL;
      default:
        return folly::LogLevel::INFO;
    }
  }
} // namespace torchserve