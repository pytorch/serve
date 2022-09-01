#include "logging.hh"

namespace torchserve {
  void Logger::InitDefaultLogger() {
    auto config = folly::parseLogConfig("INFO:consoleLogger;consoleLogger=stream:stream=stdout,async=true");
    folly::LoggerDB::get().resetConfig(config);
  }
  void Logger::InitLogger(const std::string& logger_config) {
    folly::LogConfig config;
    try {
      config = folly::parseLogConfig(logger_config);
      folly::LoggerDB::get().registerHandlerFactory(std::make_unique<folly::FileHandlerFactory>());
    } catch(const std::range_error& e) {
      // Do nothing if a FileHandler has already been registered
      if (std::strcmp(e.what(), "a LogHandlerFactory for the type \"file\" already exists") != 0) {
        throw e;
      }
    } catch(const folly::LogConfigParseError& e) {
      throw std::invalid_argument("Failed to parse logging config");
    }
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