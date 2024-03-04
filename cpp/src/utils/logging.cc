#include "logging.hh"

#include <folly/init/Init.h>
#include <folly/logging/FileHandlerFactory.h>
#include <folly/logging/Init.h>
#include <folly/logging/LogConfigParser.h>
#include <folly/logging/LoggerDB.h>
#include <folly/logging/StreamHandlerFactory.h>


namespace torchserve {
folly::LogLevel ConvertTSLogLevelToFollyLogLevel(LogLevel log_level) {
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

void Logger::InitDefaultLogger() {
  std::string logger_config_path("./_build/resources/logging.config");
  InitLogger(logger_config_path);
}

void Logger::InitLogger(const std::string& logger_config_path) {
  folly::LogConfig config;
  try {
    std::ifstream log_config_file(logger_config_path);
    std::stringstream buffer;
    buffer << log_config_file.rdbuf();
    std::string logger_config(buffer.str());
    config = folly::parseLogConfig(logger_config);
    folly::LoggerDB::get().registerHandlerFactory(
        std::make_unique<folly::FileHandlerFactory>());
  } catch (const std::range_error& e) {
    // Do nothing if a FileHandler has already been registered
    if (std::strcmp(
            e.what(),
            "a LogHandlerFactory for the type \"file\" already exists") != 0) {
      throw e;
    }
  } catch (const folly::LogConfigParseError& e) {
    throw std::invalid_argument("Failed to parse logging config: " +
                                logger_config_path);
  }
  folly::LoggerDB::get().resetConfig(config);
}
}  // namespace torchserve
