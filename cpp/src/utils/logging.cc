#include "logging.hh"

#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <yaml-cpp/yaml.h>


namespace torchserve {
spdlog::level::level_enum ConvertTsToSpdLogLevel(LogLevel log_level) {
  switch (log_level) {
    case LogLevel::TRACE:
    case LogLevel::DEBUG:
      return spdlog::level::debug;
    case LogLevel::WARN:
      return spdlog::level::warn;
    case LogLevel::ERROR:
      return spdlog::level::err;
    case LogLevel::FATAL:
      return spdlog::level::critical;
    default:
      return spdlog::level::info;
  }
}

LogLevel ConvertStringToLogLevel(std::string loglevel) {
  if(loglevel == "DEBUG")
    return LogLevel::DEBUG;
  else if(loglevel == "WARN")
    return LogLevel::WARN;
  else if(loglevel == "TRACE")
    return LogLevel::TRACE;
  else if(loglevel == "FATAL")
    return LogLevel::FATAL;
  else
    return LogLevel::INFO;
}

Logger::Logger(){}
Logger::~Logger(){}

void Logger::InitDefaultLogger() {
  InitLogger("");
}

void Logger::InitLogger(const std::string& logger_config_path) {
  std::string logfile = "";
  LogLevel loglevel = LogLevel::INFO;
  bool async = true;

  if(logger_config_path.length()) {
    try{
      YAML::Node config_node = YAML::LoadFile(logger_config_path);

      if(config_node["config"]){
        if(config_node["config"]["loglevel"])
          loglevel = ConvertStringToLogLevel(config_node["config"]["loglevel"].as<std::string>());
        if(config_node["config"]["logfile"])
          logfile = config_node["config"]["logfile"].as<std::string>();
        if(config_node["config"]["async"]){
          async = config_node["config"]["async"].as<std::string>() == "true";
        }
      }
    } catch (YAML::Exception& e) {
      std::string error_message =
          "Failed to load logging YAML configuration file: " +
          logger_config_path + ". " + e.what();
      throw std::invalid_argument(error_message);
    }
  }
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  console_sink->set_pattern("[%l] %m%d %H:%M:%S.%e %v");

  std::vector<spdlog::sink_ptr> sinks;
  sinks.push_back(console_sink);
  if(logfile.length()){
    if(async){
      auto rotating_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(logfile, 1024*1024*10, 3);
      rotating_sink->set_pattern("[%l] %m%d %H:%M:%S.%e %v");
      sinks.push_back(rotating_sink);
    }else{
      auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(logfile);
      file_sink->set_pattern("[%l] %m%d %H:%M:%S.%e %v");
      sinks.push_back(file_sink);
    }
  }
  Logger::logger = std::make_shared<spdlog::logger>("multi_sink", sinks.begin(), sinks.end());
  logger->set_level(ConvertTsToSpdLogLevel(loglevel));
}

std::shared_ptr<spdlog::logger> Logger::logger;
}  // namespace torchserve
