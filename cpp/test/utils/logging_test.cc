#include <gtest/gtest.h>

#if __has_include(<filesystem>)
#include <filesystem>
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
#else
#error require filesystem
#endif

#include <fmt/format.h>
#include <fstream>

#include "src/utils/logging.hh"

#if __cpp_lib_filesystem >= 201703
namespace fs = std::filesystem;
#elif __cpp_lib_experimental_filesystem >= 201406
namespace fs = std::experimental::filesystem;
#else
#error require filesystem
#endif

namespace torchserve {
  void Cleanup(const std::string& logfile_path) {
    if (!fs::remove(logfile_path)) {
      std::cout << "Failed to delete test log file" << logfile_path << std::endl;
    };
    torchserve::Logger::InitDefaultLogger();
  }

  TEST(LoggingTest, TestIncorrectLogInitialization) {
    std::string logger_config_path_str="test/resources/logging/invalid.config";
    EXPECT_THROW(torchserve::Logger::InitLogger(logger_config_path_str), std::invalid_argument);
  }

  TEST(LoggingTest, TestJSONConfigLogInitialization) {
    std::string logger_config_path_str="test/resources/logging/log_json.config";
    EXPECT_NO_THROW(torchserve::Logger::InitLogger(logger_config_path_str));
  }

  TEST(LoggingTest, TestFileLogInitialization) {
    std::string logfile_path = "test/resources/logging/test.log";
    std::string logger_config_path_str="test/resources/logging/log_to_file.config";
    torchserve::Logger::InitLogger(logger_config_path_str);
    std::string log_line("Test");
    TS_LOG(INFO, log_line);
    EXPECT_TRUE(fs::exists(logfile_path));

    std::string contents;
    std::ifstream logfile(logfile_path);
    if (logfile.is_open()) {
      std::getline(logfile, contents);
      logfile.close();
    }
    EXPECT_TRUE(contents.compare(contents.size() - log_line.size(), log_line.size(), log_line) == 0);

    // Cleanup
    Cleanup(logfile_path);
  }
}
