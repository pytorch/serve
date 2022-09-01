#include <gtest/gtest.h>
#include <experimental/filesystem>
#include <fmt/format.h>
#include <fstream>

#include "src/utils/logging.hh"

namespace fs = std::experimental::filesystem;

namespace torchserve {
  void Cleanup(const std::string& logfile_path) {
    if (!fs::remove(logfile_path)) {
      std::cout << "Failed to delete test log file" << logfile_path << std::endl;
    };
    torchserve::Logger::InitDefaultLogger();
  }

  TEST(LoggingTest, TestIncorrectLogInitialization) {
    std::string logger_config="INVALID_CONFIG";
    EXPECT_THROW(torchserve::Logger::InitLogger(logger_config), std::invalid_argument);
  }

  TEST(LoggingTest, TestFileLogInitialization) {
    std::string logfile_path = fmt::format("{}test.log", fs::temp_directory_path().c_str());
    std::string logger_config = fmt::format("INFO:default; default=file:path={},async=false", logfile_path);
    torchserve::Logger::InitLogger(logger_config);
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