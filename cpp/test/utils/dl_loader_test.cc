#include "src/utils/dl_loader.hh"

#include <gtest/gtest.h>

#include <memory>

#include "src/utils/ifoo.hh"

using namespace std;

namespace torchserve {
class DLLoaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
#ifdef __APPLE__
    lib_path_ = "../src/utils/libfoo.dylib";
#elif __linux__
    lib_path_ = "../src/utils/libfoo.so";
#endif

    dl_loader_ = new torchserve::DLLoader<IFoo>(lib_path_);

    dl_loader_->OpenDL();
  }

  void TearDown() override { delete dl_loader_; }

  std::string lib_path_;
  torchserve::DLLoader<IFoo>* dl_loader_ = nullptr;
};

TEST_F(DLLoaderTest, TestGetInstance) {
  std::shared_ptr<IFoo> foo = dl_loader_->GetInstance();
  EXPECT_TRUE(foo != nullptr);
  int result = foo->add(1, 2);
  ASSERT_EQ(result, 3);
}
}  // namespace torchserve
