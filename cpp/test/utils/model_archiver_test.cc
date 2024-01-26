#include <gtest/gtest.h>

#include "src/utils/model_archive.hh"

namespace torchserve {
TEST(ManifestTest, TestInitialize) {
  torchserve::Manifest manifest;
  manifest.Initialize(
      "test/resources/examples/mnist/base_handler/MAR-INF/"
      "MANIFEST.json");
  ASSERT_EQ(manifest.GetCreatOn(), "28/07/2020 06:32:08");
  ASSERT_EQ(manifest.GetArchiverVersion(), "0.2.0");
  ASSERT_EQ(manifest.GetRuntimeType(), "LSP");
  ASSERT_EQ(manifest.GetModel().model_name, "mnist_scripted_v2");
  ASSERT_EQ(manifest.GetModel().serialized_file, "mnist_script.pt");
  ASSERT_EQ(manifest.GetModel().handler, "TorchScriptHandler");
  ASSERT_EQ(manifest.GetModel().model_version, "2.0");
}
}  // namespace torchserve
