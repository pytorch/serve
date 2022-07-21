#ifndef TS_CPP_UTILS_DL_LOADER_HH_
#define TS_CPP_UTILS_DL_LOADER_HH_

#include <iostream>
#include <dlfcn.h>
#include <memory>
#include <string>

namespace torchserve {
  /**
   * @brief DLLoader is the implementation of loading shared library for Unix.
   * 
   * @tparam T 
   */
  template  <class T>
  class DLLoader {
    public:
    DLLoader(
      const std::string& lib_path,
      const std::string& create_object_func_name,
      const std::string& delete_object_func_name) :
      handle_(nullptr),
      lib_path_(lib_path), 
      create_object_func_name_(create_object_func_name), 
      delete_object_func_name_(delete_object_func_name) {};
    ~DLLoader() = default;

    void OpenDL();
    void CloseDL();
    std::shared_ptr<T>	GetInstance();

    private:
    std::string lib_path_;
    std::string create_object_func_name_;
    std::string delete_object_func_name_;
    void* handle_;
  };
} // namespace torchserve
#endif // TS_CPP_UTILS_DL_LOADER_HH_