#ifndef TS_CPP_UTILS_DL_LOADER_HH_
#define TS_CPP_UTILS_DL_LOADER_HH_

#include <dlfcn.h>

#include <iostream>
#include <memory>
#include <string>

#include "src/utils/logging.hh"

namespace torchserve {
/**
 * @brief DLLoader is the implementation of loading shared library for Unix.
 *
 * @tparam T
 */
template <class T>
class DLLoader {
 public:
  DLLoader(const std::string& lib_path,
           const std::string& create_object_func_name = "allocator",
           const std::string& delete_object_func_name = "deleter")
      : lib_path_(lib_path),
        create_object_func_name_(create_object_func_name),
        delete_object_func_name_(delete_object_func_name),
        handle_(nullptr){};
  ~DLLoader() = default;

  void OpenDL() {
    handle_ = dlopen(lib_path_.c_str(), RTLD_NOW | RTLD_LAZY);
    if (!handle_) {
      TS_LOGF(ERROR, "Failed to open lib: {}, error: {}", lib_path_, dlerror());
    }
  };

  void CloseDL() {
    std::cerr << "CloseDL start"
              << "\n";
    if (handle_ != nullptr && dlclose(handle_) != 0) {
      TS_LOGF(ERROR, "Failed to close lib: {}, error: {}", lib_path_,
              dlerror());
    }
    std::cerr << "CloseDL done"
              << "\n";
  };

  std::shared_ptr<T> GetInstance() {
    if (handle_ == nullptr) {
      return std::shared_ptr<T>(nullptr);
    }

    using createClass = T* (*)();
    using deleteClass = void (*)(T*);
    char* error = nullptr;

    createClass create_func = reinterpret_cast<createClass>(
        dlsym(handle_, create_object_func_name_.c_str()));
    error = dlerror();
    if (error != nullptr) {
      TS_LOGF(ERROR, "create_func, error: {}", error);
      CloseDL();
      return std::shared_ptr<T>(nullptr);
    }

    deleteClass delete_func = reinterpret_cast<deleteClass>(
        dlsym(handle_, delete_object_func_name_.c_str()));
    error = dlerror();
    if (error != nullptr) {
      TS_LOGF(ERROR, "delete_func, error: {}", error);
      CloseDL();
      return std::shared_ptr<T>(nullptr);
    }

    return std::shared_ptr<T>(create_func(),
                              [delete_func](T* p) { delete_func(p); });
  }

 private:
  std::string lib_path_;
  std::string create_object_func_name_;
  std::string delete_object_func_name_;
  void* handle_;
};
}  // namespace torchserve
#endif  // TS_CPP_UTILS_DL_LOADER_HH_
