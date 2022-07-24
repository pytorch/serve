#ifndef TS_CPP_UTILS_DL_LOADER_HH_
#define TS_CPP_UTILS_DL_LOADER_HH_

#include <iostream>
#include <dlfcn.h>
#include <glog/logging.h>
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
      const std::string& create_object_func_name = "allocator",
      const std::string& delete_object_func_name = "deleter") :
      lib_path_(lib_path), 
      create_object_func_name_(create_object_func_name),
      delete_object_func_name_(delete_object_func_name),
      handle_(nullptr) {};
    ~DLLoader() = default;

    void OpenDL() {
      if (!(handle_ = dlopen(lib_path_.c_str(), RTLD_NOW | RTLD_LAZY))) {
				LOG(ERROR) << dlerror();
			}
    };

    void CloseDL() {
      if (dlclose(handle_) != 0) {
        LOG(ERROR) << dlerror();
      }
    };

    std::shared_ptr<T>	GetInstance() {
      using createClass = T *(*)();
	    using deleteClass = void (*)(T *);

      auto create_func = reinterpret_cast<createClass>(
        dlsym(handle_, create_object_func_name_.c_str()));
      
      auto delete_func = reinterpret_cast<deleteClass>(
        dlsym(handle_, delete_object_func_name_.c_str()));
      
      if (!create_func || !delete_func) {
        CloseDL();
        LOG(ERROR) << dlerror();
      }
      std::cout << "ready" << std::endl;
      
      return std::shared_ptr<T>(
        create_func(),
        [delete_func](T *p) {
          delete_func(p);
        });
    }

    private:
    std::string lib_path_;
    std::string create_object_func_name_;
    std::string delete_object_func_name_;
    void *handle_;
  };
} // namespace torchserve
#endif // TS_CPP_UTILS_DL_LOADER_HH_