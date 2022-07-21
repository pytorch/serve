#include <glog/logging.h>
#include "src/utils/dl_loader.hh"

namespace torchserve {
  template <class T>
  void DLLoader<T>::OpenDL() {
    if (!(handle_ = dlopen(lib_path_.c_str(), RTLD_NOW | RTLD_LAZY))) {
				LOG(ERROR) << dlerror();
			}
  }

  template <class T>
  void DLLoader<T>::CloseDL() {
    if (dlclose(handle_) != 0) {
      LOG(ERROR) << dlerror();
    }
  }

  template <typename T>
  std::shared_ptr<T>	DLLoader<T>::GetInstance() {
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

    return std::shared_ptr(
      create_func(),
      [delete_func](T* p) {
        delete_func(p);
      });
  }
} // namespace torchserve