#pragma once
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <wanderary/utils/json_utils.h>

template <typename Base>
class ClassRegistry {
 public:
  using Json = wdr::utils::json;
  using CreatorFunc =
      std::function<std::unique_ptr<Base>(const wdr::utils::json& cfg)>;

  // 注册一个Class
  static void registerClass(const std::string& name, CreatorFunc creator) {
    registry()[name] = creator;
  }

  // 基于Class名创建一个实例
  static std::unique_ptr<Base> createInstance(const std::string& name,
                                              const wdr::utils::json& cfg) {
    Base::make_active();
    auto it = registry().find(name);
    return it != registry().end() ? it->second(cfg) : nullptr;
  }

  static std::set<std::string> RegisteredClassNames() {
    Base::make_active();
    std::set<std::string> names;
    for (auto& item : registry()) names.insert(item.first);
    return names;
  }

 private:
  static std::map<std::string, CreatorFunc>& registry() {
    static std::map<std::string, CreatorFunc> instance;
    return instance;
  }
};

// 注册Class，Class定义之后，再调用这个宏即可
#define REGISTER_DERIVED_CLASS(Base, Derived)         \
  class Derived##Register##From##Base {               \
   public:                                            \
    Derived##Register##From##Base() {                 \
      ClassRegistry<Base>::registerClass(             \
          #Derived, [](const wdr::utils::json& cfg) { \
            return std::make_unique<Derived>(cfg);    \
          });                                         \
    }                                                 \
  };                                                  \
  static Derived##Register##From##Base Derived##From##Base_register;
