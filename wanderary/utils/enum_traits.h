#pragma once

#include <map>
#include <string>
#include <utility>

#include <glog/logging.h>

#include <boost/preprocessor.hpp>

// 单个枚举类型的定义, 比如A = 1
#define ENUM_NUMBERED_SEQ_X(s, data, elem) \
  BOOST_PP_TUPLE_ELEM(0, elem) = BOOST_PP_TUPLE_ELEM(1, elem)

// ENUM到string的映射, 比如A = "A"
#define ENUM_TO_STRING_SEQ_X(s, name, elem)          \
  std::make_pair(name::BOOST_PP_TUPLE_ELEM(0, elem), \
                 BOOST_PP_TUPLE_ELEM(2, elem))

// string到ENUM的映射, 比如"A" = A
#define STRING_TO_ENUM_SEQ_X(s, name, elem)    \
  std::make_pair(BOOST_PP_TUPLE_ELEM(2, elem), \
                 name::BOOST_PP_TUPLE_ELEM(0, elem))

#define INT_TO_ENUM_SEQ_X(s, name, elem)       \
  std::make_pair(BOOST_PP_TUPLE_ELEM(1, elem), \
                 name::BOOST_PP_TUPLE_ELEM(0, elem))

#define VAR_NAME_TYPE2STRING(name) map_##name##2str
#define VAR_NAME_STRING2TYPE(name) map_str2##name
#define VAR_NAME_INT2TYPE(name) map_int2##name
#define FUN_NAME_TYPE2STRING(name) BOOST_PP_CAT(name, 2str)
#define FUN_NAME_STRING2TYPE(name) BOOST_PP_CAT(str2, name)
#define FUN_NAME_INT2TYPE(name) BOOST_PP_CAT(int2, name)

#define ADD_SWITCH_TEST_UNIQUE(r, name, elem) \
  case name::BOOST_PP_TUPLE_ELEM(0, elem):    \
    break;

#define ENUM_NUMBERED_REGISTER(name, enumerators)                             \
  enum class name {                                                           \
    BOOST_PP_SEQ_ENUM(                                                        \
        BOOST_PP_SEQ_TRANSFORM(ENUM_NUMBERED_SEQ_X, ~, enumerators))          \
  };                                                                          \
  inline static const std::map<name, std::string> VAR_NAME_TYPE2STRING(       \
      name) = {BOOST_PP_SEQ_ENUM(BOOST_PP_SEQ_TRANSFORM(ENUM_TO_STRING_SEQ_X, \
                                                        name, enumerators))}; \
  inline static const std::map<std::string, name> VAR_NAME_STRING2TYPE(       \
      name) = {BOOST_PP_SEQ_ENUM(BOOST_PP_SEQ_TRANSFORM(STRING_TO_ENUM_SEQ_X, \
                                                        name, enumerators))}; \
  inline static const std::map<int, name> VAR_NAME_INT2TYPE(name) = {         \
      BOOST_PP_SEQ_ENUM(                                                      \
          BOOST_PP_SEQ_TRANSFORM(INT_TO_ENUM_SEQ_X, name, enumerators))};     \
  namespace {                                                                 \
  void UniqueEnumCheck##name() {                                              \
    switch (name()) {                                                         \
      BOOST_PP_SEQ_FOR_EACH(ADD_SWITCH_TEST_UNIQUE, name, enumerators)        \
    }                                                                         \
  }                                                                           \
  }

#define ENUM_CONVERSION_REGISTER(name, UNKNOWN_TYPE, UNKNOWN_STR)             \
  inline name FUN_NAME_STRING2TYPE(name)(const std::string &str) {            \
    if (VAR_NAME_STRING2TYPE(name).find(str) !=                               \
        VAR_NAME_STRING2TYPE(name).end()) {                                   \
      return VAR_NAME_STRING2TYPE(name).at(str);                              \
    } else {                                                                  \
      LOG(FATAL) << "[str2enum] Unknown " << #name << " type: " << str;       \
    }                                                                         \
    return UNKNOWN_TYPE;                                                      \
  }                                                                           \
  inline std::string FUN_NAME_TYPE2STRING(name)(name type) {                  \
    if (VAR_NAME_TYPE2STRING(name).find(type) !=                              \
        VAR_NAME_TYPE2STRING(name).end()) {                                   \
      return VAR_NAME_TYPE2STRING(name).at(type);                             \
    } else {                                                                  \
      LOG(FATAL) << "[enum2str] Unknown " << #name                            \
                 << " type: " << static_cast<int>(type);                      \
    }                                                                         \
    return UNKNOWN_STR;                                                       \
  }                                                                           \
  inline name FUN_NAME_INT2TYPE(name)(int num) {                              \
    if (VAR_NAME_INT2TYPE(name).find(num) != VAR_NAME_INT2TYPE(name).end()) { \
      return VAR_NAME_INT2TYPE(name).at(num);                                 \
    } else {                                                                  \
      LOG(FATAL) << "[int2enum] Unknown " << #name << " type: " << num;       \
    }                                                                         \
    return UNKNOWN_TYPE;                                                      \
  }
