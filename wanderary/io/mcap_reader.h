#pragma once
#include <map>
#include <set>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <mcap/reader.hpp>

namespace wdr::io {

class MCAPReader {
 public:
  explicit MCAPReader(const std::string &filename);
  ~MCAPReader();
  void close();

  int size(const std::string &topic) const;

  int64_t ReadImage(const std::string &topic, int index, cv::Mat *data,
                    bool decode);

 private:
  bool is_open_{false};
  mcap::McapReader reader_;
  std::map<std::string, std::vector<mcap::Timestamp>> channel_times_;
};

}  // namespace wdr::io
