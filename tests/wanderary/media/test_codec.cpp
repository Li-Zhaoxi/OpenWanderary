#include <glog/logging.h>
#include <gtest/gtest.h>

#include "wanderary/media/codec_jpg.h"
#include "wanderary/media/media_codec.h"

using CodecDescriptor = wdr::media::CodecDescriptor;
using CodecStartupParams = wdr::media::CodecStartupParams;
using MediaCodecJpg = wdr::media::MediaCodecJpg;

TEST(MediaCodec, GetCodecDescriptor) {
  const CodecDescriptor desc =
      wdr::media::GetCodecDescriptor(MediaCodecID::kMJPEG);
  EXPECT_TRUE(desc.has_value());

  if (!desc.has_value()) return;

  // mjpg id, 硬编码
  EXPECT_TRUE(desc.id() == MediaCodecID::kMJPEG)
      << "codec_id: " << MediaCodecID2str(desc.id());
  EXPECT_TRUE(desc.mode() == MediaCodecMode::kHARDWARE)
      << "codec_mode: " << MediaCodecMode2str(desc.mode());
  EXPECT_EQ(desc.name(), "mjpeg");
  EXPECT_EQ(desc.long_name(), "Motion JPEG");
  EXPECT_EQ(desc.mime_types(), "image/jpeg");
}

TEST(MediaCodec, GetDefaultContext) {
  const auto ctx = wdr::media::GetDefaultContext(MediaCodecID::kMJPEG, false);
  EXPECT_TRUE(ctx.id() == MediaCodecID::kMJPEG)
      << "codec_id: " << MediaCodecID2str(ctx.id());
  EXPECT_EQ(ctx.encoder(), false);
  EXPECT_EQ(ctx.instance_index(), 0);
}

TEST(MediaCodecJpg, InitRelease) {
  auto ctx = wdr::media::CodecContext::CreateJpgEncode(
      MediaCodecID::kMJPEG, 1920, 1080, CodecPixelFormat::kNV12);
  MediaCodecState state = MediaCodecState::kNone;

  EXPECT_EQ(ctx.width(), 1920);
  EXPECT_EQ(ctx.height(), 1080);

  // 初始化
  wdr::media::InitializeCodecContext(&ctx);
  state = wdr::media::GetCodecState(&ctx);
  EXPECT_TRUE(state == MediaCodecState::kInitialized)
      << "codec_state: " << MediaCodecState2str(state);

  // 配置
  wdr::media::CodecConfigure(&ctx);
  state = wdr::media::GetCodecState(&ctx);
  EXPECT_TRUE(state == MediaCodecState::kConfigured)
      << "codec_state: " << MediaCodecState2str(state);

  // 启动
  CodecStartupParams params(ctx.id(), ctx.encoder());
  wdr::media::CodecStart(&ctx, &params);
  state = wdr::media::GetCodecState(&ctx);
  EXPECT_TRUE(state == MediaCodecState::kStarted)
      << "codec_state: " << MediaCodecState2str(state);

  // 停止
  wdr::media::CodecStop(&ctx);
  state = wdr::media::GetCodecState(&ctx);
  EXPECT_TRUE(state == MediaCodecState::kInitialized)
      << "codec_state: " << MediaCodecState2str(state);

  // 释放
  wdr::media::ReleaseCodecContext(&ctx);
  state = wdr::media::GetCodecState(&ctx);
  EXPECT_TRUE(state == MediaCodecState::kUninitialized)
      << "codec_state: " << MediaCodecState2str(state);
}

TEST(MediaCodecJpg, CodecEncode) {
  MediaCodecJpg codec(MediaCodecID::kMJPEG, true, 1920, 1080,
                      CodecPixelFormat::kNV12);
}
