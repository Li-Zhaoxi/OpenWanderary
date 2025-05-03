#pragma once
#include <string>

#include <hb_media_codec.h>
#include <wanderary/utils/enum_traits.h>

ENUM_NUMBERED_REGISTER(
    MediaCodecID,
    ((kNONE, media_codec_id_t::MEDIA_CODEC_ID_NONE, "media_codec_id_none"))
    /* Video Codecs */
    ((kH264, media_codec_id_t::MEDIA_CODEC_ID_H264, "media_codec_id_h264"))  //
    ((kH265, media_codec_id_t::MEDIA_CODEC_ID_H265, "media_codec_id_h265"))  //
    ((kMJPEG, media_codec_id_t::MEDIA_CODEC_ID_MJPEG,
      "media_codec_id_mjpeg"))                                               //
    ((kJPEG, media_codec_id_t::MEDIA_CODEC_ID_JPEG, "media_codec_id_jpeg"))  //
    /* Audio Codecs */
    ((kFLAC, media_codec_id_t::MEDIA_CODEC_ID_FLAC, "media_codec_id_flac"))  //
    ((kPCM_MULAW, media_codec_id_t::MEDIA_CODEC_ID_PCM_MULAW,
      "media_codec_id_pcm_mulaw"))  //
    ((kPCM_ALAW, media_codec_id_t::MEDIA_CODEC_ID_PCM_ALAW,
      "media_codec_id_pcm_alaw"))  //
    ((kADPCM_G726, media_codec_id_t::MEDIA_CODEC_ID_ADPCM_G726,
      "media_codec_id_adpcm_g726"))  //
    ((kADPCM, media_codec_id_t::MEDIA_CODEC_ID_ADPCM,
      "media_codec_id_adpcm"))                                               //
    ((kAAC, media_codec_id_t::MEDIA_CODEC_ID_AAC, "media_codec_id_aac"))     //
    ((kMP3, media_codec_id_t::MEDIA_CODEC_ID_MP3, "media_codec_id_mp3"))     //
    ((kMP2, media_codec_id_t::MEDIA_CODEC_ID_MP2, "media_codec_id_mp2"))     //
    ((kTAK, media_codec_id_t::MEDIA_CODEC_ID_TAK, "media_codec_id_tak"))     //
    ((kAC3, media_codec_id_t::MEDIA_CODEC_ID_AC3, "media_codec_id_ac3"))     //
    ((kWMA, media_codec_id_t::MEDIA_CODEC_ID_WMA, "media_codec_id_wma"))     //
    ((kAMR, media_codec_id_t::MEDIA_CODEC_ID_AMR, "media_codec_id_amr"))     //
    ((kAPE, media_codec_id_t::MEDIA_CODEC_ID_APE, "media_codec_id_ape"))     //
    ((kG729, media_codec_id_t::MEDIA_CODEC_ID_G729, "media_codec_id_g729"))  //
    ((kG723, media_codec_id_t::MEDIA_CODEC_ID_G723, "media_codec_id_g723"))  //
    ((kG722, media_codec_id_t::MEDIA_CODEC_ID_G722, "media_codec_id_g722"))  //
    ((kIAC, media_codec_id_t::MEDIA_CODEC_ID_IAC, "media_codec_id_iac"))     //
    ((kRALF, media_codec_id_t::MEDIA_CODEC_ID_RALF, "media_codec_id_ralf"))  //
    ((kQDMC, media_codec_id_t::MEDIA_CODEC_ID_QDMC, "media_codec_id_qdmc"))  //
    ((kDTS, media_codec_id_t::MEDIA_CODEC_ID_DTS, "media_codec_id_dts"))     //
    ((kGSM, media_codec_id_t::MEDIA_CODEC_ID_GSM, "media_codec_id_gsm"))     //
    ((kTTA, media_codec_id_t::MEDIA_CODEC_ID_TTA, "media_codec_id_tta"))     //
    ((kQCELP, media_codec_id_t::MEDIA_CODEC_ID_QCELP,
      "media_codec_id_qcelp"))                                            //
    ((kMLP, media_codec_id_t::MEDIA_CODEC_ID_MLP, "media_codec_id_mlp"))  //
    ((kATRAC1, media_codec_id_t::MEDIA_CODEC_ID_ATRAC1,
      "media_codec_id_atrac1"))  //
    ((kIMC, media_codec_id_t::MEDIA_CODEC_ID_IMC,
      "media_codec_id_imc"))                                                 //
    ((kEAC, media_codec_id_t::MEDIA_CODEC_ID_EAC, "media_codec_id_eac"))     //
    ((kMP1, media_codec_id_t::MEDIA_CODEC_ID_MP1, "media_codec_id_mp1"))     //
    ((kSIPR, media_codec_id_t::MEDIA_CODEC_ID_SIPR, "media_codec_id_sipr"))  //
    ((kOPUS, media_codec_id_t::MEDIA_CODEC_ID_OPUS, "media_codec_id_opus"))  //
    ((kCELT, media_codec_id_t::MEDIA_CODEC_ID_CELT, "media_codec_id_celt"))
    /* Subtitle Codecs */
    ((kMOV_TEXT, media_codec_id_t::MEDIA_CODEC_ID_MOV_TEXT,
      "media_codec_id_mov_text"))  //
)
ENUM_CONVERSION_REGISTER(MediaCodecID, MediaCodecID::kNONE,
                         "media_codec_id_none")

ENUM_NUMBERED_REGISTER(
    MediaCodecMode,                                                         //
    ((kNONE, -1, "media_codec_mode_none"))                                  //
    ((kSOFTWARE, media_codec_mode_t::MC_SOFTWARE, "media_codec_software"))  //
    ((kHARDWARE, media_codec_mode_t::MC_HARDWARE, "media_codec_hardware"))  //
)
ENUM_CONVERSION_REGISTER(MediaCodecMode, MediaCodecMode::kNONE,
                         "media_codec_mode_none")

ENUM_NUMBERED_REGISTER(
    MediaCodecState,  //
    ((kNone, media_codec_state_t::MEDIA_CODEC_STATE_NONE,
      "media_codec_state_none"))  //
    ((kUninitialized, media_codec_state_t::MEDIA_CODEC_STATE_UNINITIALIZED,
      "media_codec_state_uninitialized"))  //
    ((kInitialized, media_codec_state_t::MEDIA_CODEC_STATE_INITIALIZED,
      "media_codec_state_initialized"))  //
    ((kConfigured, media_codec_state_t::MEDIA_CODEC_STATE_CONFIGURED,
      "media_codec_state_configured"))  //
    ((kStarted, media_codec_state_t::MEDIA_CODEC_STATE_STARTED,
      "media_codec_state_started"))  //
    ((kPaused, media_codec_state_t::MEDIA_CODEC_STATE_PAUSED,
      "media_codec_state_paused"))  //
    ((kFlushing, media_codec_state_t::MEDIA_CODEC_STATE_FLUSHING,
      "media_codec_state_flushing"))  //
    ((kError, media_codec_state_t::MEDIA_CODEC_STATE_ERROR,
      "media_codec_state_error"))  //
)
ENUM_CONVERSION_REGISTER(MediaCodecState, MediaCodecState::kNone,
                         "media_codec_state_none")

ENUM_NUMBERED_REGISTER(  //
    CodecPixelFormat,    //
    ((kNONE, mc_pixel_format_t::MC_PIXEL_FORMAT_NONE,
      "media_codec_pixel_format_none"))  //
    ((kYUV420P, mc_pixel_format_t::MC_PIXEL_FORMAT_YUV420P,
      "media_codec_pixel_format_yuv420p"))  //
    ((kNV12, mc_pixel_format_t::MC_PIXEL_FORMAT_NV12,
      "media_codec_pixel_format_nv12"))  //
    ((kNV21, mc_pixel_format_t::MC_PIXEL_FORMAT_NV21,
      "media_codec_pixel_format_nv21"))  //
    ((kYUV422P, mc_pixel_format_t::MC_PIXEL_FORMAT_YUV422P,
      "media_codec_pixel_format_yuv422p"))  //
    ((kNV16, mc_pixel_format_t::MC_PIXEL_FORMAT_NV16,
      "media_codec_pixel_format_nv16"))  //
    ((kNV61, mc_pixel_format_t::MC_PIXEL_FORMAT_NV61,
      "media_codec_pixel_format_nv61"))  //
    ((kYUYV422, mc_pixel_format_t::MC_PIXEL_FORMAT_YUYV422,
      "media_codec_pixel_format_yuyv422"))  //
    ((kYVYU422, mc_pixel_format_t::MC_PIXEL_FORMAT_YVYU422,
      "media_codec_pixel_format_yvyu422"))  //
    ((kUYVY422, mc_pixel_format_t::MC_PIXEL_FORMAT_UYVY422,
      "media_codec_pixel_format_uyvy422"))  //
    ((kVYUY422, mc_pixel_format_t::MC_PIXEL_FORMAT_VYUY422,
      "media_codec_pixel_format_vyuy422"))  //
    ((kYUV444, mc_pixel_format_t::MC_PIXEL_FORMAT_YUV444,
      "media_codec_pixel_format_yuv444"))  //
    ((kYUV444P, mc_pixel_format_t::MC_PIXEL_FORMAT_YUV444P,
      "media_codec_pixel_format_yuv444p"))  //
    ((kNV24, mc_pixel_format_t::MC_PIXEL_FORMAT_NV24,
      "media_codec_pixel_format_nv24"))  //
    ((kNV42, mc_pixel_format_t::MC_PIXEL_FORMAT_NV42,
      "media_codec_pixel_format_nv42"))  //
    ((kYUV440P, mc_pixel_format_t::MC_PIXEL_FORMAT_YUV440P,
      "media_codec_pixel_format_yuv440p"))  //
    ((kYUV400, mc_pixel_format_t::MC_PIXEL_FORMAT_YUV400,
      "media_codec_pixel_format_yuv400"))  //
)
ENUM_CONVERSION_REGISTER(CodecPixelFormat, CodecPixelFormat::kNONE,
                         "media_codec_pixel_format_none")

namespace wdr::media {

bool is_codec_video(MediaCodecID id);
bool is_codec_jpg(MediaCodecID id);

class CodecDescriptor {
 public:
  explicit CodecDescriptor(const media_codec_descriptor_t *desc)
      : desc_(desc) {}

  bool has_value() const { return desc_ != nullptr; }
  MediaCodecID id() const;
  MediaCodecMode mode() const;
  std::string name() const;
  std::string long_name() const;
  std::string mime_types() const;

 private:
  const media_codec_descriptor_t *desc_{nullptr};
};

class CodecContext {
 public:
  CodecContext();

  media_codec_context_t &mutable_context() { return ctx_; }

  MediaCodecID id() const;
  bool encoder() const { return ctx_.encoder; }
  int instance_index() const { return ctx_.instance_index; }

  int width() const;
  int height() const;

  static CodecContext CreateJpgEncode(MediaCodecID codec_id, int width,
                                      int height, CodecPixelFormat pixfmt);

 private:
  media_codec_context_t ctx_;
};

class CodecStartupParams {
 public:
  explicit CodecStartupParams(MediaCodecID codec_id, bool encode);

  void set_receive_frame_number(int frame_number);
  mc_av_codec_startup_params_t &mutable_params() { return params_; }

 private:
  const MediaCodecID codec_id_;
  const bool encode_;
  mc_av_codec_startup_params_t params_;
};

CodecDescriptor GetCodecDescriptor(MediaCodecID codec_id);
CodecContext GetDefaultContext(MediaCodecID codec_id, bool encoder);
void InitializeCodecContext(CodecContext *ctx);
MediaCodecState GetCodecState(CodecContext *ctx);
void ReleaseCodecContext(CodecContext *ctx);
void CodecConfigure(CodecContext *ctx);
void CodecStart(CodecContext *ctx, CodecStartupParams *params);
void CodecStop(CodecContext *ctx);

}  // namespace wdr::media
