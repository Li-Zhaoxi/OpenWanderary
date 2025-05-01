#pragma once
#include <string>

#include <hb_media_codec.h>
#include <wanderary/utils/enum_traits.h>

ENUM_NUMBERED_REGISTER(
    MediaCodecID,
    ((kNONE, media_codec_id_t::MEDIA_CODEC_ID_NONE, "media_codec_id_none"))
    /* Video Codecs */
    ((kH264, media_codec_id_t::MEDIA_CODEC_ID_H264, "media_codec_id_h264"))(
        (kH265, media_codec_id_t::MEDIA_CODEC_ID_H265, "media_codec_id_h265"))((
        kMJPEG, media_codec_id_t::MEDIA_CODEC_ID_MJPEG, "media_codec_id_mjpeg"))
    /* Audio Codecs */
    ((kFLAC, media_codec_id_t::MEDIA_CODEC_ID_FLAC, "media_codec_id_flac"))((
        kPCM_MULAW, media_codec_id_t::MEDIA_CODEC_ID_PCM_MULAW,
        "media_codec_id_pcm_mulaw"))((kPCM_ALAW,
                                      media_codec_id_t::MEDIA_CODEC_ID_PCM_ALAW,
                                      "media_codec_id_pcm_alaw"))(
        (kADPCM_G726, media_codec_id_t::MEDIA_CODEC_ID_ADPCM_G726,
         "media_codec_id_adpcm_g726"))((kADPCM,
                                        media_codec_id_t::MEDIA_CODEC_ID_ADPCM,
                                        "media_codec_id_adpcm"))(
        (kAAC, media_codec_id_t::MEDIA_CODEC_ID_AAC, "media_codec_id_aac"))(
        (kMP3, media_codec_id_t::MEDIA_CODEC_ID_MP3, "media_codec_id_mp3"))(
        (kMP2, media_codec_id_t::MEDIA_CODEC_ID_MP2, "media_codec_id_mp2"))(
        (kTAK, media_codec_id_t::MEDIA_CODEC_ID_TAK, "media_codec_id_tak"))(
        (kAC3, media_codec_id_t::MEDIA_CODEC_ID_AC3, "media_codec_id_ac3"))(
        (kWMA, media_codec_id_t::MEDIA_CODEC_ID_WMA, "media_codec_id_wma"))(
        (kAMR, media_codec_id_t::MEDIA_CODEC_ID_AMR, "media_codec_id_amr"))(
        (kAPE, media_codec_id_t::MEDIA_CODEC_ID_APE, "media_codec_id_ape"))(
        (kG729, media_codec_id_t::MEDIA_CODEC_ID_G729, "media_codec_id_g729"))(
        (kG723, media_codec_id_t::MEDIA_CODEC_ID_G723, "media_codec_id_g723"))(
        (kG722, media_codec_id_t::MEDIA_CODEC_ID_G722, "media_codec_id_g722"))(
        (kIAC, media_codec_id_t::MEDIA_CODEC_ID_IAC, "media_codec_id_iac"))(
        (kRALF, media_codec_id_t::MEDIA_CODEC_ID_RALF, "media_codec_id_ralf"))(
        (kQDMC, media_codec_id_t::MEDIA_CODEC_ID_QDMC, "media_codec_id_qdmc"))(
        (kDTS, media_codec_id_t::MEDIA_CODEC_ID_DTS, "media_codec_id_dts"))(
        (kGSM, media_codec_id_t::MEDIA_CODEC_ID_GSM, "media_codec_id_gsm"))(
        (kTTA, media_codec_id_t::MEDIA_CODEC_ID_TTA, "media_codec_id_tta"))(
        (kQCELP, media_codec_id_t::MEDIA_CODEC_ID_QCELP,
         "media_codec_id_qcelp"))((kMLP, media_codec_id_t::MEDIA_CODEC_ID_MLP,
                                   "media_codec_id_mlp"))(
        (kATRAC1, media_codec_id_t::MEDIA_CODEC_ID_ATRAC1,
         "media_codec_id_atrac1"))((kIMC, media_codec_id_t::MEDIA_CODEC_ID_IMC,
                                    "media_codec_id_imc"))(
        (kEAC, media_codec_id_t::MEDIA_CODEC_ID_EAC, "media_codec_id_eac"))(
        (kMP1, media_codec_id_t::MEDIA_CODEC_ID_MP1, "media_codec_id_mp1"))(
        (kSIPR, media_codec_id_t::MEDIA_CODEC_ID_SIPR, "media_codec_id_sipr"))(
        (kOPUS, media_codec_id_t::MEDIA_CODEC_ID_OPUS, "media_codec_id_opus"))(
        (kCELT, media_codec_id_t::MEDIA_CODEC_ID_CELT, "media_codec_id_celt"))
    /* Subtitle Codecs */
    ((kMOV_TEXT, media_codec_id_t::MEDIA_CODEC_ID_MOV_TEXT,
      "media_codec_id_mov_text"))((kTOTAL,
                                   media_codec_id_t::MEDIA_CODEC_ID_TOTAL,
                                   "media_codec_id_total"))

)
ENUM_CONVERSION_REGISTER(MediaCodecID, MediaCodecID::kNONE,
                         "media_codec_id_none")

ENUM_NUMBERED_REGISTER(
    MediaCodecMode,
    ((kNONE, -1, "media_codec_mode_none"))(
        (kSOFTWARE, media_codec_mode_t::MC_SOFTWARE, "media_codec_software"))(
        (kHARDWARE, media_codec_mode_t::MC_HARDWARE, "media_codec_hardware")))
ENUM_CONVERSION_REGISTER(MediaCodecMode, MediaCodecMode::kNONE,
                         "media_codec_mode_none")

namespace wdr::media {

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

CodecDescriptor GetCodecDescriptor(MediaCodecID codec_id);

}  // namespace wdr::media
