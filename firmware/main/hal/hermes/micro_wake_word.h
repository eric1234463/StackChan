/*
 * On-device microWakeWord ("喂Eric") for the xiaozhi WakeWord interface.
 * Features: TFLM micro-frontend (esphome/esp-micro-speech-features, Apache-2.0).
 * Inference: espressif/esp-tflite-micro streaming model with resource variables.
 */
#pragma once

#include <wake_word.h>

#include <array>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <frontend.h>

namespace tflite {
class MicroInterpreter;
class MicroResourceVariables;
}  // namespace tflite

class MicroWakeWord : public WakeWord {
public:
    MicroWakeWord();
    ~MicroWakeWord() override;

    bool Initialize(AudioCodec* codec, srmodel_list_t* models_list) override;
    void Feed(const std::vector<int16_t>& data) override;
    void OnWakeWordDetected(std::function<void(const std::string& wake_word)> callback) override;
    void Start() override;
    void Stop() override;
    size_t GetFeedSize() override { return 160; }  // 10 ms at 16 kHz
    void EncodeWakeWordData() override {}
    bool GetWakeWordOpus(std::vector<uint8_t>& opus) override { return false; }
    const std::string& GetLastDetectedWakeWord() const override { return wake_word_; }

private:
    static constexpr int kSlidingWindow = 5;  // from wei_eric.json

    bool InferAndCheck();
    void ResetDetector();

    AudioCodec* codec_ = nullptr;
    std::function<void(const std::string&)> callback_;
    std::mutex mutex_;
    bool running_ = false;

    FrontendState frontend_{};
    bool frontend_ready_ = false;

    uint8_t* arena_     = nullptr;
    uint8_t* var_arena_ = nullptr;
    tflite::MicroResourceVariables* variables_ = nullptr;
    std::unique_ptr<tflite::MicroInterpreter> interpreter_;
    int stride_          = 0;  // feature frames consumed per inference (model input dim 1)
    int features_filled_ = 0;

    std::array<uint8_t, kSlidingWindow> recent_{};
    int recent_index_      = 0;
    int ignore_inferences_ = 0;

    const std::string wake_word_ = "喂Eric";
};
