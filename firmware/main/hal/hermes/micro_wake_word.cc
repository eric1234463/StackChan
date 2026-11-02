#include <sdkconfig.h>
#if CONFIG_HERMES_MICRO_WAKE_WORD
#include "micro_wake_word.h"

#include <audio_service.h>  // AS_OPUS_ENC_CONFIG: the uplink encoder settings

#include <esp_heap_caps.h>
#include <esp_log.h>

#include <algorithm>
#include <cstdio>

#include <esp_timer.h>

#include <frontend_util.h>
#include <tensorflow/lite/micro/micro_allocator.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_resource_variable.h>
#include <tensorflow/lite/schema/schema_generated.h>

#define TAG "MicroWakeWord"

extern const uint8_t model_start[] asm("_binary_hermes_wei_eric_tflite_start");

namespace {

// Runtime parameters of the trained model (see wei_eric.json / training_parameters.yaml).
constexpr float kProbabilityCutoff = 0.20f;
constexpr size_t kTensorArenaSize  = 80000 + 16 * 1024;  // manifest size + headroom
constexpr size_t kVarArenaSize     = 1024;
constexpr int kFeatureCount        = 40;
constexpr int kSampleRate          = 16000;
constexpr int kRefractoryInferences = 33;  // ~1 s at 30 ms per inference (stride 3)

using OpResolver = tflite::MicroMutableOpResolver<20>;

bool RegisterOps(OpResolver& r)
{
    // Op set used by microWakeWord streaming models.
    return r.AddCallOnce() == kTfLiteOk && r.AddVarHandle() == kTfLiteOk && r.AddReshape() == kTfLiteOk &&
           r.AddReadVariable() == kTfLiteOk && r.AddStridedSlice() == kTfLiteOk &&
           r.AddConcatenation() == kTfLiteOk && r.AddAssignVariable() == kTfLiteOk && r.AddConv2D() == kTfLiteOk &&
           r.AddMul() == kTfLiteOk && r.AddAdd() == kTfLiteOk && r.AddMean() == kTfLiteOk &&
           r.AddFullyConnected() == kTfLiteOk && r.AddLogistic() == kTfLiteOk && r.AddQuantize() == kTfLiteOk &&
           r.AddDepthwiseConv2D() == kTfLiteOk && r.AddAveragePool2D() == kTfLiteOk &&
           r.AddMaxPool2D() == kTfLiteOk && r.AddPad() == kTfLiteOk && r.AddPack() == kTfLiteOk &&
           r.AddSplitV() == kTfLiteOk;
}

// Frontend settings the microWakeWord models are trained with (TFLM micro_speech defaults, 30 ms / 10 ms).
void FillFrontendConfig(FrontendConfig& c)
{
    c.window.size_ms                     = 30;
    c.window.step_size_ms                = 10;
    c.filterbank.num_channels            = kFeatureCount;
    c.filterbank.lower_band_limit        = 125.0f;
    c.filterbank.upper_band_limit        = 7500.0f;
    c.filterbank.output_scale_shift      = 7;
    c.noise_reduction.smoothing_bits     = 10;
    c.noise_reduction.even_smoothing     = 0.025f;
    c.noise_reduction.odd_smoothing      = 0.06f;
    c.noise_reduction.min_signal_remaining = 0.05f;
    c.pcan_gain_control.enable_pcan      = 1;
    c.pcan_gain_control.strength         = 0.95f;
    c.pcan_gain_control.offset           = 80.0f;
    c.pcan_gain_control.gain_bits        = 21;
    c.log_scale.enable_log               = 1;
    c.log_scale.scale_shift              = 6;
}

// Frontend output (~0..670) -> int8 model input, as in the TFLM micro_speech feature provider:
// input = feature * 256 / (25.6 * 26.0) - 128.
int8_t QuantizeFeature(uint16_t value)
{
    int32_t v = (static_cast<int32_t>(value) * 256 + 333) / 666 - 128;
    return static_cast<int8_t>(std::clamp<int32_t>(v, -128, 127));
}

OpResolver& Resolver()
{
    static OpResolver resolver;
    static bool registered = RegisterOps(resolver);
    (void)registered;
    return resolver;
}

}  // namespace

MicroWakeWord::MicroWakeWord() = default;

MicroWakeWord::~MicroWakeWord()
{
    interpreter_.reset();
    if (frontend_ready_) {
        FrontendFreeStateContents(&frontend_);
    }
    heap_caps_free(arena_);
    heap_caps_free(var_arena_);
    heap_caps_free(history_);
    heap_caps_free(encode_stack_);
    heap_caps_free(encode_tcb_);
}

bool MicroWakeWord::Initialize(AudioCodec* codec, srmodel_list_t* /*models_list*/)
{
    codec_ = codec;

    FrontendConfig config;
    FillFrontendConfig(config);
    if (!FrontendPopulateState(&config, &frontend_, kSampleRate)) {
        ESP_LOGE(TAG, "Frontend init failed");
        return false;
    }
    frontend_ready_ = true;

    const tflite::Model* model = tflite::GetModel(model_start);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema %lu != %d", model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    arena_     = static_cast<uint8_t*>(heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    var_arena_ = static_cast<uint8_t*>(heap_caps_malloc(kVarArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    history_   = static_cast<int16_t*>(heap_caps_malloc(kHistorySamples * sizeof(int16_t), MALLOC_CAP_SPIRAM));
    if (arena_ == nullptr || var_arena_ == nullptr || history_ == nullptr) {
        ESP_LOGE(TAG, "Arena allocation failed");
        return false;
    }
    auto* var_allocator = tflite::MicroAllocator::Create(var_arena_, kVarArenaSize);
    variables_          = tflite::MicroResourceVariables::Create(var_allocator, 20);
    interpreter_ = std::make_unique<tflite::MicroInterpreter>(model, Resolver(), arena_, kTensorArenaSize, variables_);
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed (missing op or arena too small)");
        interpreter_.reset();
        return false;
    }

    TfLiteTensor* input  = interpreter_->input(0);
    TfLiteTensor* output = interpreter_->output(0);
    if (input->type != kTfLiteInt8 || input->dims->size != 3 || input->dims->data[2] != kFeatureCount ||
        output->type != kTfLiteUInt8) {
        ESP_LOGE(TAG, "Unexpected model I/O shape or type");
        interpreter_.reset();
        return false;
    }
    stride_ = input->dims->data[1];
    ESP_LOGI(TAG, "Ready: %s, stride %d, arena used %u/%u", wake_word_.c_str(), stride_,
             (unsigned)interpreter_->arena_used_bytes(), (unsigned)kTensorArenaSize);
    ResetDetector();
    return true;
}

void MicroWakeWord::OnWakeWordDetected(std::function<void(const std::string& wake_word)> callback)
{
    callback_ = std::move(callback);
}

void MicroWakeWord::ResetDetector()
{
    FrontendReset(&frontend_);
    if (variables_ != nullptr) {
        variables_->ResetAll();
    }
    features_filled_ = 0;
    recent_.fill(0);
    recent_index_      = 0;
    history_filled_    = 0;  // the clip covers only audio since this reset
    ignore_inferences_ = kSlidingWindow;  // let the streaming state settle after a reset
}

void MicroWakeWord::Start()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (interpreter_ == nullptr) {
        return;
    }
    ResetDetector();
    running_ = true;
}

void MicroWakeWord::Stop()
{
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = false;
}

void MicroWakeWord::Feed(const std::vector<int16_t>& data)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) {
        return;
    }

    // Keep the microphone channel only (CoreS3 input is interleaved mic + playback reference).
    const int channels = codec_->input_channels();
    std::vector<int16_t> mono;
    mono.reserve(data.size() / channels);
    for (size_t i = 0; i < data.size(); i += channels) {
        mono.push_back(data[i]);
    }
    for (int16_t sample : mono) {
        history_[history_pos_] = sample;
        history_pos_           = (history_pos_ + 1) % kHistorySamples;
    }
    history_filled_ = std::min(history_filled_ + mono.size(), kHistorySamples);

    size_t offset = 0;
    while (offset < mono.size()) {
        size_t consumed = 0;
        FrontendOutput out = FrontendProcessSamples(&frontend_, mono.data() + offset, mono.size() - offset, &consumed);
        offset += consumed;
        if (out.size != kFeatureCount) {
            if (consumed == 0) {
                break;
            }
            continue;
        }

        int8_t* slot = interpreter_->input(0)->data.int8 + features_filled_ * kFeatureCount;
        for (int i = 0; i < kFeatureCount; ++i) {
            slot[i] = QuantizeFeature(out.values[i]);
        }
        if (++features_filled_ < stride_) {
            continue;
        }
        features_filled_ = 0;

        if (InferAndCheck()) {
            running_ = false;
            if (callback_) {
                callback_(wake_word_);
            }
            return;
        }
    }
}

bool MicroWakeWord::InferAndCheck()
{
    if (interpreter_->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
        return false;
    }
    recent_[recent_index_] = interpreter_->output(0)->data.uint8[0];
    recent_index_          = (recent_index_ + 1) % kSlidingWindow;

    if (ignore_inferences_ > 0) {
        --ignore_inferences_;
        return false;
    }

    int sum = 0;
    for (uint8_t p : recent_) {
        sum += p;
    }
    if (sum < kProbabilityCutoff * 255.0f * kSlidingWindow) {
        return false;
    }
    // Oldest first; logged and sent to the bridge so false triggers can be told apart by score.
    char detail[96];
    int n = snprintf(detail, sizeof(detail), "\"score\":%.2f,\"window\":[", sum / (255.0f * kSlidingWindow));
    for (int i = 0; i < kSlidingWindow; ++i) {
        n += snprintf(detail + n, sizeof(detail) - n, "%s%.2f", i ? "," : "",
                      recent_[(recent_index_ + i) % kSlidingWindow] / 255.0f);
    }
    snprintf(detail + n, sizeof(detail) - n, "]");
    detail_ = detail;
    ESP_LOGI(TAG, "Detected %s (%s)", wake_word_.c_str(), detail);
    ignore_inferences_ = kRefractoryInferences;
    return true;
}

void MicroWakeWord::EncodeWakeWordData()
{
    {
        // A clip nobody consumed (session aborted) may still be encoding; its task owns the static stack.
        std::unique_lock<std::mutex> lock(opus_mutex_);
        opus_cv_.wait(lock, [this]() { return !encoding_; });
        opus_.clear();
        encoding_ = true;
    }
    {
        // Feed is stopped after a detection (running_ false), so the history is stable here.
        std::lock_guard<std::mutex> lock(mutex_);
        encode_pcm_.resize(history_filled_);
        size_t start = (history_pos_ + kHistorySamples - history_filled_) % kHistorySamples;
        for (size_t i = 0; i < history_filled_; ++i) {
            encode_pcm_[i] = history_[(start + i) % kHistorySamples];
        }
    }
    // Opus encoding needs a large stack; keep it in PSRAM (as upstream CustomWakeWord does).
    constexpr size_t kStackSize = 4096 * 7;
    if (encode_stack_ == nullptr) {
        encode_stack_ = static_cast<StackType_t*>(heap_caps_malloc(kStackSize, MALLOC_CAP_SPIRAM));
        encode_tcb_   = static_cast<StaticTask_t*>(heap_caps_malloc(sizeof(StaticTask_t), MALLOC_CAP_INTERNAL));
    }
    if (encode_stack_ == nullptr || encode_tcb_ == nullptr) {
        ESP_LOGE(TAG, "Wake audio encoder allocation failed");
        std::lock_guard<std::mutex> lock(opus_mutex_);
        encoding_ = false;
        opus_.emplace_back();  // end of (empty) clip
        opus_cv_.notify_all();
        return;
    }
    xTaskCreateStatic(
        [](void* arg) {
            auto* self = static_cast<MicroWakeWord*>(arg);
            auto finish = [self]() {
                std::lock_guard<std::mutex> lock(self->opus_mutex_);
                self->encoding_ = false;
                self->opus_.emplace_back();  // end of clip
                self->opus_cv_.notify_all();
            };
            int64_t t0                        = esp_timer_get_time();
            esp_opus_enc_config_t cfg         = AS_OPUS_ENC_CONFIG();
            void* encoder                     = nullptr;
            esp_opus_enc_open(&cfg, sizeof(cfg), &encoder);
            if (encoder == nullptr) {
                ESP_LOGE(TAG, "Wake audio encoder open failed");
                finish();
                vTaskDelete(nullptr);
                return;
            }
            int frame_bytes = 0, out_size = 0;
            esp_opus_enc_get_frame_size(encoder, &frame_bytes, &out_size);
            const size_t frame = frame_bytes / sizeof(int16_t);
            std::vector<uint8_t> out_buf(out_size);
            int packets = 0;
            for (size_t i = 0; i + frame <= self->encode_pcm_.size(); i += frame) {
                esp_audio_enc_in_frame_t in   = {};
                esp_audio_enc_out_frame_t out = {};
                in.buffer  = reinterpret_cast<uint8_t*>(self->encode_pcm_.data() + i);
                in.len     = frame_bytes;
                out.buffer = out_buf.data();
                out.len    = out_size;
                // Zero-byte (DTX) packets are skipped: an empty packet marks the end of the clip.
                if (esp_opus_enc_process(encoder, &in, &out) == ESP_AUDIO_ERR_OK && out.encoded_bytes > 0) {
                    std::lock_guard<std::mutex> lock(self->opus_mutex_);
                    self->opus_.emplace_back(out_buf.begin(), out_buf.begin() + out.encoded_bytes);
                    self->opus_cv_.notify_all();
                    ++packets;
                }
            }
            esp_opus_enc_close(encoder);
            ESP_LOGI(TAG, "Wake audio: %d packets in %d ms", packets, (int)((esp_timer_get_time() - t0) / 1000));
            finish();
            vTaskDelete(nullptr);
        },
        "encode_wake_word", kStackSize, this, 2, encode_stack_, encode_tcb_);
}

bool MicroWakeWord::GetWakeWordOpus(std::vector<uint8_t>& opus)
{
    std::unique_lock<std::mutex> lock(opus_mutex_);
    opus_cv_.wait(lock, [this]() { return !opus_.empty(); });
    opus.swap(opus_.front());
    opus_.pop_front();
    return !opus.empty();
}

#endif  // CONFIG_HERMES_MICRO_WAKE_WORD
