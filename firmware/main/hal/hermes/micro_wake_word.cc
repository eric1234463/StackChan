#include <sdkconfig.h>
#if CONFIG_HERMES_MICRO_WAKE_WORD
#include "micro_wake_word.h"

#include <esp_heap_caps.h>
#include <esp_log.h>

#include <algorithm>

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
    if (arena_ == nullptr || var_arena_ == nullptr) {
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
    ESP_LOGI(TAG, "Detected %s (mean probability %.2f)", wake_word_.c_str(), sum / (255.0f * kSlidingWindow));
    ignore_inferences_ = kRefractoryInferences;
    return true;
}

#endif  // CONFIG_HERMES_MICRO_WAKE_WORD
