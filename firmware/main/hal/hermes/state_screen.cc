#include <sdkconfig.h>
#if CONFIG_HERMES_STATE_IMAGES
#include "state_screen.h"

#include <esp_log.h>
#include <esp_timer.h>
#include <lvgl.h>

#include <cstring>

#define TAG "HermesScreen"

namespace {

enum Image { kLoading, kIdle, kListening, kThinking, kWorking, kReplying, kError, kNoWifi, kNoHa, kImageCount };

#define HERMES_PNG(name)                                                       \
    extern const uint8_t name##_start[] asm("_binary_hermes_" #name "_png_start"); \
    extern const uint8_t name##_end[] asm("_binary_hermes_" #name "_png_end");
HERMES_PNG(loading)
HERMES_PNG(idle)
HERMES_PNG(listening)
HERMES_PNG(thinking)
HERMES_PNG(replying)
HERMES_PNG(error)
HERMES_PNG(no_wifi)
HERMES_PNG(no_ha)

constexpr int64_t kErrorShowUs = 3 * 1000 * 1000;
constexpr int64_t kNoHaShowUs  = 10 * 1000 * 1000;

lv_image_dsc_t g_dsc[kImageCount];
lv_obj_t* g_image       = nullptr;
int g_current           = -1;
DeviceState g_state     = kDeviceStateStarting;
bool g_thinking         = false;  // transcript received; kept through the ack sentence
bool g_working          = false;  // ack played, waiting for Hermes
bool g_wifi_ok          = true;
int64_t g_error_until   = 0;
int64_t g_no_ha_until   = 0;

void Describe(Image id, const uint8_t* start, const uint8_t* end)
{
    lv_image_dsc_t& d = g_dsc[id];
    memset(&d, 0, sizeof(d));
    d.header.magic = LV_IMAGE_HEADER_MAGIC;
    d.header.cf    = LV_COLOR_FORMAT_RAW;  // decoded by lodepng
    d.header.w     = 320;
    d.header.h     = 240;
    d.data         = start;
    d.data_size    = static_cast<uint32_t>(end - start);
}

Image Desired()
{
    const int64_t now = esp_timer_get_time();
    // Priority: no_wifi > no_ha > error > activity > idle (spec §6.3).
    if (!g_wifi_ok) return kNoWifi;
    if (now < g_no_ha_until) return kNoHa;
    if (now < g_error_until) return kError;
    switch (g_state) {
        case kDeviceStateIdle:
            return kIdle;
        case kDeviceStateListening:
            return g_thinking ? kThinking : kListening;
        case kDeviceStateSpeaking:
            if (g_working) return kWorking;
            return g_thinking ? kThinking : kReplying;
        default:
            return kLoading;  // starting, connecting, activating, upgrading
    }
}

void Refresh()
{
    if (g_image == nullptr) return;
    const Image want = Desired();
    if (want == g_current) return;
    g_current = want;
    lv_image_set_src(g_image, &g_dsc[want]);
}

}  // namespace

namespace hermes_screen {

void Init()
{
    if (g_image != nullptr) return;
    Describe(kLoading, loading_start, loading_end);
    Describe(kIdle, idle_start, idle_end);
    Describe(kListening, listening_start, listening_end);
    Describe(kThinking, thinking_start, thinking_end);
    // SHORTCUT: working reuses the loading art until a dedicated hermes_working.png exists.
    Describe(kWorking, loading_start, loading_end);
    Describe(kReplying, replying_start, replying_end);
    Describe(kError, error_start, error_end);
    Describe(kNoWifi, no_wifi_start, no_wifi_end);
    Describe(kNoHa, no_ha_start, no_ha_end);

    // System layer sits above the avatar, status bar and decorators; not clickable,
    // so taps still reach the avatar panel underneath (tap-to-talk).
    g_image = lv_image_create(lv_layer_sys());
    lv_obj_remove_flag(g_image, LV_OBJ_FLAG_CLICKABLE);
    lv_obj_set_size(g_image, 320, 240);
    lv_obj_align(g_image, LV_ALIGN_CENTER, 0, 0);
    Refresh();
    ESP_LOGI(TAG, "State screen ready");
}

void OnDeviceState(DeviceState state)
{
    // The bridge keeps one TTS session over ack + reply, so the turn's flags live through Speaking.
    if (state != kDeviceStateSpeaking) {
        g_thinking = false;
        g_working  = false;
    }
    if (state == kDeviceStateListening || state == kDeviceStateSpeaking) g_no_ha_until = 0;  // bridge reachable
    g_state = state;
    Refresh();
}

void OnUserTranscript()
{
    g_thinking = true;
    Refresh();
}

void OnAssistantSentence()
{
    g_thinking = false;
    g_working  = false;
    Refresh();
}

void OnEmotion(const char* emotion)
{
    if (emotion == nullptr) return;
    const int64_t now = esp_timer_get_time();
    if (strcmp(emotion, "working") == 0) {
        g_working = true;  // bridge: ack played, Hermes still working
    } else if (strcmp(emotion, "circle_xmark") == 0 || strcmp(emotion, "cloud_slash") == 0) {
        g_no_ha_until = now + kNoHaShowUs;  // WebSocket connect/handshake to the bridge failed
    } else if (strcmp(emotion, "sad") == 0 || strcmp(emotion, "triangle_exclamation") == 0) {
        g_error_until = now + kErrorShowUs;  // bridge alert (STT/TTS failure) or device error
    } else {
        return;
    }
    Refresh();
}

void Tick(bool wifi_connected)
{
    g_wifi_ok = wifi_connected;
    Refresh();  // also expires timed error / no_ha overlays
}

}  // namespace hermes_screen

#endif  // CONFIG_HERMES_STATE_IMAGES
