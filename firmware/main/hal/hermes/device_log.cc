#include "device_log.h"

#include <cJSON.h>
#include <esp_attr.h>
#include <esp_log.h>
#include <esp_system.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>

#include <cstdarg>
#include <cstdio>

namespace {

constexpr uint32_t kMagic    = 0x48524D4C;  // "HRML"
constexpr size_t kRingBytes  = 32 * 1024;   // ~400 log lines

// Not zeroed at boot (CONFIG_SPIRAM_ALLOW_NOINIT_SEG_EXTERNAL_MEMORY): survives a software reset, not power loss.
// head/uploaded count bytes ever written; the ring holds the last kRingBytes of them.
struct Ring {
    uint32_t magic;
    uint32_t head;
    uint32_t uploaded;
    char buf[kRingBytes];
};
EXT_RAM_NOINIT_ATTR Ring g_ring;

portMUX_TYPE g_lock = portMUX_INITIALIZER_UNLOCKED;
vprintf_like_t g_previous = nullptr;
const char* g_reset_reason = "unknown";

const char* ResetReasonName(esp_reset_reason_t r)
{
    switch (r) {
        case ESP_RST_POWERON: return "poweron";
        case ESP_RST_SW: return "software";
        case ESP_RST_PANIC: return "panic";
        case ESP_RST_INT_WDT: return "int_wdt";
        case ESP_RST_TASK_WDT: return "task_wdt";
        case ESP_RST_WDT: return "wdt";
        case ESP_RST_BROWNOUT: return "brownout";
        case ESP_RST_DEEPSLEEP: return "deepsleep";
        case ESP_RST_USB: return "usb";
        case ESP_RST_JTAG: return "jtag";
        default: return "other";
    }
}

int LogHook(const char* fmt, va_list args)
{
    char line[256];
    va_list copy;
    va_copy(copy, args);
    int n = vsnprintf(line, sizeof(line), fmt, copy);
    va_end(copy);
    if (n <= 0) {
        return n;
    }
    const bool truncated = n >= (int)sizeof(line);
    const int kept       = truncated ? (int)sizeof(line) - 1 : n;  // a truncated line keeps its start
    portENTER_CRITICAL(&g_lock);
    for (int i = 0; i < kept; ++i) {
        g_ring.buf[g_ring.head++ % kRingBytes] = line[i];
    }
    portEXIT_CRITICAL(&g_lock);
    if (!truncated) {
        return fputs(line, stdout) < 0 ? -1 : n;  // already formatted: print it instead of formatting twice
    }
    return g_previous ? g_previous(fmt, args) : vprintf(fmt, args);
}

// WebSocket text frames must be valid UTF-8; a truncated line or the ring's cut point can split a character.
void ReplaceInvalidUtf8(std::string& s)
{
    for (size_t i = 0; i < s.size();) {
        unsigned char c = s[i];
        size_t len      = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
        bool ok         = len > 0 && i + len <= s.size();
        for (size_t k = 1; ok && k < len; ++k) {
            ok = (static_cast<unsigned char>(s[i + k]) >> 6) == 0x2;
        }
        if (!ok) {
            s[i++] = '?';
            continue;
        }
        i += len;
    }
}

}  // namespace

namespace hermes_log {

void Init()
{
    esp_reset_reason_t reason = esp_reset_reason();
    g_reset_reason            = ResetReasonName(reason);
    // A power-on (or garbage) ring starts empty; after a software reset the previous boot's lines are kept.
    if (g_ring.magic != kMagic || reason == ESP_RST_POWERON || reason == ESP_RST_BROWNOUT ||
        g_ring.uploaded > g_ring.head) {
        g_ring.magic    = kMagic;
        g_ring.head     = 0;
        g_ring.uploaded = 0;
    }
    g_previous = esp_log_set_vprintf(LogHook);
    ESP_LOGI("HermesLog", "boot, reset reason %s", g_reset_reason);
}

std::string TakeUploadJson()
{
    static std::string text;  // PSRAM via the default allocator (> SPIRAM_MALLOC_ALWAYSINTERNAL)
    uint32_t dropped;
    portENTER_CRITICAL(&g_lock);
    uint32_t start = g_ring.uploaded;
    uint32_t end   = g_ring.head;
    portEXIT_CRITICAL(&g_lock);
    if (end == start) {
        return "";
    }
    dropped = end - start > kRingBytes ? end - start - kRingBytes : 0;
    start += dropped;
    text.resize(end - start);
    // Copied without the lock (a long critical section would stall Wi-Fi); a writer would need to wrap the whole
    // ring during the copy to corrupt it. Lines written meanwhile go in the next upload.
    for (uint32_t i = start; i < end; ++i) {
        text[i - start] = g_ring.buf[i % kRingBytes];
    }
    portENTER_CRITICAL(&g_lock);
    g_ring.uploaded = end;
    portEXIT_CRITICAL(&g_lock);
    ReplaceInvalidUtf8(text);

    cJSON* root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "type", "device_log");
    cJSON_AddStringToObject(root, "reset_reason", g_reset_reason);
    cJSON_AddNumberToObject(root, "uptime_ms", (double)(esp_timer_get_time() / 1000));
    cJSON_AddNumberToObject(root, "dropped", dropped);
    cJSON_AddStringToObject(root, "text", text.c_str());
    char* json = cJSON_PrintUnformatted(root);
    std::string out = json ? json : "";
    cJSON_free(json);
    cJSON_Delete(root);
    return out;
}

}  // namespace hermes_log
