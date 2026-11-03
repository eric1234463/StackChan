/*
 * Device-side log for the Hermes build: every ESP_LOG line is also kept in a PSRAM ring that survives software
 * resets (panic, watchdog, esp_restart), and the lines not yet uploaded are sent to the bridge each time the
 * channel opens. So what happened while the device was offline, or just before it rebooted, reaches the Mac.
 */
#pragma once

#include <string>

namespace hermes_log {

void Init();  // call first in app_main: hooks esp_log, keeps the ring across a software reset

// JSON message for the bridge: {"type":"device_log","reset_reason":"...","uptime_ms":N,"dropped":N,"text":"..."}
// with the lines since the last call (at most the ring size); "" when there is nothing new.
std::string TakeUploadJson();

}  // namespace hermes_log
