/*
 * Full-screen custom illustration per voice state (spec §6.3), drawn on lv_layer_sys() above the avatar.
 * All functions must be called with the LVGL/display lock held.
 */
#pragma once

#include <device_state.h>

namespace hermes_screen {

void Init();                                  // create the image object (idempotent)
void OnDeviceState(DeviceState state);        // xiaozhi state machine changes
void OnUserTranscript();                      // bridge sent `stt` -> thinking
void OnEmotion(const char* emotion);          // alerts carry their emotion: error / bridge unreachable
void Tick(bool wifi_connected);               // ~1 Hz: Wi-Fi status and timed overlays

}  // namespace hermes_screen
