#include "head_cues.h"

#include <atomic>

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <hal/hal.h>
#include <stackchan/stackchan.h>

namespace {

constexpr int kForwardPitch = 300;  // 30 degrees up
constexpr int kNodDepth     = 150;  // dip 15 degrees
constexpr int kNodSpeed     = 600;  // 0-1000; the spring animation needs ~250 ms per half nod at this speed
constexpr int kHalfNodMs    = 250;
constexpr int kNods         = 2;

std::atomic<bool> g_nodding{false};

void NodTask(void*)
{
    int base;
    {
        LvglLockGuard lock;
        base = GetStackChan().motion().pitchServo().getCurrentAngle();
    }
    for (int i = 0; i < kNods; i++) {
        for (int target : {base - kNodDepth, base}) {  // the servo clamps to its 3-87 degree limit
            {
                LvglLockGuard lock;
                GetStackChan().motion().pitchServo().moveWithSpeed(target, kNodSpeed);
            }
            vTaskDelay(pdMS_TO_TICKS(kHalfNodMs));
        }
    }
    g_nodding = false;
    vTaskDelete(nullptr);
}

}  // namespace

namespace hermes_head {

void FaceForward()
{
    LvglLockGuard lock;
    GetStackChan().motion().moveWithSpeed(0, kForwardPitch, 250);
}

void Nod()
{
    if (g_nodding.exchange(true)) return;
    if (xTaskCreate(NodTask, "hermes_nod", 3072, nullptr, 3, nullptr) != pdPASS) g_nodding = false;
}

}  // namespace hermes_head
