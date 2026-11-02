#include "head_cues.h"

#include <atomic>

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <hal/hal.h>
#include <stackchan/stackchan.h>

namespace {

constexpr int kForwardPitch = 300;  // 30 degrees up
constexpr int kNodDepth     = 250;  // dip 25 degrees (to 5 degrees up); 15 was too subtle to notice
constexpr int kNodSpeed     = 800;  // 0-1000; 600 needed ~250 ms per 15-degree half nod
constexpr int kHalfNodMs    = 300;
constexpr int kNods         = 2;
constexpr int kShakeYaw     = 200;  // 20 degrees each side of forward
constexpr int kShakeSpeed   = 500;
constexpr int kHalfShakeMs  = 450;
constexpr int kShakePollMs  = 50;   // how soon a stop request is noticed

std::atomic<bool> g_nodding{false};
std::atomic<bool> g_shaking{false};      // the shake task is running
std::atomic<bool> g_stop_shake{false};  // ask it to finish

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

void MoveYaw(int yaw)
{
    LvglLockGuard lock;
    GetStackChan().motion().yawServo().moveWithSpeed(yaw, kShakeSpeed);
}

void ShakeTask(void*)
{
    for (int side = 1; !g_stop_shake; side = -side) {
        MoveYaw(side * kShakeYaw);
        for (int t = 0; t < kHalfShakeMs && !g_stop_shake; t += kShakePollMs) {
            vTaskDelay(pdMS_TO_TICKS(kShakePollMs));
        }
    }
    MoveYaw(0);
    g_shaking = false;
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

void StartShake()
{
    g_stop_shake = false;
    if (g_shaking.exchange(true)) return;
    if (xTaskCreate(ShakeTask, "hermes_shake", 3072, nullptr, 3, nullptr) != pdPASS) g_shaking = false;
}

void StopShake()
{
    g_stop_shake = true;
}

}  // namespace hermes_head
