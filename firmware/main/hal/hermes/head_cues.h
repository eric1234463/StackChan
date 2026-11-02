/*
 * Head gestures for the Hermes build: face forward at boot, nod while the bridge plays its acknowledgement.
 * Angles are in 0.1 degree (pitch 0 = level, 900 = straight up; yaw 0 = forward). Safe to call from any task.
 */
#pragma once

namespace hermes_head {

void FaceForward();  // yaw 0, pitch 30 degrees up
void Nod();          // two nods from the current pitch; ignored while a nod is running

}  // namespace hermes_head
