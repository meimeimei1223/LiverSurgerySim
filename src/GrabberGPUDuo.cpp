// GrabberGPUDuo.cpp
#include "GrabberGPUDuo.h"
#include <iostream>
#include <limits>

GrabberGPUDuo::GrabberGPUDuo() : physicsObject(nullptr), grabDistance(0.0f),
    prevPosition(0.0f), velocity(0.0f), time(0.0f), hit_position(0.0f),
    pView(nullptr), pProjection(nullptr),
    pWindowWidth(nullptr), pWindowHeight(nullptr),
    isDragging_(false),  // ★変更: 内部変数を初期化
    isSmartGrabMode(false), sphereGrabOffset(0.0f) {}

void GrabberGPUDuo::setPhysicsObject(SoftBodyGPUDuo* object) {
    physicsObject = object;
}

// ★変更: isDraggingポインタを削除
void GrabberGPUDuo::setGlobalRefs(glm::mat4* view, glm::mat4* projection,
                                  int* windowWidth, int* windowHeight) {
    pView = view;
    pProjection = projection;
    pWindowWidth = windowWidth;
    pWindowHeight = windowHeight;
    // pIsDragging は削除
}

// ★追加: 外部からドラッグ状態を強制リセット（モード切替時用）
void GrabberGPUDuo::forceEndDrag() {
    if (isDragging_) {
        if (physicsObject) {
            // スマートグラブモードの場合
            if (isSmartGrabMode) {
                physicsObject->smartEndGrab(hit_position, velocity);
                isSmartGrabMode = false;
            } else {
                physicsObject->endLowResGrab(hit_position, velocity);
            }
        }
        isDragging_ = false;
        std::cout << "Grab forcefully ended" << std::endl;
    }
}

void GrabberGPUDuo::startGrab(float screenX, float screenY) {
    auto t0 = std::chrono::high_resolution_clock::now();

    std::cout << "=== Grabber::startGrab called ===" << std::endl;

    if (!physicsObject) {
        std::cout << "ERROR: No physics object!" << std::endl;
        return;
    }
    if (!pView || !pProjection || !pWindowWidth || !pWindowHeight) {
        std::cout << "ERROR: Global references not set!" << std::endl;
        return;
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    RayCastGPUDuo::Ray worldRay = RayCastGPUDuo::screenToRay(screenX, screenY,
                                                             *pView, *pProjection,
                                                             glm::vec4(0, 0, *pWindowWidth, *pWindowHeight));

    auto t2 = std::chrono::high_resolution_clock::now();

    RayCastGPUDuo::RayHit hit = RayCastGPUDuo::intersectMesh(worldRay, *physicsObject);

    auto t3 = std::chrono::high_resolution_clock::now();

    if (hit.hit) {
        isDragging_ = true;  // ★変更: 内部変数を使用
        hit_position = hit.position;
        grabDistance = glm::length(hit_position - worldRay.origin);
        prevPosition = hit_position;
        velocity = glm::vec3(0.0f);
        time = 0.0f;

        auto t4 = std::chrono::high_resolution_clock::now();

        physicsObject->startLowResGrab(hit_position);

        auto t5 = std::chrono::high_resolution_clock::now();

        std::cout << "=== startGrab Timing ===" << std::endl;
        std::cout << "Validation: " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0 << " ms" << std::endl;
        std::cout << "screenToRay: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0 << " ms" << std::endl;
        std::cout << "intersectMesh: " << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / 1000.0 << " ms" << std::endl;
        std::cout << "Setup vars: " << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() / 1000.0 << " ms" << std::endl;
        std::cout << "startLowResGrab: " << std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count() / 1000.0 << " ms" << std::endl;
        std::cout << "TOTAL: " << std::chrono::duration_cast<std::chrono::microseconds>(t5 - t0).count() / 1000.0 << " ms" << std::endl;
        std::cout << "========================" << std::endl;
    }
}



void GrabberGPUDuo::moveGrab(float screenX, float screenY, float deltaTime) {
    if (!physicsObject) {
        std::cout << "moveGrab: No physics object!" << std::endl;
        return;
    }
    RayCastGPUDuo::Ray worldRay = RayCastGPUDuo::screenToRay(screenX, screenY, *pView, *pProjection,
                                                             glm::vec4(0, 0, *pWindowWidth, *pWindowHeight));
    glm::vec3 newPosition = worldRay.origin + worldRay.direction * grabDistance;
    if (time > 0.0f) {
        velocity = (newPosition - prevPosition) / time;
    }
    hit_position = newPosition;
    physicsObject->moveLowResGrabbed(newPosition, velocity);
    prevPosition = newPosition;
    time = deltaTime;
    isDragging_ = true;  // ★変更: 内部変数を使用
}


void GrabberGPUDuo::endGrab() {
    if (physicsObject) {
        physicsObject->endLowResGrab(hit_position, velocity);
    }
    isDragging_ = false;  // ★変更: 内部変数を使用
    std::cout << "isDragging: " << isDragging_ << std::endl;
}

void GrabberGPUDuo::update(float deltaTime) {
    time += deltaTime;
}

// =====================================
// スマートグラブ機能（グループ系から移植）
// =====================================

// スマートグラブ開始 - スフィア優先でチェック
void GrabberGPUDuo::startSmartGrab(double screenX, double screenY, float threshold) {
    if (!physicsObject) return;

    if (!pView || !pProjection || !pWindowWidth || !pWindowHeight) {
        std::cout << "ERROR: Global references not set!" << std::endl;
        return;
    }

    RayCastGPUDuo::Ray worldRay = RayCastGPUDuo::screenToRay(screenX, screenY, *pView, *pProjection,
                                                             glm::vec4(0, 0, *pWindowWidth, *pWindowHeight));

    // Step 1: まずハンドルスフィアをチェック（優先）
    float closestSphereT = std::numeric_limits<float>::max();
    int hitSphereIndex = -1;

    for (size_t i = 0; i < physicsObject->handleGroups.size(); i++) {
        float t;
        glm::vec3 sphereCenter = physicsObject->handleGroups[i].centerPosition;

        if (raySphereIntersect(worldRay.origin, worldRay.direction,
                               sphereCenter, threshold, t)) {
            if (t < closestSphereT) {
                closestSphereT = t;
                hitSphereIndex = static_cast<int>(i);
            }
        }
    }

    // スフィアにヒットした場合、直接インデックスでグラブ開始
    if (hitSphereIndex >= 0) {
        std::cout << "SmartGrab: Hit handle sphere " << hitSphereIndex << std::endl;

        glm::vec3 sphereCenter = physicsObject->handleGroups[hitSphereIndex].centerPosition;
        glm::vec3 clickPosition = worldRay.origin + worldRay.direction * closestSphereT;

        sphereGrabOffset = clickPosition - sphereCenter;  // オフセットを保存

        hit_position = sphereCenter;
        grabDistance = closestSphereT;  // 表面までの距離
        prevPosition = sphereCenter;
        velocity = glm::vec3(0.0f);
        time = 0.0f;

        physicsObject->startGrabHandleGroupByIndex(hitSphereIndex);
        isDragging_ = true;  // ★変更: 内部変数を使用
        isSmartGrabMode = true;

        std::cout << "-> Handle Group Grab activated (direct sphere hit)" << std::endl;
        return;
    }

    // Step 2: スフィアにヒットしなかった場合、メッシュとの交差判定
    RayCastGPUDuo::RayHit hit = RayCastGPUDuo::intersectMesh(worldRay, *physicsObject);

    std::cout << "SmartGrab Mesh Hit: " << (hit.hit ? "Yes" : "No") << std::endl;

    if (hit.hit) {
        hit_position = hit.position;
        grabDistance = glm::length(hit_position - worldRay.origin);
        prevPosition = hit_position;
        velocity = glm::vec3(0.0f);
        time = 0.0f;

        physicsObject->smartGrab(hit_position, threshold);
        isDragging_ = true;  // ★変更: 内部変数を使用
        isSmartGrabMode = true;

        if (physicsObject->currentGrabMode == SoftBodyGPUDuo::GRAB_HANDLE_GROUP) {
            std::cout << "-> Handle Group Grab activated" << std::endl;
        } else {
            std::cout << "-> Normal Grab activated" << std::endl;
        }
    }
}

// スマートグラブ移動
void GrabberGPUDuo::moveSmartGrab(double screenX, double screenY, float deltaTime) {
    if (!physicsObject || !isSmartGrabMode) {
        std::cout << "moveSmartGrab: No physics object or not in smart grab mode!" << std::endl;
        return;
    }

    RayCastGPUDuo::Ray worldRay = RayCastGPUDuo::screenToRay(screenX, screenY, *pView, *pProjection,
                                                             glm::vec4(0, 0, *pWindowWidth, *pWindowHeight));

    glm::vec3 newPosition = worldRay.origin + worldRay.direction * grabDistance;

    // ハンドルグループモードならオフセットを適用
    if (physicsObject->currentGrabMode == SoftBodyGPUDuo::GRAB_HANDLE_GROUP) {
        newPosition = newPosition - sphereGrabOffset;
    }

    if (time > 0.0f) {
        velocity = (newPosition - prevPosition) / time;
    }

    hit_position = newPosition;

    physicsObject->smartMove(newPosition, velocity);

    prevPosition = newPosition;
    time = deltaTime;
    isDragging_ = true;  // ★変更: 内部変数を使用
}

// スマートグラブ終了
void GrabberGPUDuo::endSmartGrab() {
    if (!physicsObject || !isSmartGrabMode) {
        std::cout << "endSmartGrab: No physics object or not in smart grab mode!" << std::endl;
        return;
    }

    // smartEndGrabを呼び出す
    physicsObject->smartEndGrab(hit_position, velocity);

    isDragging_ = false;  // ★変更: 内部変数を使用
    isSmartGrabMode = false;
    std::cout << "Smart grab ended, isDragging: " << isDragging_ << std::endl;
}

// レイ-スフィア交差判定
bool GrabberGPUDuo::raySphereIntersect(const glm::vec3& rayOrigin,
                                       const glm::vec3& rayDir,
                                       const glm::vec3& sphereCenter,
                                       float sphereRadius,
                                       float& t) {
    glm::vec3 oc = rayOrigin - sphereCenter;

    float a = glm::dot(rayDir, rayDir);
    float b = 2.0f * glm::dot(oc, rayDir);
    float c = glm::dot(oc, oc) - sphereRadius * sphereRadius;

    float discriminant = b * b - 4.0f * a * c;

    if (discriminant < 0) {
        return false;
    }

    float sqrtD = sqrt(discriminant);
    float t1 = (-b - sqrtD) / (2.0f * a);
    float t2 = (-b + sqrtD) / (2.0f * a);

    if (t1 > 0.001f) {
        t = t1;
        return true;
    } else if (t2 > 0.001f) {
        t = t2;
        return true;
    }

    return false;
}
