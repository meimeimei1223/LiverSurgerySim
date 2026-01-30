// GrabberGPUDuo.h
#ifndef GRABBERGPUDUO_H
#define GRABBERGPUDUO_H
#include <glm/glm.hpp>
#include "SoftBodyGPUDuo.h"
#include "RayCastGPUDuo.h"

class GrabberGPUDuo {
public:
    GrabberGPUDuo();
    void setPhysicsObject(SoftBodyGPUDuo* object);

    // ★変更: isDraggingポインタを削除
    void setGlobalRefs(glm::mat4* view, glm::mat4* projection,
                       int* windowWidth, int* windowHeight);

    void startGrab(float screenX, float screenY);
    void moveGrab(float screenX, float screenY, float deltaTime);
    void endGrab();
    void update(float deltaTime);

    // ヒット位置（publicに変更）
    glm::vec3 hit_position;

    // =====================================
    // ★追加: isDragging状態のゲッター
    // =====================================
    bool isDragging() const { return isDragging_; }

    // ★追加: 外部からドラッグ状態を強制リセット（モード切替時用）
    void forceEndDrag();

    // =====================================
    // スマートグラブ機能（グループ系から移植）
    // =====================================
    void startSmartGrab(double screenX, double screenY, float threshold = 0.5f);
    void moveSmartGrab(double screenX, double screenY, float dt);
    void endSmartGrab();

private:
    SoftBodyGPUDuo* physicsObject;
    float grabDistance;
    glm::vec3 prevPosition;
    glm::vec3 velocity;
    float time;
    glm::mat4* pView;
    glm::mat4* pProjection;
    int* pWindowWidth;
    int* pWindowHeight;

    // ★変更: ポインタから内部変数に変更
    bool isDragging_ = false;

    // スマートグラブ用
    bool isSmartGrabMode = false;
    glm::vec3 sphereGrabOffset;  // スフィアクリック時のオフセット

    // レイ-スフィア交差判定
    bool raySphereIntersect(const glm::vec3& rayOrigin,
                            const glm::vec3& rayDir,
                            const glm::vec3& sphereCenter,
                            float sphereRadius,
                            float& t);
};
#endif
