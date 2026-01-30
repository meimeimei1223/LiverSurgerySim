// SphereDeformGPUDuo.cpp
// 統合版：単一スフィア描画 + 複数スフィア管理

#include "SphereDeformGPUDuo.h"
#include "SoftBodyGPUDuo.h"
#include <algorithm>

//=============================================================================
// SphereDeformHandleGPUDuo 実装
//=============================================================================

void SphereDeformHandleGPUDuo::generate(float radius, int sectors, int stacks,
                                        const glm::vec3& center) {
    // 中心位置と半径を保存
    currentCenter = center;
    currentRadius = radius;

    vertices.clear();
    indices.clear();

    float x, y, z, xy;
    float nx, ny, nz, lengthInv = 1.0f / radius;
    float sectorStep = 2 * M_PI / sectors;
    float stackStep = M_PI / stacks;
    float sectorAngle, stackAngle;

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;

    // 頂点生成
    for(int i = 0; i <= stacks; ++i) {
        stackAngle = M_PI / 2 - i * stackStep;
        xy = radius * cosf(stackAngle);
        z = radius * sinf(stackAngle);

        for(int j = 0; j <= sectors; ++j) {
            sectorAngle = j * sectorStep;
            x = xy * cosf(sectorAngle);
            y = xy * sinf(sectorAngle);

            // centerを加えて頂点位置を決定
            positions.push_back(glm::vec3(x + center.x, y + center.y, z + center.z));

            // 法線（正規化済み）
            nx = x * lengthInv;
            ny = y * lengthInv;
            nz = z * lengthInv;
            normals.push_back(glm::vec3(nx, ny, nz));
        }
    }

    // インデックス生成
    int k1, k2;
    for(int i = 0; i < stacks; ++i) {
        k1 = i * (sectors + 1);
        k2 = k1 + sectors + 1;

        for(int j = 0; j < sectors; ++j, ++k1, ++k2) {
            if(i != 0) {
                indices.push_back(k1);
                indices.push_back(k2);
                indices.push_back(k1 + 1);
            }
            if(i != (stacks-1)) {
                indices.push_back(k1 + 1);
                indices.push_back(k2);
                indices.push_back(k2 + 1);
            }
        }
    }

    // インターリーブ形式に変換（位置3 + 法線3）
    vertices.clear();
    for(size_t i = 0; i < positions.size(); i++) {
        vertices.push_back(positions[i].x);
        vertices.push_back(positions[i].y);
        vertices.push_back(positions[i].z);
        vertices.push_back(normals[i].x);
        vertices.push_back(normals[i].y);
        vertices.push_back(normals[i].z);
    }
}

void SphereDeformHandleGPUDuo::setup() {
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GLfloat),
                 vertices.data(), GL_STATIC_DRAW);

    // 位置属性
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)0);
    glEnableVertexAttribArray(0);

    // 法線属性
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat),
                          (void*)(3 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint),
                 indices.data(), GL_STATIC_DRAW);

    glBindVertexArray(0);
}

void SphereDeformHandleGPUDuo::cleanup() {
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
}

void SphereDeformHandleGPUDuo::updatePosition(const glm::vec3& newCenter) {
    // 移動量を計算
    glm::vec3 delta = newCenter - currentCenter;

    // 全ての頂点を移動（インターリーブ形式：位置3 + 法線3）
    for (size_t i = 0; i < vertices.size(); i += 6) {
        vertices[i] += delta.x;      // x座標
        vertices[i + 1] += delta.y;  // y座標
        vertices[i + 2] += delta.z;  // z座標
        // 法線（i+3, i+4, i+5）は変更しない
    }

    // 現在の中心位置を更新
    currentCenter = newCenter;

    // VBOを更新
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(GLfloat),
                    vertices.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void SphereDeformHandleGPUDuo::draw(ShaderProgram& shader, const glm::vec3& color,
                                    const glm::mat4& view, const glm::mat4& projection,
                                    const glm::vec3& cameraPos) {
    // モデル行列は単位行列（頂点がすでに正しい位置にある）
    glm::mat4 modelMatrix = glm::mat4(1.0f);

    shader.use();
    shader.setUniform("model", modelMatrix);
    shader.setUniform("view", view);
    shader.setUniform("projection", projection);
    shader.setUniform("lightPos", cameraPos);
    shader.setUniform("viewPos", cameraPos);
    shader.setUniform("objectColor", color);
    shader.setUniform("useVertexColor", false);

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

//=============================================================================
// SphereHandleManager 実装
//=============================================================================

// ========== コンストラクタ・デストラクタ ==========

SphereHandleManager::SphereHandleManager()
    : handlePlaceMode_(false)
    , handleRadius_(DEFAULT_HANDLE_RADIUS)
    , minDistanceFactor_(2.0f)  // デフォルト: 半径の2倍
{
    resetColorSlots();
}

SphereHandleManager::~SphereHandleManager() {
    cleanup();
}

// ========== モード制御 ==========

void SphereHandleManager::startPlaceMode() {
    handlePlaceMode_ = true;
    std::cout << "[SphereHandleManager] Place mode started. "
              << "Current spheres: " << getSphereCount()
              << "/" << MAX_SPHERES << std::endl;
}

void SphereHandleManager::endPlaceMode() {
    handlePlaceMode_ = false;
    std::cout << "[SphereHandleManager] Place mode ended. "
              << "Total spheres: " << getSphereCount() << std::endl;
}

// ========== スフィア操作 ==========

bool SphereHandleManager::placeSphere(const glm::vec3& position, SoftBodyGPUDuo* softBody) {
    // 最大数チェック
    if (isFull()) {
        std::cout << "[SphereHandleManager] Cannot place: max spheres reached ("
                  << MAX_SPHERES << ")" << std::endl;
        return false;
    }

    // 距離チェック
    if (isTooCloseToExisting(position)) {
        float minDistance = handleRadius_ * minDistanceFactor_;
        std::cout << "[SphereHandleManager] Cannot place: too close to existing sphere "
                  << "(min distance: " << minDistance << ")" << std::endl;
        return false;
    }

    // 色スロット確保
    int colorId = allocateColorSlot();
    if (colorId < 0) {
        std::cout << "[SphereHandleManager] Cannot place: no color slot available" << std::endl;
        return false;
    }

    // スフィア作成
    SphereDeformHandleGPUDuo newSphere;
    newSphere.generate(handleRadius_, SPHERE_SECTORS, SPHERE_STACKS, position);
    newSphere.setup();

    // SoftBodyにhandleGroup作成（先に作成して成功を確認）
    if (softBody) {
        size_t groupCountBefore = softBody->handleGroups.size();

        softBody->createHandleGroupByRadius(position, handleRadius_);

        // handleGroupが追加されなかった場合
        if (softBody->handleGroups.size() == groupCountBefore) {
            std::cout << "[SphereHandleManager] Warning: HandleGroup was not created! "
                      << "Position may be outside mesh." << std::endl;
            newSphere.cleanup();
            releaseColorSlot(colorId);
            return false;
        }

        // handleGroupが空の場合はロールバック
        const auto& lastGroup = softBody->handleGroups.back();
        if (lastGroup.vertices.empty()) {
            std::cout << "[SphereHandleManager] Warning: No vertices captured! Rolling back." << std::endl;
            softBody->removeHandleGroup(softBody->handleGroups.size() - 1);
            newSphere.cleanup();
            releaseColorSlot(colorId);
            return false;
        }

        std::cout << "[SphereHandleManager] Handle group created with "
                  << lastGroup.vertices.size() << " vertices" << std::endl;
    }

    // データ追加（handleGroup作成成功後）
    spherePositions_.push_back(position);
    sphereMarkers_.push_back(newSphere);
    sphereColorIds_.push_back(colorId);

    std::cout << "[SphereHandleManager] Sphere placed at ("
              << position.x << ", " << position.y << ", " << position.z << ") "
              << "(" << getSphereCount() << "/" << MAX_SPHERES << ") "
              << "Color: " << colorId << std::endl;

    // 最大数に達したら配置モード終了
    if (isFull()) {
        endPlaceMode();
        std::cout << "[SphereHandleManager] All " << MAX_SPHERES << " spheres placed!" << std::endl;
    }

    return true;
}

void SphereHandleManager::removeLastSphere(SoftBodyGPUDuo* softBody) {
    if (spherePositions_.empty()) {
        std::cout << "[SphereHandleManager] No spheres to remove" << std::endl;
        return;
    }

    // SoftBodyからhandleGroup削除
    if (softBody && !softBody->handleGroups.empty()) {
        softBody->removeHandleGroup(softBody->handleGroups.size() - 1);
    }

    // 色スロット解放
    if (!sphereColorIds_.empty()) {
        releaseColorSlot(sphereColorIds_.back());
        sphereColorIds_.pop_back();
    }

    // スフィアデータ削除
    if (!sphereMarkers_.empty()) {
        sphereMarkers_.back().cleanup();
        sphereMarkers_.pop_back();
    }
    if (!spherePositions_.empty()) {
        spherePositions_.pop_back();
    }

    std::cout << "[SphereHandleManager] Last sphere removed. Remaining: "
              << getSphereCount() << std::endl;
}

void SphereHandleManager::removeSphereAt(int index, SoftBodyGPUDuo* softBody) {
    if (index < 0 || index >= static_cast<int>(spherePositions_.size())) {
        std::cout << "[SphereHandleManager] Invalid index: " << index << std::endl;
        return;
    }

    // SoftBodyからhandleGroup削除
    if (softBody && index < static_cast<int>(softBody->handleGroups.size())) {
        softBody->removeHandleGroup(index);
    }

    // 色スロット解放
    if (index < static_cast<int>(sphereColorIds_.size())) {
        releaseColorSlot(sphereColorIds_[index]);
        sphereColorIds_.erase(sphereColorIds_.begin() + index);
    }

    // スフィアデータ削除
    if (index < static_cast<int>(sphereMarkers_.size())) {
        sphereMarkers_[index].cleanup();
        sphereMarkers_.erase(sphereMarkers_.begin() + index);
    }
    if (index < static_cast<int>(spherePositions_.size())) {
        spherePositions_.erase(spherePositions_.begin() + index);
    }

    std::cout << "[SphereHandleManager] Sphere at index " << index
              << " removed. Remaining: " << getSphereCount() << std::endl;
}

void SphereHandleManager::clearAll(SoftBodyGPUDuo* softBody) {
    // SoftBodyの全handleGroupsを削除
    if (softBody) {
        softBody->clearHandleGroups();
    }

    // 全スフィアのOpenGLリソース解放
    for (auto& marker : sphereMarkers_) {
        marker.cleanup();
    }

    // データクリア
    spherePositions_.clear();
    sphereMarkers_.clear();
    sphereColorIds_.clear();
    sphereVertexIndices_.clear();

    // 色スロットリセット
    resetColorSlots();

    // 配置モードに戻す
    handlePlaceMode_ = true;

    std::cout << "[SphereHandleManager] All spheres cleared" << std::endl;
}

// ========== 位置更新 ==========

void SphereHandleManager::updateSpherePosition(int index, const glm::vec3& newPosition) {
    if (index < 0 || index >= static_cast<int>(spherePositions_.size())) {
        return;
    }

    spherePositions_[index] = newPosition;
    if (index < static_cast<int>(sphereMarkers_.size())) {
        sphereMarkers_[index].updatePosition(newPosition);
    }
}

void SphereHandleManager::syncPositionsFromSoftBody(SoftBodyGPUDuo* softBody) {
    if (!softBody) return;

    size_t count = std::min(sphereMarkers_.size(), softBody->handleGroups.size());
    for (size_t i = 0; i < count; i++) {
        // HandleGroupの中心位置を取得
        glm::vec3 groupCenter = softBody->handleGroups[i].centerPosition;
        sphereMarkers_[i].updatePosition(groupCenter);
        spherePositions_[i] = groupCenter;
    }
}

int SphereHandleManager::syncWithSoftBodyAfterCut(SoftBodyGPUDuo* softBody) {
    if (!softBody) return 0;

    int removedCount = 0;
    size_t softBodyGroupCount = softBody->handleGroups.size();

    // SoftBody側のhandleGroupsが減っている場合、末尾から削除
    while (spherePositions_.size() > softBodyGroupCount) {
        // 色スロット解放
        if (!sphereColorIds_.empty()) {
            releaseColorSlot(sphereColorIds_.back());
            sphereColorIds_.pop_back();
        }

        // スフィアデータ削除
        if (!sphereMarkers_.empty()) {
            sphereMarkers_.back().cleanup();
            sphereMarkers_.pop_back();
        }
        if (!spherePositions_.empty()) {
            spherePositions_.pop_back();
        }

        removedCount++;
    }

    if (removedCount > 0) {
        std::cout << "[SphereHandleManager] Synced after cut: removed "
                  << removedCount << " sphere(s). Remaining: "
                  << getSphereCount() << std::endl;

        // 配置モードに戻す（スフィアがなくなった場合）
        if (spherePositions_.empty()) {
            handlePlaceMode_ = true;
        }
    }

    // 残ったスフィアの位置を同期
    syncPositionsFromSoftBody(softBody);

    return removedCount;
}

// ========== 描画 ==========

void SphereHandleManager::drawAll(ShaderProgram& shader,
                                  const glm::mat4& view,
                                  const glm::mat4& projection,
                                  const glm::vec3& cameraPos) {
    for (size_t i = 0; i < sphereMarkers_.size(); i++) {
        int colorId = (i < sphereColorIds_.size()) ? sphereColorIds_[i] : 0;
        glm::vec3 color = getPointColorGPUDuo(colorId, true);
        sphereMarkers_[i].draw(shader, color, view, projection, cameraPos);
    }
}

// ========== クリーンアップ ==========

void SphereHandleManager::cleanup() {
    for (auto& marker : sphereMarkers_) {
        marker.cleanup();
    }
    sphereMarkers_.clear();
    spherePositions_.clear();
    sphereColorIds_.clear();
    sphereVertexIndices_.clear();
    resetColorSlots();
}

// ========== ゲッター ==========

glm::vec3 SphereHandleManager::getPosition(int index) const {
    if (index < 0 || index >= static_cast<int>(spherePositions_.size())) {
        return glm::vec3(0.0f);
    }
    return spherePositions_[index];
}

int SphereHandleManager::getColorId(int index) const {
    if (index < 0 || index >= static_cast<int>(sphereColorIds_.size())) {
        return -1;
    }
    return sphereColorIds_[index];
}

glm::vec3 SphereHandleManager::getColor(int index, bool isBright) const {
    int colorId = getColorId(index);
    if (colorId < 0) {
        return glm::vec3(1.0f);  // デフォルト白
    }
    return getPointColorGPUDuo(colorId, isBright);
}

// ========== ユーティリティ ==========

bool SphereHandleManager::isTooCloseToExisting(const glm::vec3& position) const {
    float minDistance = handleRadius_ * minDistanceFactor_;
    for (const auto& existingPos : spherePositions_) {
        float dist = glm::length(position - existingPos);
        if (dist < minDistance) {
            return true;
        }
    }
    return false;
}

void SphereHandleManager::printDebugInfo() const {
    std::cout << "=== SphereHandleManager Debug Info ===" << std::endl;
    std::cout << "  Place mode: " << (handlePlaceMode_ ? "ON" : "OFF") << std::endl;
    std::cout << "  Sphere count: " << getSphereCount() << "/" << MAX_SPHERES << std::endl;
    std::cout << "  Handle radius: " << handleRadius_ << std::endl;
    std::cout << "  Min distance factor: " << minDistanceFactor_ << std::endl;

    for (size_t i = 0; i < spherePositions_.size(); i++) {
        const auto& pos = spherePositions_[i];
        int colorId = (i < sphereColorIds_.size()) ? sphereColorIds_[i] : -1;
        std::cout << "  Sphere[" << i << "]: pos=("
                  << pos.x << ", " << pos.y << ", " << pos.z
                  << ") colorId=" << colorId << std::endl;
    }

    std::cout << "  Color slots used: ";
    for (int i = 0; i < MAX_SPHERES; i++) {
        std::cout << (colorSlotUsed_[i] ? "1" : "0");
    }
    std::cout << std::endl;
}

// ========== 色スロット管理（プライベート） ==========

int SphereHandleManager::allocateColorSlot() {
    for (int i = 0; i < MAX_SPHERES; i++) {
        if (!colorSlotUsed_[i]) {
            colorSlotUsed_[i] = true;
            return i;
        }
    }
    return -1;
}

void SphereHandleManager::releaseColorSlot(int slot) {
    if (slot >= 0 && slot < MAX_SPHERES) {
        colorSlotUsed_[slot] = false;
    }
}

void SphereHandleManager::resetColorSlots() {
    colorSlotUsed_.fill(false);
}
