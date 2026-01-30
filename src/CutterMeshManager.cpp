#include "CutterMeshManager.h"
#include "mCutMesh.h"       // mCutMesh構造体の完全な定義
#include "RayCastGPUDuo.h"  // Ray, RayHitTri, screenToRay, intersectMesh

// ========================================
// MultiMeshHitTetML 名前空間の変数定義
// （既存の互換性のため残す）
// ========================================
namespace MultiMeshHitTetML {
int hit_mesh_index = -1;
int hit_layer_index = -1;
bool is_dragging = false;
bool is_smooth_mesh = false;
float closest_distance = std::numeric_limits<float>::max();
glm::vec3 hit_position;
}

// ========================================
// CutterMeshState クラスの実装
// ========================================

CutterMeshState::CutterMeshState()
    : center(0.0f)
    , currentScale(1.0f)
    , rotationMatrix(1.0f)
    , translation(0.0f) {
}

void CutterMeshState::initialize(const mCutMesh& mesh) {
    originalVertices = mesh.mVertices;
    calculateCenter();
    reset();

    // MultiMeshHitTetMLの状態もクリア
    MultiMeshHitTetML::hit_mesh_index = -1;
    MultiMeshHitTetML::hit_layer_index = -1;
    MultiMeshHitTetML::is_dragging = false;
    MultiMeshHitTetML::closest_distance = std::numeric_limits<float>::max();
    MultiMeshHitTetML::hit_position = glm::vec3(0.0f);

    // ヒット状態もクリア
    hitState.reset();
}

void CutterMeshState::calculateCenter() {
    center = glm::vec3(0.0f);
    if (originalVertices.empty()) return;

    for (size_t i = 0; i < originalVertices.size(); i += 3) {
        center.x += originalVertices[i];
        center.y += originalVertices[i + 1];
        center.z += originalVertices[i + 2];
    }
    center /= (originalVertices.size() / 3);
}

void CutterMeshState::applyScale(mCutMesh& mesh, float targetScale) {
    if (originalVertices.empty()) return;

    // スケール変更を適用
    for (size_t i = 0; i < originalVertices.size(); i += 3) {
        glm::vec3 vertex(originalVertices[i],
                         originalVertices[i + 1],
                         originalVertices[i + 2]);

        // 中心からの相対位置
        glm::vec3 relative = vertex - center;

        // スケールを適用
        relative *= targetScale;

        // 回転を適用
        glm::vec4 rotated = rotationMatrix * glm::vec4(relative, 1.0f);

        // 中心に戻して移動を適用
        glm::vec3 final = glm::vec3(rotated) + center + translation;

        mesh.mVertices[i] = final.x;
        mesh.mVertices[i + 1] = final.y;
        mesh.mVertices[i + 2] = final.z;
    }
    currentScale = targetScale;
}

void CutterMeshState::updateRotation(const glm::mat4& deltaRotation) {
    rotationMatrix = deltaRotation * rotationMatrix;
}

void CutterMeshState::updateTranslation(const glm::vec3& deltaTranslation) {
    translation += deltaTranslation;
}

void CutterMeshState::reset() {
    currentScale = 1.0f;
    rotationMatrix = glm::mat4(1.0f);
    translation = glm::vec3(0.0f);
}

// ========================================
// ヒット判定関連の実装
// ========================================

bool CutterMeshState::findCutterHit(
    float screenX, float screenY,
    const glm::mat4& view, const glm::mat4& projection,
    int windowWidth, int windowHeight,
    const mCutMesh& mesh)
{
    // スクリーン座標からワールド空間のレイを生成
    RayCastGPUDuo::Ray worldRay = RayCastGPUDuo::screenToRay(
        screenX, screenY, view, projection,
        glm::vec4(0, 0, windowWidth, windowHeight));

    // カッターメッシュとの交差判定
    RayCastGPUDuo::RayHitTri hit = RayCastGPUDuo::intersectMesh(
        worldRay, mesh.mVertices, mesh.mIndices);

    if (hit.hit) {
        hitState.position = hit.position;
        hitState.target = CUTTER;
        hitState.body = nullptr;
        hitState.isDragging = true;
        return true;
    }

    hitState.reset();
    return false;
}

void CutterMeshState::setBodyHit(SoftBodyGPUDuo* body, HitTarget target, const glm::vec3& pos) {
    hitState.body = body;
    hitState.target = target;
    hitState.position = pos;
    hitState.isDragging = true;
}
