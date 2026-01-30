#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GL/glew.h>
#include <vector>
#include <limits>
#include <algorithm>
#include <iostream>

class SoftBodyGPUDuo;

// ★修正: class → struct に変更（mCutMesh.h と一致させる）
struct mCutMesh;

// ========================================
// マルチメッシュヒット情報の名前空間
// （既存の互換性のため残す）
// ========================================
namespace MultiMeshHitTetML {
extern int hit_mesh_index;
extern int hit_layer_index;
extern bool is_dragging;
extern bool is_smooth_mesh;
extern float closest_distance;
extern glm::vec3 hit_position;
}

// ========================================
// カッターメッシュの状態管理クラス
// ========================================
class CutterMeshState {
public:
    // --- ヒット対象の列挙型 ---
    enum HitTarget {
        NONE,
        LIVER,
        PORTAL,
        VEIN,
        CUTTER  // カッターメッシュ自体
    };

    // --- ヒット状態構造体 ---
    struct HitState {
        bool isDragging = false;
        HitTarget target = NONE;
        SoftBodyGPUDuo* body = nullptr;
        glm::vec3 position;

        // 便利なクエリメソッド
        bool isLiverHit() const { return target == LIVER; }
        bool isPortalHit() const { return target == PORTAL; }
        bool isVeinHit() const { return target == VEIN; }
        bool isCutterHit() const { return target == CUTTER; }
        bool isVesselHit() const { return target == PORTAL || target == VEIN; }
        bool hasHit() const { return target != NONE; }

        void reset() {
            isDragging = false;
            target = NONE;
            body = nullptr;
        }


        // ★★★ 追加 ★★★
        // ドラッグ終了のみ（target, body, position は維持 → 色が変わらない）
        void endDrag() {
            isDragging = false;
        }
    };

    // --- ターゲット別スケール管理構造体 ---
    struct TargetScales {
        float liver = 1.0f;
        float vessel = 0.5f;
        float defaultScale = 1.0f;

        float getScaleForTarget(HitTarget target) const {
            switch (target) {
            case LIVER: return liver;
            case PORTAL:
            case VEIN: return vessel;
            default: return defaultScale;
            }
        }

        void setScaleForTarget(HitTarget target, float scale) {
            switch (target) {
            case LIVER: liver = scale; break;
            case PORTAL:
            case VEIN: vessel = scale; break;
            default: defaultScale = scale; break;
            }
        }
    };

    // --- メンバー変数（公開） ---
    HitState hitState;
    TargetScales targetScales;

public:
    CutterMeshState();

    // --- 初期化 ---
    void initialize(const mCutMesh& mesh);

    // --- スケール・変換適用 ---
    void applyScale(mCutMesh& mesh, float targetScale);

    // --- 回転・移動更新 ---
    void updateRotation(const glm::mat4& deltaRotation);
    void updateTranslation(const glm::vec3& deltaTranslation);

    // --- 中心位置の再計算 ---
    void calculateCenter();

    // --- リセット ---
    void reset();

    // --- ゲッター ---
    float getCurrentScale() const { return currentScale; }
    const glm::vec3& getCenter() const { return center; }
    const glm::vec3& getTranslation() const { return translation; }
    const glm::mat4& getRotationMatrix() const { return rotationMatrix; }

    // ========================================
    // ヒット判定関連
    // ========================================

    // カッターメッシュへのレイキャスト（旧FindHit関数の統合）
    bool findCutterHit(
        float screenX, float screenY,
        const glm::mat4& view, const glm::mat4& projection,
        int windowWidth, int windowHeight,
        const mCutMesh& mesh);

    // SoftBodyへのヒット結果をセット
    void setBodyHit(SoftBodyGPUDuo* body, HitTarget target, const glm::vec3& pos);

    // ヒット状態リセット
    void resetHit() { hitState.reset(); }

    // 現在のターゲットに応じたスケール取得
    float getCurrentTargetScale() const {
        return targetScales.getScaleForTarget(hitState.target);
    }

    // 現在のターゲットのスケール設定
    void setCurrentTargetScale(float scale) {
        targetScales.setScaleForTarget(hitState.target, scale);
    }

    // カッターの色を取得（ヒット対象による）
    glm::vec3 getCutterColor() const {
        switch (hitState.target) {
        case PORTAL: return glm::vec3(0.8f, 0.2f, 0.8f);  // マゼンタ
        case VEIN:   return glm::vec3(0.2f, 0.8f, 0.8f);  // シアン
        case LIVER:  return glm::vec3(0.8f, 0.8f, 0.0f);  // 黄色
        default:     return glm::vec3(0.2f, 0.8f, 0.2f);  // デフォルト緑
        }
    }

private:
    std::vector<GLfloat> originalVertices;  // オリジナルの頂点データ
    glm::vec3 center;                       // メッシュの中心
    float currentScale;                     // 現在のスケール値
    glm::mat4 rotationMatrix;               // 累積回転行列
    glm::vec3 translation;                  // 累積移動量
};

// ========================================
