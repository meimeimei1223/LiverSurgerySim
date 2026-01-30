// SphereDeformGPUDuo.h
// 統合版：単一スフィア描画 + 複数スフィア管理
#ifndef SPHEREDEFORMGPUDUO_H
#define SPHEREDEFORMGPUDUO_H

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <array>
#include <cmath>
#include <iostream>
#include "ShaderProgram.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 前方宣言
class SoftBodyGPUDuo;

//=============================================================================
// SphereDeformHandleGPUDuo - 単一スフィアの描画・更新
//=============================================================================
class SphereDeformHandleGPUDuo {
public:
    GLuint VAO, VBO, EBO;
    std::vector<GLfloat> vertices;  // インターリーブ形式：位置(3) + 法線(3)
    std::vector<GLuint> indices;

    // centerを中心にスフィアを生成
    void generate(float radius, int sectors, int stacks,
                  const glm::vec3& center = glm::vec3(0.0f));
    void setup();
    void draw(ShaderProgram& shader, const glm::vec3& color,
              const glm::mat4& view, const glm::mat4& projection,
              const glm::vec3& cameraPos);
    void cleanup();

    // スフィアの位置を更新
    void updatePosition(const glm::vec3& newCenter);

    // ゲッター
    glm::vec3 getCenter() const { return currentCenter; }
    float getRadius() const { return currentRadius; }

private:
    glm::vec3 currentCenter;  // 現在の中心位置を保持
    float currentRadius;      // 現在の半径を保持
};

//=============================================================================
// 色取得関数
//=============================================================================
inline glm::vec3 getPointColorGPUDuo(int index, bool isBright) {
    static const std::vector<glm::vec3> baseColors = {
        glm::vec3(1.0f, 0.0f, 0.0f),   // 赤
        glm::vec3(0.0f, 1.0f, 0.0f),   // 緑
        glm::vec3(0.0f, 0.0f, 1.0f),   // 青
        glm::vec3(1.0f, 1.0f, 0.0f),   // 黄
        glm::vec3(1.0f, 0.0f, 1.0f),   // マゼンタ
        glm::vec3(0.0f, 1.0f, 1.0f)    // シアン
    };

    int safeIndex = index % static_cast<int>(baseColors.size());
    if (safeIndex < 0) safeIndex = 0;

    glm::vec3 color = baseColors[safeIndex];
    return isBright ? color : color * 0.7f;
}

//=============================================================================
// SphereHandleManager - 複数スフィアの管理
//=============================================================================
/**
 * @brief ソフトボディのハンドルスフィアを管理するクラス
 *
 * 機能:
 * - スフィアの配置・削除
 * - 色スロットの管理（固定5色）
 * - ハンドル配置モードの制御
 * - スフィア位置の更新・描画
 * - SoftBodyGPUDuoとの連携
 */
class SphereHandleManager {
public:
    // 定数
    static constexpr int MAX_SPHERES = 5;
    static constexpr float DEFAULT_HANDLE_RADIUS = 0.4f;
    static constexpr int SPHERE_SECTORS = 16;
    static constexpr int SPHERE_STACKS = 16;

    // コンストラクタ・デストラクタ
    SphereHandleManager();
    ~SphereHandleManager();

    // ========== モード制御 ==========
    /**
     * @brief ハンドル配置モードを開始
     */
    void startPlaceMode();

    /**
     * @brief ハンドル配置モードを終了
     */
    void endPlaceMode();

    /**
     * @brief 配置モード中かどうか
     */
    bool isPlaceMode() const { return handlePlaceMode_; }

    // ========== スフィア操作 ==========
    /**
     * @brief 新しいスフィアを配置
     * @param position 配置位置
     * @param softBody 関連するソフトボディ（handleGroup作成用、nullptrでも可）
     * @return 成功時true、失敗時（距離制限・最大数超過等）false
     */
    bool placeSphere(const glm::vec3& position, SoftBodyGPUDuo* softBody = nullptr);

    /**
     * @brief 最後に配置したスフィアを削除
     * @param softBody 関連するソフトボディ
     */
    void removeLastSphere(SoftBodyGPUDuo* softBody = nullptr);

    /**
     * @brief 指定インデックスのスフィアを削除
     * @param index 削除するスフィアのインデックス
     * @param softBody 関連するソフトボディ
     */
    void removeSphereAt(int index, SoftBodyGPUDuo* softBody = nullptr);

    /**
     * @brief 全スフィアをクリア
     * @param softBody 関連するソフトボディ
     */
    void clearAll(SoftBodyGPUDuo* softBody = nullptr);

    // ========== 位置更新 ==========
    /**
     * @brief 指定インデックスのスフィア位置を更新
     * @param index スフィアインデックス
     * @param newPosition 新しい位置
     */
    void updateSpherePosition(int index, const glm::vec3& newPosition);

    /**
     * @brief SoftBodyのhandleGroupsからスフィア位置を同期
     * @param softBody ソフトボディ
     */
    void syncPositionsFromSoftBody(SoftBodyGPUDuo* softBody);

    /**
     * @brief カット後にSoftBodyのhandleGroupsと同期（削除されたグループを反映）
     * @param softBody ソフトボディ
     * @return 削除されたスフィアの数
     *
     * カット操作でhandleGroupが削除された場合、対応するスフィアも削除します。
     * 毎フレームまたはカット後に呼び出してください。
     */
    int syncWithSoftBodyAfterCut(SoftBodyGPUDuo* softBody);

    // ========== 描画 ==========
    /**
     * @brief 全スフィアを描画
     * @param shader シェーダープログラム
     * @param view ビュー行列
     * @param projection プロジェクション行列
     * @param cameraPos カメラ位置
     */
    void drawAll(ShaderProgram& shader,
                 const glm::mat4& view,
                 const glm::mat4& projection,
                 const glm::vec3& cameraPos);

    // ========== クリーンアップ ==========
    /**
     * @brief OpenGLリソースを解放
     */
    void cleanup();

    // ========== ゲッター ==========
    int getSphereCount() const { return static_cast<int>(spherePositions_.size()); }
    int getMaxSpheres() const { return MAX_SPHERES; }
    float getHandleRadius() const { return handleRadius_; }
    bool isFull() const { return spherePositions_.size() >= MAX_SPHERES; }

    const std::vector<glm::vec3>& getPositions() const { return spherePositions_; }
    const std::vector<int>& getColorIds() const { return sphereColorIds_; }

    /**
     * @brief 指定インデックスのスフィア位置を取得
     */
    glm::vec3 getPosition(int index) const;

    /**
     * @brief 指定インデックスの色IDを取得
     */
    int getColorId(int index) const;

    /**
     * @brief 指定インデックスの色を取得
     */
    glm::vec3 getColor(int index, bool isBright = true) const;

    // ========== セッター ==========
    void setHandleRadius(float radius) { handleRadius_ = radius; }
    void setMinDistanceFactor(float factor) { minDistanceFactor_ = factor; }

    // ========== ユーティリティ ==========
    /**
     * @brief 指定位置が既存スフィアに近すぎるかチェック
     * @param position チェックする位置
     * @return 近すぎる場合true
     */
    bool isTooCloseToExisting(const glm::vec3& position) const;

    /**
     * @brief デバッグ情報を出力
     */
    void printDebugInfo() const;

private:
    // ========== 色スロット管理 ==========
    int allocateColorSlot();
    void releaseColorSlot(int slot);
    void resetColorSlots();

    // ========== メンバ変数 ==========
    // スフィアデータ
    std::vector<glm::vec3> spherePositions_;
    std::vector<SphereDeformHandleGPUDuo> sphereMarkers_;
    std::vector<int> sphereColorIds_;
    std::vector<int> sphereVertexIndices_;  // 頂点インデックス（将来使用）

    // 色スロット管理
    std::array<bool, MAX_SPHERES> colorSlotUsed_;

    // 状態
    bool handlePlaceMode_;
    float handleRadius_;
    float minDistanceFactor_;  // handleRadius_ * この値が最小距離




};

#endif // SPHEREDEFORMGPUDUO_H
