// CutSegmentManager.h
#pragma once

#include <glm/glm.hpp>
#include <set>
#include <vector>
#include <string>

class SoftBodyGPUDuo;
namespace VoxelSkeleton {
class VesselSegmentation;
}

class CutSegmentManager {
public:
    enum class CutMode {
        None,
        Skeleton,
        Voronoi3D
    };

    enum class ClickMode {
        None,
        Liver,
        Portal
    };

    struct SegmentInfo {
        bool valid = false;
        glm::vec3 hitPosition;
        int nearestVertex = -1;
        int hitTriIndex = -1;
        std::set<int> selectedSegments;
        std::vector<int> selectedBranches;
        int objSegmentId = -1;
        bool isLiverHit = false;
        bool isPortalHit = false;
    };

    CutSegmentManager() = default;

    // 色をメンバー変数として追加
    glm::vec4 liverBaseColor_ = glm::vec4(0.8f, 0.2f, 0.2f, 0.8f);
    glm::vec4 liverHighlightColor_ = glm::vec4(0.6f, 1.0f, 0.2f, 0.9f); // 黄緑
    glm::vec4 portalBaseColor_ = glm::vec4(0.8f, 0.2f, 0.8f, 0.8f);
    glm::vec4 portalHighlightColor_ = glm::vec4(1.0f, 0.5f, 0.0f, 0.9f);  // オレンジg

    // 新しいメソッド
    void applyBaseColorOverlay();  // 全体をベースカラーで塗る
    // CutSegmentManager.h
    void printSelectedLiverVolumeInfo();

    // メッシュ参照の設定
    void setLiver(SoftBodyGPUDuo* liver) { liver_ = liver; }
    void setPortal(SoftBodyGPUDuo* portal) { portal_ = portal; }
    void setSkeleton(VoxelSkeleton::VesselSegmentation* skeleton) { skeleton_ = skeleton; }

    // manualExtendMode 関連
    void setManualExtendMode(bool enabled) { manualExtendMode_ = enabled; }
    bool isManualExtendMode() const { return manualExtendMode_; }
    void setExtensionNodeCount(int count) { extensionNodeCount_ = count; }
    int getExtensionNodeCount() const { return extensionNodeCount_; }

    // カッターモード管理
    void setCutMode(CutMode mode) { cutMode_ = mode; }
    CutMode getCutMode() const { return cutMode_; }
    void cycleCutMode();
    std::string getCutModeName() const;
    bool isCutModeActive() const { return cutMode_ != CutMode::None; }

    // クリック選択モード管理
    void setClickMode(ClickMode mode);
    ClickMode getClickMode() const { return clickMode_; }
    void toggleLiverSelectMode();
    void togglePortalSelectMode();
    bool isLiverSelectMode() const { return clickMode_ == ClickMode::Liver; }
    bool isPortalSelectMode() const { return clickMode_ == ClickMode::Portal; }

    // カッターヒット処理
    void updateFromCutterHit(const glm::vec3& hitPosition, bool isLiver, bool isPortal);
    void applyCutOverlay();
    void resetCutOverlay();

    // クリック処理
    void handleClick(float mouseX, float mouseY, bool shiftPressed,
                     const glm::mat4& view, const glm::mat4& projection,
                     const glm::vec3& cameraPos,
                     int windowWidth, int windowHeight);

    const SegmentInfo& getInfo() const { return info_; }

private:
    int findNearestSmoothSurfaceVertex(SoftBodyGPUDuo* mesh, const glm::vec3& position);

    int raycastToVertex(SoftBodyGPUDuo* mesh,
                        float mouseX, float mouseY,
                        const glm::mat4& view, const glm::mat4& projection,
                        const glm::vec3& cameraPos,
                        int windowWidth, int windowHeight,
                        glm::vec3& outHitPos, int& outTriIndex);

    void processLiverClick(int hitVertex, const glm::vec3& hitPos, int hitTriIndex, bool shiftPressed);
    void processPortalClick(int hitVertex, const glm::vec3& hitPos, int hitTriIndex, bool shiftPressed);

    CutMode cutMode_ = CutMode::None;
    ClickMode clickMode_ = ClickMode::None;
    SegmentInfo info_;

    SoftBodyGPUDuo* liver_ = nullptr;
    SoftBodyGPUDuo* portal_ = nullptr;
    VoxelSkeleton::VesselSegmentation* skeleton_ = nullptr;

    bool manualExtendMode_ = false;
    int extensionNodeCount_ = 5;

    glm::vec4 baseColor_ = glm::vec4(0.8f, 0.3f, 0.3f, 0.8f);
    glm::vec4 highlightColor_ = glm::vec4(1.0f, 0.85f, 0.2f, 0.9f);
};


