// VoronoiPathSegmentation.h
// WASM/Emscripten & Native 両対応版
//
// プラットフォーム自動検出:
//   - __EMSCRIPTEN__ 定義時 → GEOGRAM無効
//   - ネイティブ → __has_include または CMake設定で判定

#ifndef VORONOI_PATH_SEGMENTATION_H
#define VORONOI_PATH_SEGMENTATION_H

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>

//=============================================================================
// プラットフォーム検出 & GEOGRAM可否判定
//=============================================================================

#ifdef __EMSCRIPTEN__
// WebAssembly環境: GEOGRAMは使用しない
#define PLATFORM_WASM 1
#define GEOGRAM_AVAILABLE 0
#else
// ネイティブ環境
#define PLATFORM_WASM 0

#ifndef GEOGRAM_AVAILABLE
// CMakeで設定されていない場合は自動検出
#if defined(__has_include)
#if __has_include(<geogram/basic/common.h>)
#define GEOGRAM_AVAILABLE 1
#else
#define GEOGRAM_AVAILABLE 0
#endif
#else
// __has_include 非対応の場合はデフォルトOFF
#define GEOGRAM_AVAILABLE 0
#endif
#endif
#endif

// GEOGRAMヘッダー（利用可能な場合のみ）
#if GEOGRAM_AVAILABLE
#include <geogram/basic/common.h>
#include <geogram/delaunay/delaunay.h>
#endif

namespace VoxelSkeleton {

//=============================================================================
// セグメンテーションモード列挙型
//=============================================================================
enum class SegmentationMode {
    OBJ,              // OBJファイルベース（S1-S8）
    SkeletonDistance, // スケルトンベース（単純距離計算）
    Voronoi3D         // 3D Voronoi（GEOGRAM必須）
};

/// モード名を文字列で取得
inline const char* getSegmentationModeName(SegmentationMode mode) {
    switch (mode) {
    case SegmentationMode::OBJ:              return "OBJ (S1-S8)";
    case SegmentationMode::SkeletonDistance: return "Skeleton (Distance)";
    case SegmentationMode::Voronoi3D:
#if GEOGRAM_AVAILABLE
        return "Voronoi 3D (Path)";
#else
        return "Voronoi 3D (Disabled)";
#endif
    default:                                 return "Unknown";
    }
}

//=============================================================================
// ユーティリティ関数
//=============================================================================

/// プラットフォーム名を取得
inline const char* getPlatformName() {
#if PLATFORM_WASM
    return "WebAssembly";
#else
    return "Native";
#endif
}

/// Voronoi3Dモードが利用可能かどうか
inline bool isVoronoi3DAvailable() {
#if GEOGRAM_AVAILABLE
    return true;
#else
    return false;
#endif
}

/// 利用可能なモードのリストを取得
inline std::vector<SegmentationMode> getAvailableModes() {
    std::vector<SegmentationMode> modes;
    modes.push_back(SegmentationMode::OBJ);
    modes.push_back(SegmentationMode::SkeletonDistance);
#if GEOGRAM_AVAILABLE
    modes.push_back(SegmentationMode::Voronoi3D);
#endif
    return modes;
}

/// 利用可能なモード数
inline int getAvailableModeCount() {
#if GEOGRAM_AVAILABLE
    return 3;
#else
    return 2;
#endif
}

//=============================================================================
// BranchPath
// ルートから末端ノードまでの1本のパス（枝）
//=============================================================================
struct BranchPath {
    int branchId;                         // ブランチID
    int terminalSegmentId;                // 末端セグメントID
    std::vector<int> segmentIds;          // パスに含まれるセグメントID
    std::vector<glm::vec3> pathPoints;    // パスを構成する点
    glm::vec3 color;                      // 表示用の色

    BranchPath() : branchId(-1), terminalSegmentId(-1), color(0.5f) {}

    float getTotalLength() const {
        float length = 0.0f;
        for (size_t i = 1; i < pathPoints.size(); ++i) {
            length += glm::length(pathPoints[i] - pathPoints[i - 1]);
        }
        return length;
    }
};

//=============================================================================
// VoronoiPathSegmenter
//
// GEOGRAM_AVAILABLE=1: フル機能
// GEOGRAM_AVAILABLE=0: スタブ（すべて無効値を返す）
//=============================================================================
class VoronoiPathSegmenter {
public:
    VoronoiPathSegmenter();
    ~VoronoiPathSegmenter();

    /// GEOGRAMが利用可能かどうか
    static bool isGeogramAvailable() {
#if GEOGRAM_AVAILABLE
        return true;
#else
        return false;
#endif
    }

    //=========================================================================
    // 初期化・構築
    //=========================================================================

    template<typename NodeType, typename SegmentType>
    bool build(const std::vector<NodeType>& nodes,
               const std::vector<SegmentType>& segments,
               int rootSegmentId,
               float samplingInterval = 0.5f);

    bool buildFromNodes(const std::vector<glm::vec3>& nodePositions,
                        const std::vector<int>& nodeToSegmentId);

    //=========================================================================
    // クエリ
    //=========================================================================

    int getBranchAtPosition(const glm::vec3& position) const;
    int getTerminalSegmentAtPosition(const glm::vec3& position) const;
    std::vector<int> getBranchesAtPosition(const glm::vec3& position) const;
    std::vector<int> getTerminalSegmentsAtPosition(const glm::vec3& position) const;
    glm::vec3 getColorAtPosition(const glm::vec3& position) const;
    int getPrimaryBranchAtPosition(const glm::vec3& position) const;

    //=========================================================================
    // バッチ処理
    //=========================================================================

    void assignTrianglesToSegments(const std::vector<float>& vertices,
                                   const std::vector<int>& indices,
                                   std::vector<int>& outTriangleSegments) const;

    //=========================================================================
    // 状態確認・アクセス
    //=========================================================================

    bool isBuilt() const { return isBuilt_; }
    size_t getNumBranches() const { return branches_.size(); }
    size_t getNumSamplePoints() const { return numSamples_; }
    const std::vector<BranchPath>& getBranches() const { return branches_; }

    int branchToTerminal(int branchId) const {
        auto it = branchToTerminal_.find(branchId);
        return (it != branchToTerminal_.end()) ? it->second : -1;
    }

private:
    static void initializeGeogram();
    static bool geogramInitialized_;

    void generateSamplePoints(float samplingInterval);
    bool buildVoronoi();
    void assignColors();

    std::vector<BranchPath> branches_;
    std::vector<double> sampleCoords_;
    std::vector<int> sampleToBranch_;
    std::vector<int> sampleToTerminal_;
    std::vector<std::vector<int>> sampleToBranches_;
    std::vector<std::vector<int>> sampleToTerminals_;
    size_t numSamples_;

    std::map<int, int> branchToTerminal_;
    std::map<int, int> terminalToBranch_;

#if GEOGRAM_AVAILABLE
    GEO::Delaunay_var delaunay_;
#endif

    bool isBuilt_;
};

//=============================================================================
// テンプレート関数の実装
//=============================================================================
template<typename NodeType, typename SegmentType>
bool VoronoiPathSegmenter::build(const std::vector<NodeType>& nodes,
                                 const std::vector<SegmentType>& segments,
                                 int rootSegmentId,
                                 float samplingInterval) {
#if !GEOGRAM_AVAILABLE
    // GEOGRAMなし: 警告を出して失敗
    (void)nodes; (void)segments; (void)rootSegmentId; (void)samplingInterval;
    std::cerr << "[VoronoiPathSegmenter] Voronoi3D is not available." << std::endl;
    std::cerr << "  Platform: " << getPlatformName() << std::endl;
    std::cerr << "  Use OBJ or SkeletonDistance mode instead." << std::endl;
    return false;
#else
    // GEOGRAMあり: フル実装
    branches_.clear();
    sampleCoords_.clear();
    sampleToBranch_.clear();
    sampleToTerminal_.clear();
    branchToTerminal_.clear();
    terminalToBranch_.clear();
    numSamples_ = 0;
    isBuilt_ = false;

    if (segments.empty() || nodes.empty()) {
        std::cerr << "[VoronoiPathSegmenter] Error: Empty data" << std::endl;
        return false;
    }

    std::cout << "\n=== Building Voronoi 3D Path Segmentation ===" << std::endl;
    std::cout << "  Root segment: " << rootSegmentId << std::endl;
    std::cout << "  Total segments: " << segments.size() << std::endl;
    std::cout << "  Total nodes: " << nodes.size() << std::endl;
    std::cout << "  Sampling interval: " << samplingInterval << std::endl;

    // Step 1: 末端セグメントを特定
    std::vector<int> terminalIds;
    for (size_t i = 0; i < segments.size(); ++i) {
        if (segments[i].childIds.empty()) {
            terminalIds.push_back(static_cast<int>(i));
        }
    }
    std::cout << "  Terminal segments: " << terminalIds.size() << std::endl;

    // Step 2: 各末端からルートまでのパスを構築
    int branchId = 0;
    for (int terminalId : terminalIds) {
        BranchPath branch;
        branch.branchId = branchId;
        branch.terminalSegmentId = terminalId;

        std::vector<int> segmentChain;
        int currentSeg = terminalId;
        int maxDepth = static_cast<int>(segments.size());
        int depth = 0;

        while (currentSeg >= 0 && currentSeg < static_cast<int>(segments.size()) && depth < maxDepth) {
            segmentChain.push_back(currentSeg);
            currentSeg = segments[currentSeg].parentId;
            depth++;
        }

        std::reverse(segmentChain.begin(), segmentChain.end());
        branch.segmentIds = segmentChain;

        for (int segId : segmentChain) {
            if (segId < 0 || segId >= static_cast<int>(segments.size())) continue;

            const auto& seg = segments[segId];
            for (int nodeId : seg.nodeIds) {
                if (nodeId < 0 || nodeId >= static_cast<int>(nodes.size())) continue;

                glm::vec3 pos = nodes[nodeId].position;
                if (branch.pathPoints.empty() ||
                    glm::length(pos - branch.pathPoints.back()) > 1e-6f) {
                    branch.pathPoints.push_back(pos);
                }
            }
        }

        if (branch.pathPoints.size() >= 2) {
            branchToTerminal_[branchId] = terminalId;
            terminalToBranch_[terminalId] = branchId;
            branches_.push_back(branch);
            branchId++;
        }
    }

    std::cout << "  Built " << branches_.size() << " branch paths" << std::endl;

    if (branches_.empty()) {
        std::cerr << "[VoronoiPathSegmenter] Error: No branches found" << std::endl;
        return false;
    }

    generateSamplePoints(samplingInterval);
    std::cout << "  Generated " << numSamples_ << " sample points" << std::endl;

    if (numSamples_ == 0) {
        std::cerr << "[VoronoiPathSegmenter] Error: No sample points" << std::endl;
        return false;
    }

    if (!buildVoronoi()) {
        return false;
    }

    assignColors();
    isBuilt_ = true;

    std::cout << "=== Voronoi 3D Build Complete ===" << std::endl;
    return true;
#endif
}

} // namespace VoxelSkeleton

#endif // VORONOI_PATH_SEGMENTATION_H
