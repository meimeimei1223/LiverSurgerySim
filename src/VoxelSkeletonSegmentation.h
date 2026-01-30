#ifndef VOXEL_SKELETON_SEGMENTATION_H
#define VOXEL_SKELETON_SEGMENTATION_H

#include "CentVoxTetrahedralizerHybrid.h"  // SimpleBVHを使用
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <memory>
#include <glm/glm.hpp>
#include <GL/glew.h>
#include "VoronoiPathSegmentation.h"

class ShaderProgram;
class SoftBodyGPUDuo;  // ← ここ！namespace の外
//=============================================================================
// ボクセルベースのスケルトン抽出＆セグメンテーション
// Voxel Thinning（細線化）法による門脈セグメンテーション
//=============================================================================

namespace VoxelSkeleton {

// ファイルの先頭付近、namespaceの前に追加

//=============================================================================
// OBJセグメント専用BVH（データを所有する）
//=============================================================================
class OBJSegmentBVH {
private:
    struct BVHNode {
        glm::vec3 min;
        glm::vec3 max;
        int leftChild;
        int rightChild;
        std::vector<size_t> triangles;
    };

    std::vector<BVHNode> nodes_;
    std::vector<float> vertices_;
    std::vector<int> indices_;
    static constexpr int MAX_TRIANGLES_PER_LEAF = 10;

public:
    OBJSegmentBVH() = default;

    void build(const std::vector<float>& vertices, const std::vector<int>& indices);
    int countIntersections(const glm::vec3& origin, const glm::vec3& direction) const;
    bool isEmpty() const { return nodes_.empty(); }

private:
    int buildRecursive(std::vector<size_t>& triangles, int depth);
    void traverseAndCount(int nodeIdx, const glm::vec3& origin,
                          const glm::vec3& direction, int& count) const;
    bool rayBoxIntersect(const glm::vec3& origin, const glm::vec3& direction,
                         const glm::vec3& boxMin, const glm::vec3& boxMax) const;
    bool rayTriangleIntersect(const glm::vec3& origin, const glm::vec3& direction,
                              size_t triIdx) const;
    glm::vec3 getVertex(size_t triIdx, int vertexIdx) const;
    glm::vec3 getTriangleCenter(size_t triIdx) const;
};

struct SkeletonNodeBinding {
    int tetIdx;              // portal SoftBodyのどの四面体に属するか
    glm::vec4 baryCoords;    // バリセントリック座標
    glm::vec3 initialPos;    // 初期位置（参照用）

    SkeletonNodeBinding() : tetIdx(-1), baryCoords(0.25f), initialPos(0) {}
};

struct SkeletonNode {
    int id;
    glm::ivec3 voxelIndex;
    glm::vec3 position;
    float radius;
    std::vector<int> neighbors;
    int segmentId;

    SkeletonNode() : id(-1), voxelIndex(0), position(0), radius(0), segmentId(-1) {}
};

struct Segment {
    int id;
    std::vector<int> nodeIds;
    std::set<int> triangleIndices;
    float averageRadius;
    glm::vec3 color;
    int parentId;
    std::vector<int> childIds;
    int hierarchyLevel;

    Segment() : id(-1), averageRadius(0), color(0.7f), parentId(-1), hierarchyLevel(0) {}
};

struct SelectionState {
    int selectedSegmentId;
    std::set<int> selectedSegments;
    glm::vec3 highlightColor;

    SelectionState() : selectedSegmentId(-1), highlightColor(1.0f, 0.8f, 0.0f) {}
    void clear() { selectedSegmentId = -1; selectedSegments.clear(); }
};

class VesselSegmentation {
public:
    VesselSegmentation(int gridSize = 100);
    ~VesselSegmentation();

    // OBJファイルから解析
    bool analyzeFromFile(const std::string& objPath);

    // 頂点・インデックスから解析
    bool analyze(const std::vector<GLfloat>& vertices,
                 const std::vector<GLuint>& indices);

    void selectByTriangle(int triangleIndex);
    void selectSegment(int segmentId);
    void clearSelection();

    void draw(ShaderProgram& shader,
              const std::vector<GLfloat>& vertices,
              const std::vector<GLuint>& indices,
              const glm::mat4& model,
              const glm::mat4& view,
              const glm::mat4& projection,
              const glm::vec3& cameraPos);

    void drawSkeleton(ShaderProgram& shader,
                      const glm::mat4& model,
                      const glm::mat4& view,
                      const glm::mat4& projection);

    int getSegmentCount() const { return segments_.size(); }
    const std::vector<Segment>& getSegments() const { return segments_; }
    const std::vector<SkeletonNode>& getNodes() const { return nodes_; }
    int getSegmentByTriangle(int triangleIndex) const;
    const SelectionState& getSelection() const { return selection_; }

    // ★ 追加：下流セグメント取得（public版）
    std::set<int> getDownstreamSegments(int segmentId) const {
        std::set<int> result;
        collectDownstream(segmentId, result);
        return result;
    }
    // デバッグ機能
    struct DebugInfo {
        std::vector<std::pair<int, int>> loopEdges;
        std::vector<std::set<int>> connectedComponents;
        std::vector<int> isolatedNodes;
        int loopCount;
        int componentCount;
        bool hasDisconnection;
    };

    DebugInfo analyzeTopology();
    void drawDebug(ShaderProgram& shader,
                   const glm::mat4& model,
                   const glm::mat4& view,
                   const glm::mat4& projection,
                   const DebugInfo& debugInfo);

    // ループ除去・検証機能
    int removeLoops();
    bool verifyTreeStructure();

    // 不連続成分の接続
    int connectDisconnectedComponents();



private:
    int gridSize_;
    glm::vec3 gridMin_, gridMax_, voxelSize_;

    std::vector<std::vector<std::vector<bool>>> insideVoxels_;
    std::vector<std::vector<std::vector<float>>> distanceField_;
    std::vector<std::vector<std::vector<bool>>> skeletonVoxels_;

    std::vector<SkeletonNode> nodes_;
    std::map<int, int> voxelToNode_;

    std::vector<Segment> segments_;
    std::map<int, int> triangleToSegment_;
    std::map<int, std::set<int>> segmentConnections_;
    int rootSegmentId_;

    SelectionState selection_;

    std::unique_ptr<SimpleBVH> bvh_;

    GLuint skeletonVAO_, skeletonVBO_;
    bool buffersInitialized_;

    std::vector<GLfloat> meshVertices_;
    std::vector<GLuint> meshIndices_;

    // 内部メソッド
    bool loadOBJ(const std::string& path);
    void initializeVoxelGrid();
    void classifyInsideVoxels();
    void carveExternalVoxels();
    void computeDistanceTransform();

    // スケルトン抽出（Voxel Thinning）
    void extractSkeleton();
    bool canRemoveVoxel(int x, int y, int z);
    int countNeighbors26(int x, int y, int z);
    void pruneShortBranches(int minLength);
    void connectNearEndpoints();

    // グラフ・セグメント
    void buildSkeletonGraph();
    void removeGraphLoops();
    void segmentSkeleton();
    void mergeSmallSegments(int minNodes);
    void assignTrianglesToSegments();
    void buildHierarchy();

    // ヘルパー
    int voxelIndex1D(int x, int y, int z) const;
    glm::vec3 voxelCenter(int x, int y, int z) const;
    bool isValidVoxel(int x, int y, int z) const;
    void collectDownstream(int segmentId, std::set<int>& result) const;
    void assignColors();
    static glm::vec3 getColorForIndex(int index);

    void initSkeletonBuffers();
    void updateSkeletonBuffers();


public:
    // スケルトンノードのバインディング情報
    std::vector<SkeletonNodeBinding> nodeBindings;
    bool isBoundToSoftBody = false;

    // portal SoftBodyにスケルトンをバインド
    void bindToPortalSoftBody(const std::vector<float>& positions,
                              const std::vector<int>& tetIds,
                              size_t numTets);

    // portal SoftBodyの変形に追従してノード位置を更新
    void updateNodesFromPortal(const std::vector<float>& positions,
                               const std::vector<int>& tetIds);

    // バインディング解除
    void unbindFromSoftBody();


    // publicセクションに追加（getDownstreamSegmentsの近くに）

    // ★ スケルトン延長機能
    bool extendTerminalBranch(int segmentId, int nodesToAdd = 5);
    void autoExtendShortTerminalBranches(float shortThreshold = 0.5f, float targetRatio = 0.8f);
    void toggleExtendedSkeleton();
    bool isUsingExtendedSkeleton() const { return useExtendedSkeleton_; }
    bool isTerminalSegment(int segmentId) const;

    // privateセクションに追加（buildHierarchy()の近くに）

    // ★ スムージング
    void smoothSkeletonGraph(int iterations = 3, float factor = 0.5f);

    // ★ スケルトン延長用ヘルパー
    std::vector<int> findTerminalSegments();
    int findEndpointNode(int segmentId);
    float getAverageNodeSpacing(int segmentId);
    float calculateSegmentLength(int segmentId);
    glm::vec3 calculateExtensionDirection(int endpointNodeId, float repulsionWeight = 0.3f);
    glm::vec3 calculateAdaptiveDirection(int endpointNodeId, const glm::vec3& baseDirection);

    // ★ バックアップ用
    struct SkeletonBackup {
        std::vector<SkeletonNode> nodes;
        std::vector<Segment> segments;
        std::map<int, int> triangleToSegment;
        bool hasBackup = false;
    };

    SkeletonBackup originalSkeleton_;
    SkeletonBackup extendedSkeleton_;
    bool useExtendedSkeleton_ = false;

    void backupCurrentSkeleton(SkeletonBackup& backup);
    void restoreSkeletonFromBackup(const SkeletonBackup& backup);


    SkeletonBackup autoExtendedSkeleton_;  // 自動延長版（別途保持）
    bool manualExtended_ = false;          // 手動延長中フラグ
    // 手動延長用
    void backupBeforeManualExtension();
    void saveManualExtension();
    void revertToAutoExtended();
    bool isManualExtended() const { return manualExtended_; }
    bool hasBackup() const { return originalSkeleton_.hasBackup; }


    // private セクションに追加
    std::set<std::pair<int, int>> artificialConnections_;  // 人工接続を記録

    // private メソッドに追加
    glm::vec3 getNodeDirection(int nodeId) const;
    // private:
    float pointToSegmentDistance(const glm::vec3& point,
                                 const glm::vec3& segStart,
                                 const glm::vec3& segEnd,
                                 glm::vec3& closestPoint) const;
    // private:
    glm::vec3 getExtendDirection(int endpointId) const;
  public:

      // private:
      // private:
      std::vector<float> liverVertices_;    // GLfloat → float
      std::vector<int> liverIndices_;       // GLuint → int (既にintかも)
      bool hasLiverMesh_ = false;

      std::unique_ptr<SimpleBVH> liverBVH_;


      // public:
      // 変更後
      void setLiverMesh(const std::vector<float>& vertices,
                        const std::vector<int>& indices);
      bool isInsideLiver(const glm::vec3& position) const;
      // public:
      // public:
      float calculateDistanceFromRoot(int segmentId) const;
      float calculateSegmentLength(int segmentId) const;
      // 変更後
      void autoExtendShortTerminalBranches(float shortThreshold = 0.5f,
                                           float targetRatio = 0.8f,
                                           float distanceRatioThreshold = 0.7f);



      //=============================================================================
      // OBJベースのセグメンテーション
      //=============================================================================
      struct OBJSegment {
          int id;                           // セグメントID (1-8など)
          std::string name;                 // "S1", "S2", etc.
          std::vector<float> vertices;      // OBJの頂点
          std::vector<int> indices;         // OBJの三角形インデックス
          glm::vec3 boundMin, boundMax;     // バウンディングボックス
          glm::vec3 color;                  // 表示用の色

          // ★ BVHを追加
          std::unique_ptr<OBJSegmentBVH> bvh;

          OBJSegment() : id(-1), boundMin(FLT_MAX), boundMax(-FLT_MAX), color(0.5f) {}

          // ムーブコンストラクタ
          OBJSegment(OBJSegment&& other) noexcept = default;
          OBJSegment& operator=(OBJSegment&& other) noexcept = default;

          // コピーは禁止
          OBJSegment(const OBJSegment&) = delete;
          OBJSegment& operator=(const OBJSegment&) = delete;
      };

      // private: に追加
      std::vector<OBJSegment> objSegments_;                    // S1-S8のOBJデータ
      std::vector<int> nodeToOBJSegmentId_;                    // ノード → OBJセグメントID
      std::vector<int> triangleToOBJSegmentId_;                // 三角形 → OBJセグメントID
      bool useOBJSegmentation_ = false;                        // OBJベースを使用中か
      int selectedOBJSegment_ = -1;                            // 選択中のOBJセグメント

      // public: に追加
      // OBJセグメンテーション関連
      bool loadOBJSegments(const std::vector<std::string>& objPaths);
      bool loadOBJSegment(const std::string& objPath, int segmentId, const std::string& name);
      void applyOBJSegmentation();
      void toggleOBJSegmentation();
      bool isUsingOBJSegmentation() const { return useOBJSegmentation_; }

      // OBJセグメント選択
      void selectOBJSegment(int segmentId);
      void clearOBJSegmentSelection();
      int getSelectedOBJSegment() const { return selectedOBJSegment_; }
      const std::vector<int>& getNodeToOBJSegmentId() const { return nodeToOBJSegmentId_; }

      // 内部判定
      bool isInsideOBJSegment(const glm::vec3& position, int segmentIndex) const;
      // public:
      const std::vector<OBJSegment>& getOBJSegments() const { return objSegments_; }
      const std::vector<int>& getTriangleToOBJSegmentId() const { return triangleToOBJSegmentId_; }
      glm::vec3 getOBJSegmentColor(int segmentId) const;


      // クリック位置からOBJセグメントIDを取得
      int getOBJSegmentAtPosition(const glm::vec3& position) const;


private:
      //=========================================================================
      // Voronoi 3D セグメンテーション（パスサンプリング + GEOGRAM）
      //=========================================================================
      std::unique_ptr<VoronoiPathSegmenter> voronoiSegmenter_;
      std::vector<int> triangleToVoronoiSegment_;    // Voronoi3Dの結果キャッシュ

      //=========================================================================
      // セグメンテーションモード管理
      //=========================================================================
      SegmentationMode currentMode_ = SegmentationMode::SkeletonDistance;

      // 各モードの結果キャッシュ（モード切り替え時に再計算を避ける）
      std::map<int, int> triangleToSegment_Distance_;   // SkeletonDistanceの結果
      std::map<int, int> triangleToSegment_Voronoi_;    // Voronoi3Dの結果
      // OBJは既存の triangleToOBJSegmentId_ を使用
      void rebuildSegmentTriangleLists();

  public:
      //=========================================================================
      // セグメンテーションモード管理 (public)
      //=========================================================================

      /// 現在のモードを取得
      SegmentationMode getSegmentationMode() const { return currentMode_; }

      /// モードを設定し、対応するセグメンテーションを適用
      void setSegmentationMode(SegmentationMode mode);

      /// 次のモードに切り替え（Key 'O' 用）
      /// OBJ → SkeletonDistance → Voronoi3D → OBJ ...
      void cycleSegmentationMode();

      /// 現在のモード名を取得
      const char* getCurrentModeName() const {
          return getSegmentationModeName(currentMode_);
      }

      //=========================================================================
      // Voronoi 3D セグメンテーション (public)
      //=========================================================================

      /// Voronoi 3Dセグメンターを構築
      /// @param samplingInterval パスのサンプリング間隔（デフォルト0.5）
      bool buildVoronoi3D(float samplingInterval = 0.5f);

      /// Voronoi 3D分割を適用
      void applyVoronoi3DSegmentation();

      /// Voronoi 3Dが構築済みかどうか
      bool hasVoronoi3D() const {
          return voronoiSegmenter_ && voronoiSegmenter_->isBuilt();
      }

      /// 指定位置のブランチIDを取得（Voronoi3D）
      int getVoronoiBranchAtPosition(const glm::vec3& pos) const;

      /// 指定位置の末端セグメントIDを取得（Voronoi3D）
      int getVoronoiSegmentAtPosition(const glm::vec3& pos) const;

      /// Voronoiセグメンターへのアクセス
      const VoronoiPathSegmenter* getVoronoiSegmenter() const {
          return voronoiSegmenter_.get();
      }

      /// ブランチ数を取得
      size_t getNumBranches() const {
          return voronoiSegmenter_ ? voronoiSegmenter_->getNumBranches() : 0;
      }

      //=========================================================================
      // 描画支援（モード対応）
      //=========================================================================

      /// 現在のモードに応じた三角形の色を取得
      glm::vec3 getTriangleColorByCurrentMode(int triangleIndex) const;

  private:
      int selectedBranchId_ = -1;  // Voronoi3D用


  private:
      std::set<int> selectedBranchIds_;  // 複数選択対応

  public:
      // ブランチ選択（複数対応）
      void selectBranches(const std::vector<int>& branchIds) {
          selectedBranchIds_.clear();
          for (int id : branchIds) {
              selectedBranchIds_.insert(id);
          }
      }
      void selectBranch(int branchId) {
          selectedBranchIds_.clear();
          selectedBranchIds_.insert(branchId);
      }
      const std::set<int>& getSelectedBranches() const { return selectedBranchIds_; }
      int getSelectedBranch() const {
          return selectedBranchIds_.empty() ? -1 : *selectedBranchIds_.begin();
      }
      void clearBranchSelection() { selectedBranchIds_.clear(); }
  public:
      // ===== ファクトリメソッド =====
      // OBJあり版
      static VesselSegmentation* create(
          const std::string& portalPath,
          const std::vector<std::string>& objPaths,
          class SoftBodyGPUDuo* liver,
          class SoftBodyGPUDuo* portal = nullptr,
          int resolution = 120);

      // OBJなし版
      static VesselSegmentation* create(
          const std::string& portalPath,
          class SoftBodyGPUDuo* liver,
          class SoftBodyGPUDuo* portal = nullptr,
          int resolution = 120);


};

} // namespace VoxelSkeleton

#endif
