#ifndef SOFTBODYGPUDUO_H
#define SOFTBODYGPUDUO_H
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include <GL/glew.h>
#include "Hash.h"
#include <set>
#include <map>
#include <functional>  // 追加
#include <chrono>
#include <array>


// 前方宣言
namespace VoxelSkeleton {
class VesselSegmentation;
struct SkeletonNode;
struct Segment;
}


class ShaderProgram;

class SoftBodyGPUDuo {
public:
    struct MeshData {
        std::vector<float> verts;
        std::vector<int> tetIds;
        std::vector<int> tetEdgeIds;
        std::vector<int> tetSurfaceTriIds;
    };

    // SoftBody.h に追加
    struct SkeletonBinding {
        // 四面体 → セグメントID マッピング
        std::vector<int> tetToSegmentId;

        // 頂点 → セグメントID マッピング（表面描画用）
        std::vector<int> vertexToSegmentId;

        // 現在選択されているセグメントID群
        std::set<int> selectedSegments;

        // セグメントごとの色（セグメントID → 色）
        std::map<int, glm::vec4> segmentColors;

        // 非選択部分の色
        glm::vec4 unselectedColor = glm::vec4(0.6f, 0.6f, 0.6f, 0.3f);

        // バインド済みフラグ
        bool isBound = false;

        // 選択変更時のコールバック（オプション）
        std::function<void(const std::set<int>&)> onSelectionChanged;
        // ★ OBJベースのセグメントID（S1-S8）
        std::vector<int> tetToOBJSegmentId;      // 四面体 → OBJセグメントID
        std::vector<int> vertexToOBJSegmentId;   // 頂点 → OBJセグメントID
        bool objSegmentsBound = false;

        // ★★★ Voronoi3D用のマッピング（OBJと同じ構造）★★★
        std::vector<std::vector<int>> vertexToVoronoi3DBranchIds;  // 頂点→ブランチIDリスト
        std::vector<glm::vec3> vertexToVoronoi3DColor;             // 頂点→基本色
        bool voronoi3DBound = false;
        // ★★★ 追加 ★★★
        std::vector<std::vector<int>> tetToVoronoi3DBranchIds;  // 四面体→ブランチIDリスト（カット用キャッシュ）


        void clear() {
            tetToSegmentId.clear();
            vertexToSegmentId.clear();
            tetToOBJSegmentId.clear();
            vertexToOBJSegmentId.clear();
            vertexToVoronoi3DBranchIds.clear();   // ★追加
            vertexToVoronoi3DColor.clear();       // ★追加
            selectedSegments.clear();
            segmentColors.clear();
            isBound = false;
            objSegmentsBound = false;
            voronoi3DBound = false;               // ★追加
            tetToVoronoi3DBranchIds.clear();   // ★追加
        }


    };

    SkeletonBinding skeletonBinding;
    // カット対象の四面体を取得（3モード共通）
    // skeleton: VesselSegmentationへの参照
    // 戻り値: カット対象の四面体インデックスリスト
    std::vector<int> getSelectedTetsForCut(const VoxelSkeleton::VesselSegmentation& skeleton) const;

    // Voronoi3D四面体キャッシュのバインド（bindVoronoi3Dから呼ばれる）
    void bindVoronoi3DTets(const VoxelSkeleton::VesselSegmentation& skeleton);
    // publicセクションに追加
    void bindOBJSegments(const VoxelSkeleton::VesselSegmentation& skeleton);
    int getTetOBJSegmentId(int tetIndex) const;
    int getVertexOBJSegmentId(int vertexIndex) const;
    // public:
    void forceUpdateSegmentColors();


    // 既存の selectSegmentWithDownstream の下に追加
    void toggleSegmentSelection(int segmentId,
                                const VoxelSkeleton::VesselSegmentation& skeleton);

    // 追加選択用（オーバーロード）
    void selectSegmentWithDownstream(int segmentId,
                                     const VoxelSkeleton::VesselSegmentation& skeleton,
                                     bool addToSelection);

    //--------------------------------------------------------------------------
    // スケルトンバインディング
    //--------------------------------------------------------------------------

    // スケルトンと四面体メッシュを紐付け
    void bindToSkeleton(const VoxelSkeleton::VesselSegmentation& skeleton);

    // 手動でセグメントIDを設定（外部で計算した場合）
    void setTetSegmentIds(const std::vector<int>& tetSegmentIds);

    // バインド解除
    void unbindSkeleton();

    // バインド状態確認
    bool isSkeletonBound() const { return skeletonBinding.isBound; }

    //--------------------------------------------------------------------------
    // セグメント選択
    //--------------------------------------------------------------------------

    // 単一セグメントを選択（下流も含む）
    void selectSegmentWithDownstream(int segmentId,
                                     const VoxelSkeleton::VesselSegmentation& skeleton);

    // 複数セグメントを直接選択
    void selectSegments(const std::set<int>& segmentIds);

    // 選択をクリア
    void clearSegmentSelection();

    // 選択状態取得
    const std::set<int>& getSelectedSegments() const {
        return skeletonBinding.selectedSegments;
    }

    //--------------------------------------------------------------------------
    // 色設定
    //--------------------------------------------------------------------------

    // セグメントの色を設定
    void setSegmentColor(int segmentId, const glm::vec4& color);

    // 全セグメントに自動で色を割り当て
    void assignSegmentColors(int totalSegments);

    // 非選択部分の色を設定
    void setUnselectedColor(const glm::vec4& color);

    //--------------------------------------------------------------------------
    // 描画（セグメント色分け対応）
    //--------------------------------------------------------------------------

    // スムーズメッシュをセグメント色分けで描画
    void drawSmoothMeshWithSegments(ShaderProgram& shader);

    // HighResメッシュをセグメント色分けで描画
    void drawHighResMeshWithSegments(ShaderProgram& shader);

    //--------------------------------------------------------------------------
    // ユーティリティ
    //--------------------------------------------------------------------------

    // 特定の四面体が属するセグメントIDを取得
    int getTetSegmentId(int tetIndex) const;

    // 特定の頂点が属するセグメントIDを取得
    int getVertexSegmentId(int vertexIndex) const;

    // セグメントに属する四面体インデックスを取得
    std::vector<int> getTetsInSegment(int segmentId) const;

    // セグメントに属する表面三角形インデックスを取得
    std::vector<int> getSurfaceTrianglesInSegment(int segmentId) const;

    // 統計情報出力
    void printSkeletonBindingStats() const;


    // =====================================
    // SoftBody.h - publicセクションに追加
    // 親ソフトボディ追従機能
    // =====================================
    // 追従モードの種類
    enum FollowMode {
        FOLLOW_NONE,           // 追従なし（デフォルト）
        FOLLOW_PARENT_SOFTBODY // 親ソフトボディに追従
    };

    // 追従モードのパラメータ
    struct FollowParams {
        float barycentricEpsilon;   // 重心座標の許容誤差
        float maxAcceptableDist;    // 最大許容距離
        float lowQualityFactor;     // 低品質判定の倍率
        float border;               // 検索範囲の余裕

        // デフォルトコンストラクタ
        FollowParams()
            : barycentricEpsilon(0.01f)
            , maxAcceptableDist(0.1f)
            , lowQualityFactor(2.0f)
            , border(0.05f)
        {}

        // パラメータ指定コンストラクタ
        FollowParams(float epsilon, float maxDist, float factor, float bord)
            : barycentricEpsilon(epsilon)
            , maxAcceptableDist(maxDist)
            , lowQualityFactor(factor)
            , border(bord)
        {}
    };

    // 追従モード用データ
    FollowMode followMode = FOLLOW_NONE;
    SoftBodyGPUDuo* parentSoftBody = nullptr;
    FollowParams followParams;

    // 親に対するスキニング情報 [tetIdx, b0, b1, b2] per LowRes vertex
    // サイズ: numLowResParticles * 4
    // b3 = 1 - b0 - b1 - b2 で計算
    std::vector<float> skinningToParent;

    // 親に固定されている頂点フラグ
    std::vector<bool> isAnchoredToParent;

    // アンカー頂点数
    int numAnchoredVertices = 0;

    // 元の質量分布を保持（アンカー解除時に復元用）
    std::vector<float> originalInvMasses;

    // 追従モード用メソッド
    void setParentSoftBody(SoftBodyGPUDuo* parent, const FollowParams& params = FollowParams());
    void computeSkinningToParent();
    void updateFromParent();  // 親の動きに追従
    void clearParentSoftBody();
    bool hasParentSoftBody() const { return parentSoftBody != nullptr; }
    int getNumAnchoredVertices() const { return numAnchoredVertices; }
    FollowMode getFollowMode() const { return followMode; }
private:
    //--------------------------------------------------------------------------
    // 内部実装
    //--------------------------------------------------------------------------

    // 四面体の重心を計算
    glm::vec3 computeTetCentroid(int tetIndex) const;

    // 最も近いスケルトンノードを探す
    int findNearestSkeletonNode(const glm::vec3& position,
                                const std::vector<VoxelSkeleton::SkeletonNode>& nodes) const;

    // 頂点にセグメントIDを伝播
    void propagateSegmentIdsToVertices();

    // 頂点カラー配列を更新
    void updateVertexColors();

    //--------------------------------------------------------------------------
    // GPU関連（セグメント色分け用）
    //--------------------------------------------------------------------------
public:
    // 頂点カラーバッファ
    GLuint segmentColorVBO = 0;
    std::vector<float> vertexColors;  // RGBA per vertex
    bool segmentColorsNeedUpdate = true;

    void setupSegmentColorBuffer();
    void updateSegmentColorBuffer();
    void deleteSegmentColorBuffer();
public:
    SoftBodyGPUDuo(const MeshData& lowResTetMesh, const MeshData& highResTetMesh,
             float edgeCompliance, float volCompliance);
    ~SoftBodyGPUDuo();
    void lowResPreSolve(float dt, const glm::vec3& gravity);
    void lowResSolve(float dt);
    void lowResPostSolve(float dt);
    void updateLowResMesh();
    void updateLowResTetMeshes();
    void updateHighResMesh();
    void drawLowResTet(ShaderProgram& shader);
    void drawLowResTetMesh(ShaderProgram& shader);
    void drawHighResMesh(ShaderProgram& shader);
    void drawHighResTetMesh(ShaderProgram& shader);
    void startLowResGrab(const glm::vec3& pos);
    void moveLowResGrabbed(const glm::vec3& pos, const glm::vec3& vel);
    void endLowResGrab(const glm::vec3& pos, const glm::vec3& vel);
    void applyShapeRestoration(float strength);
    void initialize(const MeshData& lowResMesh, const MeshData& highResMesh,
                    float edgeComp, float volComp);
    void clear();
    static MeshData loadTetMesh(const std::string& filename);
    static MeshData ReadVertexAndFace(const std::string& objPath);
    bool showLowHighTetMesh;
    bool showHighResMesh;
    glm::mat4 lowRes_modelMatrix;
    const std::vector<float>& getLowResPositions() const { return lowRes_positions; }
    const std::vector<float>& getHighResPositions() const { return highRes_positions; }
    const MeshData& getLowResMeshData() const { return lowResMeshData; }
    const MeshData& getHighResMeshData() const { return  highResMeshData; }
    std::map<int, int> vertexMapping;
    std::set<int> fixedVertices;
    int lowRes_grabId;
    float lowRes_grabInvMass;
    glm::vec3 lowRes_grabOffset;
    std::vector<int> lowRes_activeParticles;
    std::vector<float> lowRes_oldInvMasses;
    std::vector<float> lowRes_positions;
    std::vector<float> lowRes_prevPositions;
    std::vector<float> lowRes_velocities;
    std::vector<int> lowRes_tetIds;
    std::vector<int> lowRes_edgeIds;
    std::vector<float> lowRes_invMasses;
    std::vector<float> lowRes_restVols;
    std::vector<float> lowRes_edgeLengths;
    MeshData lowResMeshData;
    MeshData highResMeshData;
    size_t numLowResParticles;
    size_t numLowTets;
    std::vector<float> highRes_positions;
    size_t numHighTets;
    size_t numHighResVerts;
    std::vector<int> highResTetIds;
    std::vector<int> highResEdgeIds;
    void initPhysicsLowRes();
    void computeLowResNormals();
    bool smoothDisplayMode = false;
    int smoothingIterations = 1;
    float smoothingFactor = 0.5f;
    bool enableSizeAdjustment = true;
    int scalingMethod = 2;
    std::vector<float> smoothedVertices;
    std::vector<int> smoothSurfaceTriIds;
    void enableSmoothDisplay(bool enable);
    void setSmoothingParameters(int iterations, float factor, bool adjustSize = true, int method = 2);
    void generateSmoothSurface();
    void applySmoothingToSurface();
    void updateSmoothMesh();
    void drawSmoothMesh(ShaderProgram& shader);
    void clearSmoothingData();
    std::vector<int> getVertexNeighbors(int vertexId);
    std::vector<bool> edgeValid;
    std::vector<bool> lowRes_tetValid;
    std::vector<bool> highResTetValid;
    std::vector<int> highResValidEdges;
    std::vector<bool> highResEdgeValid;
    std::vector<int> highResValidTriangles;
    int highResInvalidatedCount;
    void invalidateHighResTetrahedra(const std::vector<int>& tetIndices);
    void invalidateLowResTetrahedra(const std::vector<int>& tetIndices);
    void validateHighResTetrahedra(const std::vector<int>& tetIndices);
    void validateAllHighResTetrahedra();
    int getInvalidatedHighResTetCount() const;
    void invalidateLowResTetsWithoutHighRes();
    int getInvalidatedLowResTetCount() const;
    void updateHighResEdgeValidity();
    float lowRes_motionDampingFactor = 0.98f;
    float lowRes_boundaryDampingFactor = 0.9f;
    bool lowRes_useBoundaryDamping = false;
    void applySmoothedDamping();
    // ★ 初期の逆質量（カット高速化用、初期化後は不変）
    //std::vector<float> initialInvMasses;
    std::vector<int> getActiveHighResSurfaceTriIds() const {
        if (smoothDisplayMode && !smoothSurfaceTriIds.empty()) {
            return smoothSurfaceTriIds;
        }
        return highResMeshData.tetSurfaceTriIds;
    }
    int getHighResInvalidatedTetCount() const { return highResInvalidatedCount; }
    void updateHighResValidSurface();
    int getSkinningTetIndex(size_t vertexId) const {
        if (vertexId >= numHighResVerts || skinningInfoLowToHigh.empty()) {
            return -1;
        }
        return static_cast<int>(skinningInfoLowToHigh[vertexId * 4]);
    }
    glm::vec4 getSkinningWeights(size_t vertexId) const {
        if (vertexId >= numHighResVerts || skinningInfoLowToHigh.empty()) {
            return glm::vec4(-1.0f);
        }
        size_t idx = vertexId * 4;
        float b0 = skinningInfoLowToHigh[idx + 1];
        float b1 = skinningInfoLowToHigh[idx + 2];
        float b2 = skinningInfoLowToHigh[idx + 3];
        float b3 = 1.0f - b0 - b1 - b2;
        return glm::vec4(b0, b1, b2, b3);
    }
    std::map<int, std::set<int>> highToLowTetMapping;
    std::map<int, std::set<int>> lowToHighTetMapping;
    bool tetMappingComputed = false;
    void computeTetToTetMappingLowToHigh();
    std::set<int> getLowResTetsFromHighResTet(int highResTetIdx);
    std::set<int> getHighResTetsFromLowResTet(int lowResTetIdx);
    std::vector<float> lowRes_initialPositions;
    std::vector<float> lowRes_initialRestVols;
    std::vector<float> lowRes_initialEdgeLengths;
    void saveLowResInitialShape();
    void restoreLowResInitialConstraints();
    float getTetVolumeFromInitial(int nr);
    void updateHighResTetMesh();
    std::vector<float> skinningInfoLowToHigh;
    bool useHighResMesh;
    std::vector<float> lowRes_edgeLambdas;
    std::vector<float> lowRes_volLambdas;
    std::vector<float> lowRes_tempBuffer;
    std::vector<float> lowRes_grads;
    float lowResEdgeCompliance;
    float lowResVolCompliance;
    float lowRes_damping;
    GLuint lowResVAO, lowResVBO, lowResEBO, lowResNormalVBO;
    GLuint lowResTetVAO, lowResTetVBO, lowResTetEBO;
    GLuint highResVAO, highResVBO, highResEBO, highResNormalVBO;
    GLuint highResTetVAO, highResTetVBO;
    std::vector<float> normalLines;
    static constexpr int volIdOrder[4][3] = {
        {1, 3, 2}, {0, 2, 3}, {0, 3, 1}, {0, 1, 2}
    };
    float lowResGetTetVolume(int nr);
    void solveLowResEdges(float compliance, float dt);
    void lowResSolveVolumes(float compliance, float dt);
    void setupLowResMesh(const std::vector<int>& surfaceTriIds);
    void setupLowResTetMesh();
    void setupHighResMesh();
    void setupHighResTetMesh();
    void setupNormalBuffer();
    void computeHighResNormals();
    void computeSkinningInfoLowToHigh(const std::vector<float>& highResVerts);
    void deleteLowHighBuffers();
    std::vector<int> lowResSurfaceTriToTet;
    void initLowResSurfaceToTetMapping();
    std::vector<int> highResSurfaceTriToTet;
    void initHighResSurfaceToTetMapping();
    struct BoundingBox {
        glm::vec3 min;
        glm::vec3 max;
        glm::vec3 center;
        glm::vec3 size;
    };
    BoundingBox calculateBoundingBox(const std::vector<float>& vertices, const std::set<int>& surfaceVertices);
    void adjustMeshSize(const BoundingBox& originalBBox, const BoundingBox& smoothedBBox,
                        const std::set<int>& surfaceVertices);
    GLuint smoothVAO = 0, smoothVBO = 0, smoothEBO = 0, smoothNormalVBO = 0;
    void setupSmoothBuffers();
    void updateSmoothBuffers();
    void deleteSmoothBuffers();
    std::vector<float> computeSmoothNormals();
    struct TriangleTetInfo {
        int v0, v1, v2;
        int tetIndex;
    };
    std::vector<TriangleTetInfo> smoothTriangleTetMap;
    std::vector<std::vector<int>> vertexNeighborsCache;
    bool neighborsComputed = false;
    void computeVertexNeighborsCache();
    void clearNeighborsCache();
    glm::vec3 calculateMeshCenter();
    std::vector<int> filterInvalidatedHighResSurface() const {
        std::vector<int> filteredTriIds;
        for (size_t i = 0; i < highResMeshData.tetSurfaceTriIds.size(); i += 3) {
            int v0 = highResMeshData.tetSurfaceTriIds[i];
            int v1 = highResMeshData.tetSurfaceTriIds[i + 1];
            int v2 = highResMeshData.tetSurfaceTriIds[i + 2];
            bool skipTriangle = false;
            for (size_t t = 0; t < numHighTets; t++) {
                if (!highResTetValid[t]) {
                    int matchCount = 0;
                    for (int j = 0; j < 4; j++) {
                        int vid = highResTetIds[t * 4 + j];
                        if (vid == v0 || vid == v1 || vid == v2) {
                            matchCount++;
                        }
                    }
                    if (matchCount == 3) {
                        skipTriangle = true;
                        break;
                    }
                }
            }
            if (!skipTriangle) {
                filteredTriIds.push_back(v0);
                filteredTriIds.push_back(v1);
                filteredTriIds.push_back(v2);
            }
        }
        return filteredTriIds;
    }
    std::vector<float> cachedSmoothNormals;
    bool normalsCorrected = false;
    struct ValidMeshData {
        std::vector<float> verts;
        std::vector<int> tetIds;
        std::vector<int> tetEdgeIds;
        std::vector<int> tetSurfaceTriIds;
        std::map<int, int> oldToNewVertexMap;
        int numValidTets;
        int numUsedVertices;
    };

private:
    // å››é¢ä½“ã®æ¨™æº–çš„ãªé¢å®šç¾©ï¼ˆSimpleTetMeshã¨åŒã˜ï¼‰
    static constexpr int TET_FACE_INDICES[4][3] = {
        {1, 2, 3},  // Face 0: v0ã®åå¯¾å´
        {0, 3, 2},  // Face 1: v1ã®åå¯¾å´
        {0, 1, 3},  // Face 2: v2ã®åå¯¾å´
        {0, 2, 1}   // Face 3: v3ã®åå¯¾å´
    };

    // SoftBody.h ã®publicã‚»ã‚¯ã‚·ãƒ§ãƒ³ã«è¿½åŠ
public:
    // ã‚¹ãƒˆãƒ¬ã‚¤ãƒ³ãƒªãƒŸãƒ†ã‚£ãƒ³ã‚°æ©Ÿèƒ½
    bool LowRes_enableStrainLimiting = true;  // ã‚¹ãƒˆãƒ¬ã‚¤ãƒ³ãƒªãƒŸãƒ†ã‚£ãƒ³ã‚°æœ‰åŠ¹/ç„¡åŠ¹

    // ã‚¨ãƒƒã‚¸ã®ã‚¹ãƒˆãƒ¬ã‚¤ãƒ³é–¾å€¤ï¼ˆæ®µéšŽçš„ï¼‰
    float edgeStrainSoftLimit = 1.5f;    // ã‚½ãƒ•ãƒˆãƒªãƒŸãƒƒãƒˆï¼ˆå¤‰å½¢ã‚’æ¸›è¡°é–‹å§‹ï¼‰
    float edgeStrainHardLimit = 2.5f;    // ãƒãƒ¼ãƒ‰ãƒªãƒŸãƒƒãƒˆï¼ˆå¤‰å½¢ã‚’å¼·ãåˆ¶é™ï¼‰
    float edgeStrainMaxLimit = 3.0f;     // æœ€å¤§ãƒªãƒŸãƒƒãƒˆï¼ˆå‰›ä½“åŒ–ï¼‰

    // ãƒœãƒªãƒ¥ãƒ¼ãƒ ã®ã‚¹ãƒˆãƒ¬ã‚¤ãƒ³é–¾å€¤ï¼ˆæ®µéšŽçš„ï¼‰
    float volStrainSoftLimit = 1.5f;     // ã‚½ãƒ•ãƒˆãƒªãƒŸãƒƒãƒˆ
    float volStrainHardLimit = 2.5f;     // ãƒãƒ¼ãƒ‰ãƒªãƒŸãƒƒãƒˆ
    float volStrainMaxLimit = 3.0f;      // æœ€å¤§ãƒªãƒŸãƒƒãƒˆ

    // ã‚¹ãƒˆãƒ¬ã‚¤ãƒ³çŠ¶æ…‹ã®è¿½è·¡
    std::vector<float> lowRes_edgeStrains;      // å„ã‚¨ãƒƒã‚¸ã®ç¾åœ¨ã®ã‚¹ãƒˆãƒ¬ã‚¤ãƒ³
    std::vector<float> lowRes_volStrains;       // å„å››é¢ä½“ã®ç¾åœ¨ã®ã‚¹ãƒˆãƒ¬ã‚¤ãƒ³
    std::vector<int> lowRes_edgeStrainLevel;    // å„ã‚¨ãƒƒã‚¸ã®ã‚¹ãƒˆãƒ¬ã‚¤ãƒ³ãƒ¬ãƒ™ãƒ«ï¼ˆ0:æ­£å¸¸, 1:ã‚½ãƒ•ãƒˆ, 2:ãƒãƒ¼ãƒ‰, 3:æœ€å¤§ï¼‰
    std::vector<int> lowRes_volStrainLevel;     // å„å››é¢ä½“ã®ã‚¹ãƒˆãƒ¬ã‚¤ãƒ³ãƒ¬ãƒ™ãƒ«

    // ã‚¹ãƒˆãƒ¬ã‚¤ãƒ³ã«ã‚ˆã‚‹å‰›æ€§èª¿æ•´ä¿‚æ•°
    std::vector<float> lowRes_edgeStiffnessScale;  // ã‚¨ãƒƒã‚¸ã”ã¨ã®å‰›æ€§ã‚¹ã‚±ãƒ¼ãƒ«
    std::vector<float> lowRes_volStiffnessScale;   // ãƒœãƒªãƒ¥ãƒ¼ãƒ ã”ã¨ã®å‰›æ€§ã‚¹ã‚±ãƒ¼ãƒ«

    // ã‚¹ãƒˆãƒ¬ã‚¤ãƒ³çµ±è¨ˆå–å¾—
    int lowResgetMaxEdgeStrainLevel() const;
    int lowResgetMaxVolStrainLevel() const;

    std::vector<int> lowResgetCriticalTetrahedra() const;  // é«˜ã‚¹ãƒˆãƒ¬ã‚¤ãƒ³ã®å››é¢ä½“ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹
private:
    void solveLowResEdgesStrainLimits(float compliance, float dt);
    void solveLowResVolumesStrainLimits(float compliance, float dt);
public:
    void computeLowResAllStrainLevels();

    int computeTopologyHops(int start, int end,
                            const std::vector<std::set<int>>& adjacency,
                            int maxHops);


    // SoftBody.h

public:
    // ã‚¹ã‚­ãƒ‹ãƒ³ã‚°èª¿æ•´ç”¨ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿
    struct SkinningAdjustmentParams {
        bool enabled = true;              // èª¿æ•´æ©Ÿèƒ½ã®æœ‰åŠ¹/ç„¡åŠ¹
        float blendFactor = 0.5f;         // è£œæ­£ã®å¼·åº¦ (0.0-1.0)
        float detectionThreshold = 0.1f;  // ç•°å¸¸æ¤œå‡ºã®é–¾å€¤
        int maxIterations = 3;            // ã‚¹ãƒ ãƒ¼ã‚¸ãƒ³ã‚°åå¾©å›žæ•°
    };

    SkinningAdjustmentParams skinningAdjustParams;

    void correctUnskinnedVerticesLowToHigh();

private:
    void detectAbnormalVerticesLowToHigh(std::vector<bool>& isAbnormal);
    void buildAdjacencyListLowToHigh(std::vector<std::vector<int>>& adjacencyList);
    void applySmoothingToVertex(int vertexIdx,
                                std::vector<float>& corrected_positions,
                                const std::vector<std::vector<int>>& adjacencyList);


    std::vector<float> original_highRes_positions;

    std::vector<std::vector<int>> cachedAdjacencyList;
    bool adjacencyListComputed = false;
public:
    void rebuildAdjacencyList() {
        adjacencyListComputed = false;
        cachedAdjacencyList.clear();
    }
    // SoftBody.h - publicã‚»ã‚¯ã‚·ãƒ§ãƒ³

public:
    // ã‚«ãƒƒãƒˆå¾Œã«ã‚¹ã‚­ãƒ‹ãƒ³ã‚°ã‚’å†è¨ˆç®—
    void recomputeSkinning();
    // public:
    void selectByOBJSegment(int objSegmentId, const std::vector<int>& nodeToOBJSegmentId);
    // public:
    void updateOBJSegmentColors(const VoxelSkeleton::VesselSegmentation& skeleton);
    void drawHighResMeshWithOBJSegments(ShaderProgram& shader);
    std::vector<float> objSegmentVertexColors_;  // OBJセグメント用の頂点色
    bool useOBJSegmentColors_ = false;

    // public:
    // public:
    void drawSmoothMeshWithOBJSegments(ShaderProgram& shader);
    // public:
    void updateOBJSegmentColorsWithSelection(const VoxelSkeleton::VesselSegmentation& skeleton, int selectedSegmentId);


    // public セクションに追加（体積計算機能）

    //--------------------------------------------------------------------------
    // 体積計算
    //--------------------------------------------------------------------------

    // 単一四面体の体積を計算
    float calculateTetVolume(int tetIdx) const;

    // 全体体積を計算
    float calculateTotalVolume() const;

    // 指定セグメントの体積を計算（OBJ/スケルトン両対応）
    float calculateSegmentVolume(int segmentId, bool useOBJSegmentation) const;

    // 選択中のセグメントの体積を計算
    float calculateSelectedVolume(bool useOBJSegmentation) const;

    // 全セグメントの体積マップを取得
    std::map<int, float> calculateAllSegmentVolumes(bool useOBJSegmentation) const;

    // 体積情報を出力
    void printVolumeInfo(bool useOBJSegmentation, int selectedOBJSegment = -1) const;

    // public セクションに追加

    // クリック位置からOBJセグメントIDを取得（頂点ベース）
    int getOBJSegmentIdAtVertex(int vertexId) const;

    // レイとメッシュの交差判定でOBJセグメントを取得
    int raycastOBJSegment(const glm::vec3& rayOrigin, const glm::vec3& rayDir,
                          glm::vec3& hitPoint) const;


    void updateHighResPositions();



    // 表面頂点キャッシュ
    std::vector<int> smoothSurfaceVertexCache;  // 表面頂点インデックスのリスト
    bool smoothSurfaceVertexCacheComputed = false;

    struct SmoothingCache {
        std::vector<std::vector<int>> neighbors;  // 隣接頂点リスト
        std::vector<int> surfaceVertexList;       // 表面頂点リスト
        bool isValid = false;
    };
    SmoothingCache smoothingCache;

    void buildSmoothingCache();
    void invalidateSmoothingCache() { smoothingCache.isValid = false; }

    // lowRes_grabId ã®è¿‘ãã«è¿½åŠ
    std::vector<int> lowRes_pinnedVertices;  // å›ºå®šã•ã‚ŒãŸé ‚ç‚¹ã®ãƒªã‚¹ãƒˆ
        // lowRes_tetValid ã®è¿‘ãã«è¿½åŠ
    std::vector<bool> lowRes_edgeValid;      // å„LowResã‚¨ãƒƒã‚¸ãŒæœ‰åŠ¹ã‹ã©ã†ã‹

    // LowResç”¨ãƒ¡ã‚½ãƒƒãƒ‰ï¼ˆè¿½åŠ ãŒå¿…è¦ï¼‰
    void updateLowResEdgeValidity();          // ã‚¨ãƒƒã‚¸æœ‰åŠ¹æ€§ã‚’æ›´æ–°

    // =====================================
    // ハンドルグループ機能
    // =====================================

    // グラブモード
    enum GrabMode {
        GRAB_NONE,
        GRAB_NORMAL,
        GRAB_HANDLE_GROUP
    };

    struct HandleGroup {
        int centerVertex;
        std::vector<int> vertices;
        std::vector<glm::vec3> relativePositions;
        glm::vec3 centerPosition;
        float radius = 0.0f;                        // ★追加
        bool isGrabbed = false;

        void storeRelativePositions(const std::vector<float>& positions);
        void updateCenterPosition(const std::vector<float>& positions);
    };

    static int MAX_HANDLE_GROUPS;
    std::vector<HandleGroup> handleGroups;
    int activeHandleGroup = -1;                     // ★名前変更（grabbedHandleGroupIndex → activeHandleGroup）
    GrabMode currentGrabMode = GRAB_NONE;           // ★追加
    glm::vec3 grabOffset = glm::vec3(0.0f);         // ★追加

    // 頂点検索
    int findClosestVertex(const glm::vec3& position);
    int findClosestSurfaceVertex(const glm::vec3& position);
    glm::vec3 getVertexPosition(int index) const;

    // ハンドルグループ操作
    bool createHandleGroupByRadius(const glm::vec3& sphereCenter, float radius);
    int findHandleGroupAtPosition(const glm::vec3& position, float threshold);
    bool tryStartGrabHandleGroup(const glm::vec3& hitPosition, float threshold);
    void startGrabHandleGroupByIndex(int groupIndex);
    void moveGrabbedHandleGroup(const glm::vec3& newPosition, const glm::vec3& velocity);
    void endGrabHandleGroup(const glm::vec3& position, const glm::vec3& velocity);
    void removeHandleGroup(int groupIndex);
    void clearHandleGroups();
    std::vector<glm::vec3> getHandleGroupPositions(int groupIndex) const;

    // スマートグラブ（ハンドルグループ優先の自動切り替え）
    void smartGrab(const glm::vec3& hitPosition, float handleThreshold);
    void smartMove(const glm::vec3& newPosition, const glm::vec3& velocity);
    void smartEndGrab(const glm::vec3& position, const glm::vec3& velocity);

    // ハンドルグループ情報取得
    int getActiveHandleGroup() const { return activeHandleGroup; }
    size_t getHandleGroupCount() const { return handleGroups.size(); }
    GrabMode getCurrentGrabMode() const { return currentGrabMode; }
    bool isHandleGroupGrabbed() const { return currentGrabMode == GRAB_HANDLE_GROUP; }
public:
    // public セクションに追加（updateOBJSegmentColors の近くに）

    void updateSegmentColorsByMode(const VoxelSkeleton::VesselSegmentation& skeleton);


public:
    // 休息状態の頂点座標を取得（Voronoiクエリ用）
    glm::vec3 getRestVertexPosition(int vertexIndex) const {
        if (!original_highRes_positions.empty() &&
            vertexIndex >= 0 &&
            static_cast<size_t>(vertexIndex * 3 + 2) < original_highRes_positions.size()) {
            return glm::vec3(
                original_highRes_positions[vertexIndex * 3],
                original_highRes_positions[vertexIndex * 3 + 1],
                original_highRes_positions[vertexIndex * 3 + 2]
                );
        }
        // フォールバック：現在位置を返す
        if (vertexIndex >= 0 &&
            static_cast<size_t>(vertexIndex * 3 + 2) < highRes_positions.size()) {
            return glm::vec3(
                highRes_positions[vertexIndex * 3],
                highRes_positions[vertexIndex * 3 + 1],
                highRes_positions[vertexIndex * 3 + 2]
                );
        }
        return glm::vec3(0.0f);
    }

    // ★★★ ここから追加 ★★★

    // Voronoi3D用
    void bindVoronoi3D(const VoxelSkeleton::VesselSegmentation& skeleton);
    void updateVoronoi3DColors(const VoxelSkeleton::VesselSegmentation& skeleton);
    void updateVoronoi3DColorsWithSelection(
        const VoxelSkeleton::VesselSegmentation& skeleton,
        int selectedBranchId);
    void updateVoronoi3DColorsWithSelection(
        const VoxelSkeleton::VesselSegmentation& skeleton,
        const std::vector<int>& selectedBranchIds);

    // ★★★ ここまで追加 ★★★
    // ★★★ カット境界頂点のスムージング強化機能 ★★★
    std::set<int> highResCutBoundaryVertices;  // カット境界の高解像度頂点
    float boundaryInfluence = 0.9f;             // 境界スムージング係数

    // publicメソッド
    void setCutBoundaryVertices(const std::set<int>& boundaryVertices);
    const std::set<int>& getCutBoundaryVertices() const;
    size_t getBoundaryVertexCount() const;
    void setBoundaryInfluence(float influence);
    float getBoundaryInfluence() const;

    //--------------------------------------------------------------------------
    // 診断・初期化ユーティリティ
    //--------------------------------------------------------------------------
public:
    //--------------------------------------------------------------------------
    // プリセット設定
    //--------------------------------------------------------------------------
    enum class MeshPreset {
        NONE,       // 設定なし（従来の動作）
        LIVER,      // 肝臓用プリセット
        VESSEL      // 血管用プリセット
    };

    struct SimulationConfig {
        // スムージング
        bool smoothEnabled = true;
        int smoothIterations = 3;
        float smoothFactor = 0.5f;
        bool boundaryPreserve = true;
        int boundaryIterations = 2;

        // ダンピング
        bool useBoundaryDamping = true;
        float motionDampingFactor = 0.95f;
        float boundaryDampingFactor = 0.80f;

        // 表示
        bool showLowHighTetMesh = true;

        // ストレインリミット
        bool enableStrainLimiting = false;
        float edgeStrainSoft = 1.5f;
        float edgeStrainHard = 3.0f;
        float edgeStrainMax = 5.0f;
        float volStrainSoft = 1.5f;
        float volStrainHard = 3.0f;
        float volStrainMax = 5.0f;

        // スキニング調整
        bool skinningEnabled = false;
        float skinningBlendFactor = 0.3f;
        int skinningMaxIterations = 1;
    };

    //--------------------------------------------------------------------------
    // コンストラクタ（MeshData版のみ残す）
    //--------------------------------------------------------------------------
    SoftBodyGPUDuo(const MeshData& lowResTetMesh,
                   const MeshData& highResTetMesh,
                   float edgeCompliance,
                   float volCompliance,
                   MeshPreset preset = MeshPreset::NONE);

    //--------------------------------------------------------------------------
    // ファクトリメソッド（パス指定はこちらを使う）
    //--------------------------------------------------------------------------
    static SoftBodyGPUDuo* createFromPaths(
        const std::string& lowResPath,
        const std::string& highResPath,
        float edgeCompliance,
        float volCompliance,
        MeshPreset preset = MeshPreset::NONE,
        SoftBodyGPUDuo* parent = nullptr);

    //--------------------------------------------------------------------------
    // 設定関連
    //--------------------------------------------------------------------------
    void applyConfig(const SimulationConfig& config);
    static SimulationConfig getPresetLiver();
    static SimulationConfig getPresetVessel();
    static FollowParams getDefaultFollowParams();

    //--------------------------------------------------------------------------
    // 統計出力
    //--------------------------------------------------------------------------
    void printCreationStats(const std::string& name = "") const;
    void printSkinningStats(const std::string& name = "") const;
    void printInvMassStats(const std::string& name = "") const;
    int computeEdgeValidity(const std::string& name = "");

    // SoftBodyGPUDuo.h に追加
    void solveStep(float dt, const glm::vec3& gravity) {
        lowResPreSolve(dt, gravity);
        lowResSolve(dt);
        lowResPostSolve(dt);
    }

    // ===== UNDO機能用メソッド =====

    // LowRes四面体の質量と拘束を再構築（UNDO後に呼ぶ）
    void rebuildLowResMassesAndConstraints();

    // 注意: updateHighResValidSurface() は既に存在します（line 366付近）
public:

    void validateAllLowResTetrahedra();
    void validateLowResTetrahedra(const std::vector<int>& tetIndices);
    // ★ 質量情報付きバージョンを追加
    void validateLowResTetrahedraWithMasses(const std::vector<int>& tetIndices,
                                            const std::vector<float>& originalInvMasses);

    //==========================================================================
    // CutSegmentMode用メソッド - カッター位置でのセグメント2色オーバーレイ
    //==========================================================================

    // スケルトンベースのセグメントオーバーレイ
    void applyCutSegmentOverlaySkeleton(
        const std::set<int>& selectedSegments,
        const VoxelSkeleton::VesselSegmentation& skeleton,
        const glm::vec4& baseColor,      // 非選択領域の色（赤っぽい）
        const glm::vec4& highlightColor  // 選択領域の色（黄色）
        );

    // Voronoi3Dベースのセグメントオーバーレイ
    void applyCutSegmentOverlayVoronoi(
        const std::vector<int>& selectedBranches,
        const VoxelSkeleton::VesselSegmentation& skeleton,
        const glm::vec4& baseColor,      // 非選択領域の色（赤っぽい）
        const glm::vec4& highlightColor  // 選択領域の色（黄色）
        );

    // デフォルトの色に戻す
    void resetToDefaultColors();


private:
    float originalTotalVolume_ = 0.0f;
    std::map<int, float> originalSegmentVolumes_;  // Skeleton用

public:
    void captureOriginalVolumes();
    float getOriginalTotalVolume() const { return originalTotalVolume_; }
    float getOriginalSegmentVolume(int segId) const;

    //==============================================================================
    // SoftBodyGPUDuo.h に追加するコード
    // 【追加場所】クラス定義の末尾、}; の直前（resetToDefaultColors(); の後）
    //
    // ★既存のVisual-only関連の宣言があれば全て削除してから追加してください
    //==============================================================================

    //==========================================================================
    // VISUAL-ONLY MODE
    //==========================================================================
public:
    bool isVisualOnlyMode = false;

    SoftBodyGPUDuo(const MeshData& visMesh, SoftBodyGPUDuo* parent,
                   const FollowParams& params = FollowParams());
    static SoftBodyGPUDuo* createVisualOnly(const std::string& visObjPath,
                                            SoftBodyGPUDuo* parent,
                                            const FollowParams& params = FollowParams());
    void updateVisualOnlyFromParent();
    void drawVisualOnlyMesh(ShaderProgram& shader);
    void drawVisualOnlyMesh(ShaderProgram& shader, const glm::vec4& color = glm::vec4(0.0f, 0.8f, 0.8f, 1.0f));
    // VisualOnly用のゲッター（CTSlicer用）
    const std::vector<float>& getVisPositions() const { return vis_positions; }
    const std::vector<int>& getVisSurfaceTriIds() const { return visSurfaceTriIds; }
private:
    std::vector<float> vis_positions;
    std::vector<float> vis_normals;
    std::vector<float> original_vis_positions;
    std::vector<float> skinningToParentLowRes;
    std::vector<int> visSurfaceTriIds;
    size_t numVisVerts = 0;
    GLuint visVAO = 0, visVBO = 0, visEBO = 0, visNormalVBO = 0;

    void setupVisualOnlyMesh();
    void computeSkinningToParentLowRes(const std::vector<float>& visVerts);
    void computeVisualOnlyNormals();
    void buildAdjacencyList();
    void applySmoothingCorrection(const std::vector<bool>& needsCorrection);

    //==============================================================================
    // 追加終了
    //==============================================================================

    // ============================================================
    // 有効サーフェスキャッシュ（INVALIDなテトラを除外したレイキャスト用）
    // ============================================================
private:
    std::vector<int> validSurfaceTriIds_;       // 有効な表面三角形インデックス
    std::vector<int> validSurfaceVertices_;     // 有効な表面頂点インデックス
    bool surfaceCacheDirty_ = true;             // キャッシュ無効フラグ

public:
    // キャッシュ再構築（カット後などに呼ぶ、または遅延評価で自動呼び出し）
    void rebuildValidSurfaceCache();

    // キャッシュを無効化（カット実行時に呼ぶ）
    void invalidateSurfaceCache() { surfaceCacheDirty_ = true; }

    // 有効な表面三角形インデックスを取得（レイキャスト用）
    const std::vector<int>& getValidSurfaceTriIds();

    // 有効な表面頂点インデックスを取得（頂点検索用）
    const std::vector<int>& getValidSurfaceVertices();

    // キャッシュ状態確認
    bool isSurfaceCacheValid() const { return !surfaceCacheDirty_; }
    // Structure to hold unstable tetrahedra information
    struct UnstableTetInfo {
        int lowResTetIdx;              // Low-res tetrahedron index
        float avgVelocityMag;          // Average velocity magnitude
        float maxVelocityMag;          // Maximum velocity magnitude
        int skinnedHighResTetCount;    // Number of skinned high-res tetrahedra (ALL in mapping)
        int validHighResTetCount;      // Number of VALID high-res tetrahedra (filtered by highResTetValid)
        bool isValid;                  // Whether the low-res tetrahedron is valid
    };

    // Detect unstable (vibrating) tetrahedra
    std::vector<UnstableTetInfo> detectUnstableTetrahedra(float velocityThreshold = 0.5f) const;

    // Print debug info for unstable tetrahedra
    void printUnstableTetrahedraDebugInfo(float velocityThreshold = 0.5f);

    // Reset velocities of unstable (vibrating) tetrahedra
    int resetUnstableTetVelocities(float velocityThreshold = 0.5f);

    // Get VALID high-res tets from low-res tet (filtered by highResTetValid)
    std::set<int> getValidHighResTetsFromLowResTet(int lowResTetIdx);


    void diagnoseVibrationCause(float velocityThreshold = 0.3f);

    // 高解像度頂点用のスムージング補正
    void applySmoothingCorrectionHighRes(const std::vector<bool>& needsCorrection);
    // 振動している低解像度四面体を無効化
    int invalidateUnstableLowResTets(float velocityThreshold = 0.5f);
    // 高解像度頂点用のスムージング補正（高速版）
    void applySmoothingCorrectionHighResFast(const std::vector<bool>& needsCorrection);


    // XPBDを使用するかスキニング補間のみを使用するかのフラグ
    bool useXPBDForFreeVertices = true;  // デフォルトはXPBD使用

    // モード切り替え
    void setXPBDMode(bool enabled) { useXPBDForFreeVertices = enabled; }
    bool isXPBDModeEnabled() const { return useXPBDForFreeVertices; }

    // スキニング補間のみで自由頂点を更新（XPBDの代替）
    void updateFreeVerticesWithSmoothing();




    // =====================================================
    // 1. SoftBodyGPUDuo.h の private セクションに追加
    // =====================================================

    // 高解像度メッシュの隣接関係キャッシュ（Laplacianスムージング用）
    std::vector<std::vector<int>> highResNeighborsCache_;
    bool highResNeighborsCacheBuilt_ = false;

    // エッジのレスト長キャッシュ
    std::vector<float> highResEdgeRestLengths_;
    bool highResEdgeRestLengthsBuilt_ = false;


    // =====================================================
    // 2. SoftBodyGPUDuo.h の public セクションに追加
    // =====================================================

    // スムージングパラメータ構造体
    struct SmoothingParams {
        int maxIterations = 3;
        float laplacianFactor = 0.5f;
        float edgeFactor = 0.3f;
        std::string presetName = "Balanced";
    };

    // スムージングパラメータ（メンバー変数）
    SmoothingParams smoothingParams_;

    // プリセット切り替え（次のプリセットに切り替えて名前を返す）
    std::string cycleSmoothingPreset();

    // 現在のプリセット名を取得
    std::string getSmoothingPresetName() const { return smoothingParams_.presetName; }

    // パラメータを個別に設定
    void setSmoothingParams(int iterations, float laplacian, float edge) {
        smoothingParams_.maxIterations = iterations;
        smoothingParams_.laplacianFactor = laplacian;
        smoothingParams_.edgeFactor = edge;
        smoothingParams_.presetName = "Custom";
    }

    // 高解像度隣接キャッシュを無効化（メッシュトポロジー変更時に呼ぶ）
    void invalidateHighResNeighborsCache() {
        highResNeighborsCacheBuilt_ = false;
        highResEdgeRestLengthsBuilt_ = false;
    }


    // スムージングデバッグ情報
    struct SmoothingDebugInfo {
        int totalVertices = 0;
        int needsCorrectionCount = 0;
        int actuallySmoothedCount = 0;
        float maxDisplacement = 0.0f;
        float avgDisplacement = 0.0f;
    };

    // デバッグ情報を取得
    SmoothingDebugInfo getLastSmoothingDebugInfo() const { return lastSmoothingDebug_; }

    // デバッグモードを有効化
    bool smoothingDebugMode_ = false;
    void setSmoothingDebugMode(bool enable) { smoothingDebugMode_ = enable; }

    // private セクションに追加
    SmoothingDebugInfo lastSmoothingDebug_;
    // スキニングのみモード用のhighRes更新
    void updateHighResPositionsOnlySkinning();
    void updateHighResPositionsSimpleSkinning();

    // private セクションに追加
    std::vector<float> lowResEdgeRestLengths_;        // エッジのレスト長
    std::vector<float> lowResTetRestVolumes_;         // 四面体のレスト体積
    bool lowResConstraintsCacheBuilt_ = false;

    // public セクションに追加
    void solveFreeVerticesWithConstraints();  // 新しい関数
    void invalidateLowResConstraintsCache() { lowResConstraintsCacheBuilt_ = false; }

    // private
    std::vector<std::vector<int>> lowResNeighborsCache_;
    bool lowResNeighborsCacheBuilt_ = false;

    void solveFreeVerticesStable();
    void solveFreeVerticesWithDamping();

    void solveFreeVerticesWithConstraintsStable();  // 方法5（新規）


    // public セクション
    void solveFreeVerticesXPBD();  // スキニング補完XPBD
    void buildLowResColoring();    // グラフカラーリング構築

    // スキニング補完XPBD用のパラメータ
    struct SkinningXPBDParams {
        int numIterations = 3;      // XPBD反復回数
        float edgeCompliance = 0.0001f;   // エッジ制約のコンプライアンス
        float volumeCompliance = 0.0001f; // 体積制約のコンプライアンス
    };
    SkinningXPBDParams skinningXPBDParams_;

    // private セクションに追加（既存のlowResConstraintsCacheBuilt_と統合）
    // グラフカラーリング（GSの並列化用）
    std::vector<std::vector<int>> lowResEdgeColorGroups_;
    std::vector<std::vector<int>> lowResTetColorGroups_;
    bool lowResColoringBuilt_ = false;

    // 裏返りチェック結果
    struct InversionCheckResult {
        int invertedCount = 0;           // 裏返り四面体数
        int totalValidTets = 0;          // 有効な四面体の総数
        std::vector<int> invertedTetIds; // 裏返り四面体のインデックス
        float minVolume = 0.0f;          // 最小体積
        float maxVolume = 0.0f;          // 最大体積
    };

    // 裏返りチェック関数
    InversionCheckResult checkLowResInversion() const;

    // 裏返りが1つ以上あるか（簡易版）
    bool hasLowResInversion() const;


public:
    // ★★★ 子メッシュカット時のアンカー管理 ★★★
    // 孤立した頂点（有効な四面体に属さない）のアンカーを解除
    // 戻り値: 解除された頂点のインデックスリスト
    std::vector<int> releaseAnchorsForOrphanedVertices();

    // 復元された四面体に基づいてアンカーを再設定（Undo用）
    // restoredTets: 復元された四面体のインデックス
    // savedAnchorState: 元のisAnchoredToParent状態
    // savedInvMasses: 元のlowRes_invMasses
    void restoreAnchorsForRestoredVertices(
        const std::vector<int>& restoredTets,
        const std::vector<bool>& savedAnchorState,
        const std::vector<float>& savedInvMasses);

    // 頂点が有効な四面体に含まれているか確認
    bool hasValidTetContainingVertex(int vertexId) const;


    std::vector<std::vector<int>> highResTetAdjacency;
    bool highResTetAdjacencyBuilt = false;

    // 隣接関係を構築（初回のみ実行、以降はキャッシュを返す）
    void buildHighResTetAdjacency();

    // 隣接関係を取得（未構築なら構築してから返す）
    const std::vector<std::vector<int>>& getHighResTetAdjacency() {
        if (!highResTetAdjacencyBuilt) {
            buildHighResTetAdjacency();
        }
        return highResTetAdjacency;
    }

    // 隣接関係が構築済みかどうか
    bool isHighResTetAdjacencyBuilt() const {
        return highResTetAdjacencyBuilt;
    }

};// SoftBody.h - publicセクション


#endif // SOFTBODYGPUDUO_H
