// CRSlicerCrossSec.h
// 四面体メッシュの断面ポリゴン計算によるスラブレンダリング
// - 空間ハッシュ vs 全スキャン 比較機能付き

#ifndef CR_SLICER_CROSS_SEC_H
#define CR_SLICER_CROSS_SEC_H

#include <vector>
#include <array>
#include <map>
#include <queue>
#include <atomic>
#include <mutex>
#include <glm/glm.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

class SoftBodyGPUDuo;

//=============================================================================
// 断面データ構造体
//=============================================================================
struct CRCrossSectionData {
    std::vector<float> vertices;
    std::vector<float> colors;
    std::vector<unsigned int> indices;

    bool empty() const { return vertices.empty(); }
    void clear() { vertices.clear(); colors.clear(); indices.clear(); }

    unsigned int addVertex(const glm::vec3& pos, const glm::vec4& col) {
        unsigned int idx = static_cast<unsigned int>(vertices.size() / 3);
        vertices.push_back(pos.x);
        vertices.push_back(pos.y);
        vertices.push_back(pos.z);
        colors.push_back(col.r);
        colors.push_back(col.g);
        colors.push_back(col.b);
        colors.push_back(col.a);
        return idx;
    }

    void addTriangle(unsigned int i0, unsigned int i1, unsigned int i2) {
        indices.push_back(i0);
        indices.push_back(i1);
        indices.push_back(i2);
    }
};

//=============================================================================
// プロファイリングデータ
//=============================================================================
struct CRSlicerProfile {
    double timeBFS[3] = {0, 0, 0};
    double timeHash[3] = {0, 0, 0};      // 空間ハッシュ構築時間
    double timeScan[3] = {0, 0, 0};      // 全スキャン時間
    double timeIntersect[3] = {0, 0, 0};
    double timeRaster[3] = {0, 0, 0};
    double timeMask[3] = {0, 0, 0};
    double timeTotal = 0;

    int numIntersectingTets[3] = {0, 0, 0};
    int numCandidateTets[3] = {0, 0, 0};
    int numTriangles[3] = {0, 0, 0};

    void reset() {
        for (int i = 0; i < 3; i++) {
            timeBFS[i] = timeHash[i] = timeScan[i] = timeIntersect[i] = timeRaster[i] = timeMask[i] = 0;
            numIntersectingTets[i] = numCandidateTets[i] = numTriangles[i] = 0;
        }
        timeTotal = 0;
    }
};

//=============================================================================
// CRSlicerCrossSec - メインクラス
//=============================================================================
class CRSlicerCrossSec {
public:
    CRSlicerCrossSec();
    ~CRSlicerCrossSec();

    // 初期化・終了
    bool initialize(GLFWwindow* parentWindow, int windowSize = 400);
    void cleanup();

    // ウィンドウ状態
    bool isWindowOpen() const { return windowsOpen_; }
    void closeWindows();

    // スライス位置 (0.0 - 1.0)
    void setSlicePosition(int axis, float t);
    float getSlicePosition(int axis) const;
    void moveSlice(int axis, float delta);

    // スラブ厚み
    void setSlabThickness(float t) { slabThickness_ = glm::clamp(t, 0.001f, 0.5f); }
    float getSlabThickness() const { return slabThickness_; }

    // サブスライス数
    void setNumSubSlices(int n) { numSubSlices_ = glm::clamp(n, 1, 20); }
    int getNumSubSlices() const { return numSubSlices_; }

    // マスク解像度
    void setMaskResolution(int res) { maskResolution_ = glm::clamp(res, 64, 1024); }
    int getMaskResolution() const { return maskResolution_; }

    // ★★★ 探索方式の切り替え ★★★
    enum class SearchMode {
        FULL_SCAN,      // 全スキャン + BFS
        SPATIAL_HASH    // 空間ハッシュ + BFS
    };
    void setSearchMode(SearchMode mode);
    SearchMode getSearchMode() const { return searchMode_; }
    void cycleSearchMode();
    const char* getSearchModeName() const;

    // 空間ハッシュのセル数（SPATIAL_HASHモード用）
    void setHashGridSize(int size) { hashGridSize_ = glm::clamp(size, 16, 256); }
    int getHashGridSize() const { return hashGridSize_; }

    // スムースメッシュを使用するか
    void setUseSmoothMesh(bool use) { useSmoothMesh_ = use; }
    bool getUseSmoothMesh() const { return useSmoothMesh_; }

    // 更新頻度
    void setUpdateInterval(int interval) { updateInterval_ = std::max(1, interval); }
    int getUpdateInterval() const { return updateInterval_; }

    // 強制更新フラグ
    void forceUpdate() { forceUpdate_ = true; }

    // 色モード設定
    enum class ColorMode {
        BASE_COLOR,
        OBJ_SEGMENT,
        SKELETON
    };
    void setColorMode(ColorMode mode) { colorMode_ = mode; }
    ColorMode getColorMode() const { return colorMode_; }

    // メッシュデータ設定
    void setSoftBodies(
        const std::vector<SoftBodyGPUDuo*>& softBodies,
        const std::vector<glm::vec4>& colors);

    // 更新・描画
    void update();

    // バウンディングボックス
    glm::vec3 getBoundsMin() const { return boundsMin_; }
    glm::vec3 getBoundsMax() const { return boundsMax_; }

    // プロファイリング
    void setProfilingEnabled(bool enabled) { profilingEnabled_ = enabled; }
    bool isProfilingEnabled() const { return profilingEnabled_; }
    const CRSlicerProfile& getLastProfile() const { return lastProfile_; }

    // 並列化オプション
    enum class ParallelMode {
        SERIAL,
        PARALLEL_AXES,
        PARALLEL_FLAT,
        PARALLEL_NESTED
    };
    void setParallelMode(ParallelMode mode);
    ParallelMode getParallelMode() const { return parallelMode_; }
    void cycleParallelMode();
    const char* getParallelModeName() const;

    void setDebugTiming(bool enabled) { debugTiming_ = enabled; }
    bool isDebugTiming() const { return debugTiming_; }

    //=========================================================================
    // CTSlicer互換機能
    //=========================================================================
    void setSliceAxisFromCamera(const glm::vec3& cameraPos,
                                const glm::vec3& cameraTarget,
                                const glm::mat4& viewMatrix);

    void cycleAxisOffset();
    void resetAxisOffset() {
        axisOffset_ = 0;
        currentAxis_ = autoSelectedAxis_;
    }

    void togglePreviewLock();
    bool isPreviewLocked() const { return previewLocked_; }

    int getCurrentAxis() const { return currentAxis_; }
    void setCurrentAxis(int axis) { currentAxis_ = glm::clamp(axis, 0, 2); }

    const char* getAxisName() const {
        static const char* names[] = {"X (Sagittal)", "Y (Axial)", "Z (Coronal)"};
        return names[currentAxis_];
    }

    void moveSlicePosition(float delta) {
        moveSlice(currentAxis_, delta);
    }

    void alignSlicesToPosition(const glm::vec3& worldPos);
    void alignSlicesToCutterCenter(const std::vector<float>* cutterVertices);

    //=========================================================================
    // 3D描画機能
    //=========================================================================
    void drawBoundingBox(const glm::mat4& viewMatrix,
                         const glm::mat4& projMatrix,
                         const glm::vec4& color = glm::vec4(1.0f, 1.0f, 1.0f, 0.4f));

    void drawSlicePlanes(const glm::mat4& viewMatrix,
                         const glm::mat4& projMatrix);

    //=========================================================================
    // ウィンドウ管理
    //=========================================================================
    int getWindowUnderMouse() const;
    GLFWwindow* getWindow(int axis) const {
        if (axis >= 0 && axis < 3) return windows_[axis];
        return nullptr;
    }

    void moveSliceX(float delta) { moveSlice(0, delta); }
    void moveSliceY(float delta) { moveSlice(1, delta); }
    void moveSliceZ(float delta) { moveSlice(2, delta); }

    //=========================================================================
    // カッター断面表示
    //=========================================================================
    void setCutterMesh(const std::vector<float>* vertices,
                       const std::vector<unsigned int>* indices,
                       const glm::vec4& color = glm::vec4(0.9f, 0.7f, 0.2f, 0.8f));
    void clearCutterMesh();
    void setCutterColor(const glm::vec4& color) { cutterColor_ = color; }
    const glm::vec4& getCutterColor() const { return cutterColor_; }

    void setCameraDirection(const glm::vec3& direction) { cameraDirection_ = direction; }
    void setOrbitCameraUp(const glm::vec3& up) { orbitCameraUp_ = up; }

    //=========================================================================
    // MaskBuffer
    //=========================================================================
    struct MaskBuffer {
        std::vector<uint8_t> mask;
        std::vector<glm::vec4> colorSum;
        std::vector<int> colorCount;
        int width = 0;
        int height = 0;

        void resize(int w, int h) {
            width = w;
            height = h;
            mask.assign(w * h, 0);
            colorSum.assign(w * h, glm::vec4(0.0f));
            colorCount.assign(w * h, 0);
        }

        void clear() {
            std::fill(mask.begin(), mask.end(), 0);
            std::fill(colorSum.begin(), colorSum.end(), glm::vec4(0.0f));
            std::fill(colorCount.begin(), colorCount.end(), 0);
        }
    };

private:
    // ウィンドウ
    GLFWwindow* parentWindow_ = nullptr;
    std::array<GLFWwindow*, 3> windows_ = {nullptr, nullptr, nullptr};
    bool windowsOpen_ = false;
    int windowSize_ = 400;

    // カメラ方向
    glm::vec3 cameraDirection_ = glm::vec3(0, 0, 1);
    glm::vec3 orbitCameraUp_ = glm::vec3(0, 1, 0);

    // 3D描画用リソース
    GLuint previewShaderProgram_ = 0;
    bool createPreviewShader();
    void deletePreviewShader();

    // カッター断面用
    const std::vector<float>* cutterVertices_ = nullptr;
    const std::vector<unsigned int>* cutterIndices_ = nullptr;
    glm::vec4 cutterColor_ = glm::vec4(0.9f, 0.7f, 0.2f, 0.8f);

    std::vector<glm::vec3> computeCutterIntersection(int axis, float planePos) const;
    void sortPointsByAngle(std::vector<glm::vec3>& points, int axis) const;
    void rasterizeCutterToMask(MaskBuffer& mask,
                               const std::vector<glm::vec3>& points,
                               int axis) const;

    // スライス設定
    std::array<float, 3> slicePos_ = {0.5f, 0.5f, 0.5f};
    float slabThickness_ = 0.05f;
    int numSubSlices_ = 5;
    int maskResolution_ = 512;
    int hashGridSize_ = 64;

    // 探索方式
    SearchMode searchMode_ = SearchMode::FULL_SCAN;

    // 色モード
    ColorMode colorMode_ = ColorMode::BASE_COLOR;
    bool useSmoothMesh_ = true;

    // 更新制御
    int updateInterval_ = 1;
    int frameCounter_ = 0;
    std::array<float, 3> lastSlicePos_ = {-1.0f, -1.0f, -1.0f};
    float lastSlabThickness_ = -1.0f;
    bool forceUpdate_ = true;

    // CTSlicer互換機能
    bool previewLocked_ = false;
    int autoSelectedAxis_ = 1;
    int axisOffset_ = 0;
    int currentAxis_ = 1;

    // カメラ情報
    glm::vec3 cameraPos_{0.0f};
    glm::vec3 cameraTarget_{0.0f};
    glm::vec3 cameraUp_{0.0f, 1.0f, 0.0f};
    glm::vec3 cameraRight_{1.0f, 0.0f, 0.0f};
    glm::mat4 viewMatrix_{1.0f};

    //=========================================================================
    // 四面体隣接グラフ（初期化時1回のみ構築）
    //=========================================================================
    struct TetAdjacency {
        std::vector<std::vector<int>> neighbors;
        bool computed = false;
    };
    std::map<SoftBodyGPUDuo*, TetAdjacency> tetAdjacencyCache_;

    void buildTetAdjacency(SoftBodyGPUDuo* body);

    //=========================================================================
    // 空間ハッシュ（SPATIAL_HASHモード用）
    //=========================================================================
    struct SpatialHash {
        std::vector<std::vector<int>> cells;
        float minCoord = 0.0f;
        float maxCoord = 1.0f;
        float cellSize = 0.01f;
        int numCells = 100;
        bool valid = false;

        void clear() {
            for (auto& cell : cells) cell.clear();
            valid = false;
        }

        void resize(int n) {
            numCells = n;
            cells.resize(n);
            clear();
        }

        int getCellIndex(float coord) const {
            if (cellSize <= 0.0f) return 0;
            int idx = static_cast<int>((coord - minCoord) / cellSize);
            return glm::clamp(idx, 0, numCells - 1);
        }
    };

    struct MeshSpatialHash {
        std::array<SpatialHash, 3> axisHash;
    };
    std::map<SoftBodyGPUDuo*, MeshSpatialHash> spatialHashCache_;

    void buildSpatialHash(SoftBodyGPUDuo* body,
                          const std::vector<float>& positions,
                          const std::vector<int>& tetIds,
                          int axis);

    std::vector<int> getCandidatesFromHash(SoftBodyGPUDuo* body,
                                           int axis,
                                           float slabMin,
                                           float slabMax) const;

    //=========================================================================
    // 探索関数（両方式）
    //=========================================================================
    // 全スキャン + BFS
    std::vector<int> findIntersectingTetsFullScan(
        SoftBodyGPUDuo* body,
        const std::vector<float>& positions,
        const std::vector<int>& tetIds,
        int axis, float slabMin, float slabMax);

    // 空間ハッシュ + BFS
    std::vector<int> findIntersectingTetsSpatialHash(
        SoftBodyGPUDuo* body,
        const std::vector<float>& positions,
        const std::vector<int>& tetIds,
        int axis, float slabMin, float slabMax);

    // バウンディングボックス
    glm::vec3 boundsMin_{0.0f};
    glm::vec3 boundsMax_{0.0f};

    // メッシュデータ
    struct MeshEntry {
        SoftBodyGPUDuo* body = nullptr;
        glm::vec4 color{1.0f};
    };
    std::vector<MeshEntry> meshes_;

    //=========================================================================
    // 事前計算されたメッシュデータ
    //=========================================================================
    struct PreparedMeshData {
        SoftBodyGPUDuo* body = nullptr;
        const std::vector<float>* positions = nullptr;
        const std::vector<int>* tetIds = nullptr;
        std::vector<float> vertColors;
        int colorComponents = 3;
        bool hasColors = false;
        glm::vec4 color{1.0f};
        int numVerts = 0;
        int numTets = 0;
        std::array<std::vector<int>, 3> intersectingTets;
    };

    // 断面データキャッシュ
    std::array<std::vector<CRCrossSectionData>, 3> crossSections_;

    // スレッドローカルMaskBufferプール
    std::array<MaskBuffer, 3> threadLocalMasks_;
    bool maskPoolInitialized_ = false;
    void initializeMaskPool();

    // OpenGLリソース
    std::array<GLuint, 3> shaderPrograms_ = {0, 0, 0};
    std::array<GLuint, 3> vaos_ = {0, 0, 0};
    std::array<GLuint, 3> vbos_ = {0, 0, 0};
    std::array<GLuint, 3> cbos_ = {0, 0, 0};
    std::array<GLuint, 3> ebos_ = {0, 0, 0};

    bool createShaderForWindow(int idx);
    void createBuffersForWindow(int idx);
    void cleanupWindowResources(int idx);
    void renderAxisWindow(int axis);

    // 計算関連
    void updateBoundingBox();
    void computeAllCrossSections();

    //=========================================================================
    // モード別処理関数
    //=========================================================================
    void computeCrossSectionsSerial(
        std::vector<PreparedMeshData>& meshDataList,
        const std::array<float, 3>& slabMinArr,
        const std::array<float, 3>& slabMaxArr,
        const std::array<float, 3>& centerPosArr);

    void computeCrossSectionsParallelAxes(
        std::vector<PreparedMeshData>& meshDataList,
        const std::array<float, 3>& slabMinArr,
        const std::array<float, 3>& slabMaxArr,
        const std::array<float, 3>& centerPosArr);

    void computeCrossSectionsParallelFlat(
        std::vector<PreparedMeshData>& meshDataList,
        const std::array<float, 3>& slabMinArr,
        const std::array<float, 3>& slabMaxArr,
        const std::array<float, 3>& centerPosArr);

    void computeCrossSectionsParallelNested(
        std::vector<PreparedMeshData>& meshDataList,
        const std::array<float, 3>& slabMinArr,
        const std::array<float, 3>& slabMaxArr,
        const std::array<float, 3>& centerPosArr);

    void computeAxisSubSlices(
        int axis,
        const PreparedMeshData& meshData,
        float slabMin, float slabMax, float centerPos,
        MaskBuffer& mask,
        std::vector<CRCrossSectionData>& results);

    CRCrossSectionData computeCrossSectionOptimized(
        const std::vector<float>& positions,
        const std::vector<int>& tetIds,
        const std::vector<int>& intersectingTets,
        int axis, float planePos, int numVerts,
        const std::vector<float>& vertexColors,
        int colorComponents, bool hasColors,
        const glm::vec4& baseColor) const;

    CRCrossSectionData generatePolygonFromMaskForAxis(
        const MaskBuffer& mask,
        int axis,
        const glm::vec3& boundsMin,
        const glm::vec3& boundsMax,
        float alpha,
        float slicePosNormalized) const;

    void triangulatePolygonLocal(
        CRCrossSectionData& data,
        const glm::vec3* verts,
        const glm::vec4* colors,
        int vertCount) const;

    void rasterizeCrossSectionToMaskOptimized(
        MaskBuffer& mask,
        const CRCrossSectionData& cs,
        int axis,
        const glm::vec3& boundsMin,
        const glm::vec3& boundsMax) const;

    int intersectTetrahedronWithPlane(
        const glm::vec3 tetVerts[4],
        const glm::vec4 tetColors[4],
        int axis, float planePos,
        glm::vec3 outVerts[4],
        glm::vec4 outColors[4]) const;

    // プロファイリング
    bool profilingEnabled_ = true;
    CRSlicerProfile lastProfile_;
    std::mutex profileMutex_;

    struct AtomicProfile {
        std::atomic<double> timeBFS[3] = {{0}, {0}, {0}};
        std::atomic<double> timeHash[3] = {{0}, {0}, {0}};
        std::atomic<double> timeScan[3] = {{0}, {0}, {0}};
        std::atomic<int> numIntersectingTets[3] = {{0}, {0}, {0}};
        std::atomic<int> numCandidateTets[3] = {{0}, {0}, {0}};

        void reset() {
            for (int i = 0; i < 3; i++) {
                timeBFS[i].store(0);
                timeHash[i].store(0);
                timeScan[i].store(0);
                numIntersectingTets[i].store(0);
                numCandidateTets[i].store(0);
            }
        }

        void copyTo(CRSlicerProfile& profile) {
            for (int i = 0; i < 3; i++) {
                profile.timeBFS[i] = timeBFS[i].load();
                profile.timeHash[i] = timeHash[i].load();
                profile.timeScan[i] = timeScan[i].load();
                profile.numIntersectingTets[i] = numIntersectingTets[i].load();
                profile.numCandidateTets[i] = numCandidateTets[i].load();
            }
        }
    };
    AtomicProfile atomicProfile_;

    bool debugTiming_ = false;
    ParallelMode parallelMode_ = ParallelMode::PARALLEL_AXES;

    void printProfilingReport();
    int profileCounter_ = 0;
    CRSlicerProfile sumProfile_;
};

#endif // CR_SLICER_CROSS_SEC_H
