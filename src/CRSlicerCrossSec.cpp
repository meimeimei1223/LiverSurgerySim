// CRSlicerCrossSec.cpp - Part 1
// 空間ハッシュ vs 全スキャン 比較機能付き

#include "CRSlicerCrossSec.h"
#include "SoftBodyGPUDuo.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

//=============================================================================
// シェーダーソース
//=============================================================================
static const char* vertexShaderSrc = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;
uniform mat4 uMVP;
out vec4 vColor;
void main() {
    vColor = aColor;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
)";

static const char* fragmentShaderSrc = R"(
#version 330 core
in vec4 vColor;
out vec4 FragColor;
void main() {
    FragColor = vColor;
}
)";

static const char* previewVertexShaderSrc = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
}
)";

static const char* previewFragmentShaderSrc = R"(
#version 330 core
uniform vec4 uColor;
out vec4 FragColor;
void main() {
    FragColor = uColor;
}
)";

//=============================================================================
// コンストラクタ・デストラクタ
//=============================================================================
CRSlicerCrossSec::CRSlicerCrossSec() = default;
CRSlicerCrossSec::~CRSlicerCrossSec() { cleanup(); }

//=============================================================================
// 初期化・終了
//=============================================================================
bool CRSlicerCrossSec::initialize(GLFWwindow* parentWindow, int windowSize) {
    if (windowsOpen_) return true;
    parentWindow_ = parentWindow;
    windowSize_ = windowSize;
    GLFWwindow* mainCtx = glfwGetCurrentContext();

    const char* titles[3] = {"CR Slab X (YZ)", "CR Slab Y (XZ)", "CR Slab Z (XY)"};
    for (int i = 0; i < 3; i++) {
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);

        windows_[i] = glfwCreateWindow(windowSize, windowSize, titles[i], nullptr, parentWindow);
        if (!windows_[i]) {
            for (int j = 0; j < i; j++) { if (windows_[j]) { glfwDestroyWindow(windows_[j]); windows_[j] = nullptr; } }
            glfwMakeContextCurrent(mainCtx);
            return false;
        }
        glfwSetWindowPos(windows_[i], 50 + i * (windowSize + 20), 100);
        glfwMakeContextCurrent(windows_[i]);
        if (!createShaderForWindow(i)) {
            for (int j = 0; j <= i; j++) { if (windows_[j]) { glfwDestroyWindow(windows_[j]); windows_[j] = nullptr; } }
            glfwMakeContextCurrent(mainCtx);
            return false;
        }
        createBuffersForWindow(i);
    }

    glfwMakeContextCurrent(mainCtx);
    initializeMaskPool();
    windowsOpen_ = true;

#ifdef _OPENMP
    std::cout << "[CRSlicer] Initialized (threads=" << omp_get_max_threads() << ", " << getSearchModeName() << ")" << std::endl;
#else
    std::cout << "[CRSlicer] Initialized (serial, " << getSearchModeName() << ")" << std::endl;
#endif
    return true;
}

void CRSlicerCrossSec::initializeMaskPool() {
    if (maskPoolInitialized_) return;
    for (int i = 0; i < 3; i++) threadLocalMasks_[i].resize(maskResolution_, maskResolution_);
    maskPoolInitialized_ = true;
}

void CRSlicerCrossSec::cleanup() {
    closeWindows();
    deletePreviewShader();
    meshes_.clear();
    for (auto& cs : crossSections_) cs.clear();
    tetAdjacencyCache_.clear();
    spatialHashCache_.clear();
}

void CRSlicerCrossSec::closeWindows() {
    GLFWwindow* mainCtx = glfwGetCurrentContext();
    for (int i = 0; i < 3; i++) {
        if (windows_[i]) {
            glfwMakeContextCurrent(windows_[i]);
            cleanupWindowResources(i);
            glfwDestroyWindow(windows_[i]);
            windows_[i] = nullptr;
        }
    }
    glfwMakeContextCurrent(mainCtx);
    windowsOpen_ = false;
}

//=============================================================================
// シェーダー・バッファ管理
//=============================================================================
bool CRSlicerCrossSec::createShaderForWindow(int idx) {
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertexShaderSrc, nullptr);
    glCompileShader(vs);
    GLint success;
    glGetShaderiv(vs, GL_COMPILE_STATUS, &success);
    if (!success) { glDeleteShader(vs); return false; }

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragmentShaderSrc, nullptr);
    glCompileShader(fs);
    glGetShaderiv(fs, GL_COMPILE_STATUS, &success);
    if (!success) { glDeleteShader(vs); glDeleteShader(fs); return false; }

    shaderPrograms_[idx] = glCreateProgram();
    glAttachShader(shaderPrograms_[idx], vs);
    glAttachShader(shaderPrograms_[idx], fs);
    glLinkProgram(shaderPrograms_[idx]);
    glGetProgramiv(shaderPrograms_[idx], GL_LINK_STATUS, &success);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return success != 0;
}

void CRSlicerCrossSec::createBuffersForWindow(int idx) {
    glGenVertexArrays(1, &vaos_[idx]);
    glGenBuffers(1, &vbos_[idx]);
    glGenBuffers(1, &cbos_[idx]);
    glGenBuffers(1, &ebos_[idx]);
}

void CRSlicerCrossSec::cleanupWindowResources(int idx) {
    if (vaos_[idx]) { glDeleteVertexArrays(1, &vaos_[idx]); vaos_[idx] = 0; }
    if (vbos_[idx]) { glDeleteBuffers(1, &vbos_[idx]); vbos_[idx] = 0; }
    if (cbos_[idx]) { glDeleteBuffers(1, &cbos_[idx]); cbos_[idx] = 0; }
    if (ebos_[idx]) { glDeleteBuffers(1, &ebos_[idx]); ebos_[idx] = 0; }
    if (shaderPrograms_[idx]) { glDeleteProgram(shaderPrograms_[idx]); shaderPrograms_[idx] = 0; }
}

bool CRSlicerCrossSec::createPreviewShader() {
    if (previewShaderProgram_ != 0) return true;
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &previewVertexShaderSrc, nullptr);
    glCompileShader(vs);
    GLint success;
    glGetShaderiv(vs, GL_COMPILE_STATUS, &success);
    if (!success) { glDeleteShader(vs); return false; }

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &previewFragmentShaderSrc, nullptr);
    glCompileShader(fs);
    glGetShaderiv(fs, GL_COMPILE_STATUS, &success);
    if (!success) { glDeleteShader(vs); glDeleteShader(fs); return false; }

    previewShaderProgram_ = glCreateProgram();
    glAttachShader(previewShaderProgram_, vs);
    glAttachShader(previewShaderProgram_, fs);
    glLinkProgram(previewShaderProgram_);
    glDeleteShader(vs);
    glDeleteShader(fs);
    glGetProgramiv(previewShaderProgram_, GL_LINK_STATUS, &success);
    return success != 0;
}

void CRSlicerCrossSec::deletePreviewShader() {
    if (previewShaderProgram_) { glDeleteProgram(previewShaderProgram_); previewShaderProgram_ = 0; }
}

//=============================================================================
// スライス位置管理
//=============================================================================
void CRSlicerCrossSec::setSlicePosition(int axis, float t) {
    if (axis >= 0 && axis < 3) slicePos_[axis] = glm::clamp(t, 0.0f, 1.0f);
}
float CRSlicerCrossSec::getSlicePosition(int axis) const {
    return (axis >= 0 && axis < 3) ? slicePos_[axis] : 0.5f;
}
void CRSlicerCrossSec::moveSlice(int axis, float delta) {
    if (axis >= 0 && axis < 3) slicePos_[axis] = glm::clamp(slicePos_[axis] + delta, 0.0f, 1.0f);
}

//=============================================================================
// 探索モード設定
//=============================================================================
void CRSlicerCrossSec::setSearchMode(SearchMode mode) {
    searchMode_ = mode;
    std::cout << "[CRSlicer] Search mode: " << getSearchModeName() << std::endl;
}

void CRSlicerCrossSec::cycleSearchMode() {
    setSearchMode(searchMode_ == SearchMode::FULL_SCAN ? SearchMode::SPATIAL_HASH : SearchMode::FULL_SCAN);
}

const char* CRSlicerCrossSec::getSearchModeName() const {
    return searchMode_ == SearchMode::FULL_SCAN ? "FULL_SCAN" : "SPATIAL_HASH";
}

//=============================================================================
// 並列モード設定
//=============================================================================
void CRSlicerCrossSec::setParallelMode(ParallelMode mode) {
    parallelMode_ = mode;
    std::cout << "[CRSlicer] Parallel mode: " << getParallelModeName() << std::endl;
}

void CRSlicerCrossSec::cycleParallelMode() {
    parallelMode_ = static_cast<ParallelMode>((static_cast<int>(parallelMode_) + 1) % 4);
    std::cout << "[CRSlicer] Parallel mode: " << getParallelModeName() << std::endl;
}

const char* CRSlicerCrossSec::getParallelModeName() const {
    switch (parallelMode_) {
    case ParallelMode::SERIAL: return "SERIAL";
    case ParallelMode::PARALLEL_AXES: return "PARALLEL_AXES";
    case ParallelMode::PARALLEL_FLAT: return "PARALLEL_FLAT";
    case ParallelMode::PARALLEL_NESTED: return "PARALLEL_NESTED";
    default: return "UNKNOWN";
    }
}

//=============================================================================
// メッシュデータ設定
//=============================================================================
void CRSlicerCrossSec::setSoftBodies(const std::vector<SoftBodyGPUDuo*>& softBodies, const std::vector<glm::vec4>& colors) {
    meshes_.clear();
    for (size_t i = 0; i < softBodies.size(); i++) {
        if (!softBodies[i]) continue;
        MeshEntry e;
        e.body = softBodies[i];
        e.color = (i < colors.size()) ? colors[i] : glm::vec4(0.8f, 0.3f, 0.3f, 1.0f);
        meshes_.push_back(e);
    }
    updateBoundingBox();
    for (const auto& m : meshes_) {
        if (m.body) buildTetAdjacency(m.body);
    }
}

void CRSlicerCrossSec::updateBoundingBox() {
    boundsMin_ = glm::vec3(FLT_MAX);
    boundsMax_ = glm::vec3(-FLT_MAX);
    for (const auto& m : meshes_) {
        if (!m.body) continue;
        const auto* pos = &m.body->highRes_positions;
        if (pos->empty()) continue;
        for (size_t i = 0; i < pos->size(); i += 3) {
            glm::vec3 p((*pos)[i], (*pos)[i+1], (*pos)[i+2]);
            boundsMin_ = glm::min(boundsMin_, p);
            boundsMax_ = glm::max(boundsMax_, p);
        }
    }
}

//=============================================================================
// 四面体隣接グラフ構築
//=============================================================================
void CRSlicerCrossSec::buildTetAdjacency(SoftBodyGPUDuo* body) {
    if (!body) return;
    if (tetAdjacencyCache_.count(body) && tetAdjacencyCache_[body].computed) return;

    const auto& tetIds = body->highResMeshData.tetIds;
    int numTets = static_cast<int>(tetIds.size() / 4);

    TetAdjacency& adj = tetAdjacencyCache_[body];
    adj.neighbors.resize(numTets);

    std::map<std::tuple<int,int,int>, int> faceToTet;
    static const int faceIndices[4][3] = {{0,1,2}, {0,1,3}, {0,2,3}, {1,2,3}};

    for (int t = 0; t < numTets; t++) {
        int v[4] = {tetIds[t*4+0], tetIds[t*4+1], tetIds[t*4+2], tetIds[t*4+3]};
        for (int f = 0; f < 4; f++) {
            int fv[3] = {v[faceIndices[f][0]], v[faceIndices[f][1]], v[faceIndices[f][2]]};
            std::sort(fv, fv + 3);
            auto key = std::make_tuple(fv[0], fv[1], fv[2]);
            auto it = faceToTet.find(key);
            if (it != faceToTet.end()) {
                adj.neighbors[t].push_back(it->second);
                adj.neighbors[it->second].push_back(t);
            } else {
                faceToTet[key] = t;
            }
        }
    }
    adj.computed = true;
    std::cout << "[CRSlicer] Built tet adjacency (" << numTets << " tets)" << std::endl;
}

//=============================================================================
// 空間ハッシュ構築
//=============================================================================
void CRSlicerCrossSec::buildSpatialHash(SoftBodyGPUDuo* body, const std::vector<float>& positions,
                                        const std::vector<int>& tetIds, int axis) {
    int numVerts = static_cast<int>(positions.size() / 3);
    int numTets = static_cast<int>(tetIds.size() / 4);

    MeshSpatialHash& meshHash = spatialHashCache_[body];
    SpatialHash& hash = meshHash.axisHash[axis];

    if ((int)hash.cells.size() != hashGridSize_) hash.resize(hashGridSize_);
    else hash.clear();

    float minC = FLT_MAX, maxC = -FLT_MAX;
    for (size_t i = axis; i < positions.size(); i += 3) {
        minC = std::min(minC, positions[i]);
        maxC = std::max(maxC, positions[i]);
    }

    hash.minCoord = minC;
    hash.maxCoord = maxC;
    hash.cellSize = (maxC - minC) / hash.numCells;
    if (hash.cellSize <= 0.0f) hash.cellSize = 1.0f;

    for (int t = 0; t < numTets; t++) {
        int i0 = tetIds[t*4+0], i1 = tetIds[t*4+1], i2 = tetIds[t*4+2], i3 = tetIds[t*4+3];
        if (i0 < 0 || i0 >= numVerts || i1 < 0 || i1 >= numVerts ||
            i2 < 0 || i2 >= numVerts || i3 < 0 || i3 >= numVerts) continue;

        float c0 = positions[i0*3+axis], c1 = positions[i1*3+axis];
        float c2 = positions[i2*3+axis], c3 = positions[i3*3+axis];
        float tetMin = std::min({c0, c1, c2, c3});
        float tetMax = std::max({c0, c1, c2, c3});

        int cellStart = hash.getCellIndex(tetMin);
        int cellEnd = hash.getCellIndex(tetMax);
        for (int c = cellStart; c <= cellEnd; c++) hash.cells[c].push_back(t);
    }
    hash.valid = true;
}

std::vector<int> CRSlicerCrossSec::getCandidatesFromHash(SoftBodyGPUDuo* body, int axis,
                                                         float slabMin, float slabMax) const {
    std::vector<int> candidates;
    auto it = spatialHashCache_.find(body);
    if (it == spatialHashCache_.end()) return candidates;

    const SpatialHash& hash = it->second.axisHash[axis];
    if (!hash.valid) return candidates;

    int cellStart = hash.getCellIndex(slabMin);
    int cellEnd = hash.getCellIndex(slabMax);

    int maxTetId = 0;
    for (int c = cellStart; c <= cellEnd; c++) {
        for (int tetId : hash.cells[c]) maxTetId = std::max(maxTetId, tetId);
    }
    std::vector<bool> added(maxTetId + 1, false);

    for (int c = cellStart; c <= cellEnd; c++) {
        for (int tetId : hash.cells[c]) {
            if (!added[tetId]) { added[tetId] = true; candidates.push_back(tetId); }
        }
    }
    return candidates;
}

//=============================================================================
// ★★★ 全スキャン + BFS ★★★
//=============================================================================
std::vector<int> CRSlicerCrossSec::findIntersectingTetsFullScan(
    SoftBodyGPUDuo* body, const std::vector<float>& positions,
    const std::vector<int>& tetIds, int axis, float slabMin, float slabMax)
{
    int numVerts = static_cast<int>(positions.size() / 3);
    int numTets = static_cast<int>(tetIds.size() / 4);

    auto scanStart = std::chrono::high_resolution_clock::now();

    // 全スキャンでスラブ内判定
    std::vector<bool> inSlab(numTets, false);
    for (int t = 0; t < numTets; t++) {
        int i0 = tetIds[t*4+0], i1 = tetIds[t*4+1], i2 = tetIds[t*4+2], i3 = tetIds[t*4+3];
        if (i0 < 0 || i0 >= numVerts || i1 < 0 || i1 >= numVerts ||
            i2 < 0 || i2 >= numVerts || i3 < 0 || i3 >= numVerts) continue;

        float c0 = positions[i0*3+axis], c1 = positions[i1*3+axis];
        float c2 = positions[i2*3+axis], c3 = positions[i3*3+axis];
        float tetMin = std::min({c0, c1, c2, c3});
        float tetMax = std::max({c0, c1, c2, c3});

        if (tetMax >= slabMin && tetMin <= slabMax) inSlab[t] = true;
    }

    auto scanEnd = std::chrono::high_resolution_clock::now();
    double scanTime = std::chrono::duration<double, std::milli>(scanEnd - scanStart).count();
    double old = atomicProfile_.timeScan[axis].load();
    while (!atomicProfile_.timeScan[axis].compare_exchange_weak(old, old + scanTime));

    // VALID判定
    const std::vector<bool>* tetValid = nullptr;
    if (body && !body->highResTetValid.empty()) tetValid = &body->highResTetValid;

    // 隣接グラフ
    auto adjIt = tetAdjacencyCache_.find(body);
    if (adjIt == tetAdjacencyCache_.end() || !adjIt->second.computed) {
        std::vector<int> result;
        for (int t = 0; t < numTets; t++) {
            if (!inSlab[t]) continue;
            bool isValid = !tetValid || t >= (int)tetValid->size() || (*tetValid)[t];
            if (isValid) result.push_back(t);
        }
        return result;
    }

    // BFS（複数連結成分対応）
    const TetAdjacency& adj = adjIt->second;
    std::vector<int> result;
    std::vector<bool> visited(numTets, false);

    for (int startTet = 0; startTet < numTets; startTet++) {
        if (!inSlab[startTet] || visited[startTet]) continue;

        std::queue<int> queue;
        queue.push(startTet);
        visited[startTet] = true;

        while (!queue.empty()) {
            int t = queue.front();
            queue.pop();

            if (inSlab[t]) {
                bool isValid = !tetValid || t >= (int)tetValid->size() || (*tetValid)[t];
                if (isValid) result.push_back(t);

                for (int neighbor : adj.neighbors[t]) {
                    if (!visited[neighbor] && neighbor < numTets) {
                        visited[neighbor] = true;
                        if (inSlab[neighbor]) queue.push(neighbor);
                    }
                }
            }
        }
    }

    return result;
}

//=============================================================================
// ★★★ 空間ハッシュ + BFS ★★★
//=============================================================================
std::vector<int> CRSlicerCrossSec::findIntersectingTetsSpatialHash(
    SoftBodyGPUDuo* body, const std::vector<float>& positions,
    const std::vector<int>& tetIds, int axis, float slabMin, float slabMax)
{
    int numVerts = static_cast<int>(positions.size() / 3);
    int numTets = static_cast<int>(tetIds.size() / 4);

    // 空間ハッシュ構築
    auto hashStart = std::chrono::high_resolution_clock::now();
    buildSpatialHash(body, positions, tetIds, axis);
    auto hashEnd = std::chrono::high_resolution_clock::now();
    double hashTime = std::chrono::duration<double, std::milli>(hashEnd - hashStart).count();
    double old = atomicProfile_.timeHash[axis].load();
    while (!atomicProfile_.timeHash[axis].compare_exchange_weak(old, old + hashTime));

    // 候補取得
    std::vector<int> candidates = getCandidatesFromHash(body, axis, slabMin, slabMax);
    atomicProfile_.numCandidateTets[axis].fetch_add(static_cast<int>(candidates.size()));

    if (candidates.empty()) return {};

    // 候補内でスラブ判定
    std::vector<bool> inSlab(numTets, false);
    for (int t : candidates) {
        int i0 = tetIds[t*4+0], i1 = tetIds[t*4+1], i2 = tetIds[t*4+2], i3 = tetIds[t*4+3];
        if (i0 < 0 || i0 >= numVerts || i1 < 0 || i1 >= numVerts ||
            i2 < 0 || i2 >= numVerts || i3 < 0 || i3 >= numVerts) continue;

        float c0 = positions[i0*3+axis], c1 = positions[i1*3+axis];
        float c2 = positions[i2*3+axis], c3 = positions[i3*3+axis];
        float tetMin = std::min({c0, c1, c2, c3});
        float tetMax = std::max({c0, c1, c2, c3});

        if (tetMax >= slabMin && tetMin <= slabMax) inSlab[t] = true;
    }

    // VALID判定
    const std::vector<bool>* tetValid = nullptr;
    if (body && !body->highResTetValid.empty()) tetValid = &body->highResTetValid;

    // 隣接グラフ
    auto adjIt = tetAdjacencyCache_.find(body);
    if (adjIt == tetAdjacencyCache_.end() || !adjIt->second.computed) {
        std::vector<int> result;
        for (int t : candidates) {
            if (!inSlab[t]) continue;
            bool isValid = !tetValid || t >= (int)tetValid->size() || (*tetValid)[t];
            if (isValid) result.push_back(t);
        }
        return result;
    }

    // BFS（複数連結成分対応）
    const TetAdjacency& adj = adjIt->second;
    std::vector<int> result;
    std::vector<bool> visited(numTets, false);

    for (int startTet : candidates) {
        if (!inSlab[startTet] || visited[startTet]) continue;

        std::queue<int> queue;
        queue.push(startTet);
        visited[startTet] = true;

        while (!queue.empty()) {
            int t = queue.front();
            queue.pop();

            if (inSlab[t]) {
                bool isValid = !tetValid || t >= (int)tetValid->size() || (*tetValid)[t];
                if (isValid) result.push_back(t);

                for (int neighbor : adj.neighbors[t]) {
                    if (!visited[neighbor] && neighbor < numTets) {
                        visited[neighbor] = true;
                        if (inSlab[neighbor]) queue.push(neighbor);
                    }
                }
            }
        }
    }

    return result;
}
//=============================================================================
// Part 2: update, computeAllCrossSections
//=============================================================================

//=============================================================================
// update()
//=============================================================================
void CRSlicerCrossSec::update() {
    if (!windowsOpen_) return;
    if (meshes_.empty()) return;

    GLFWwindow* mainCtx = glfwGetCurrentContext();

    GLint prevProgram, prevVAO, prevArrayBuffer, prevElementBuffer;
    GLint prevViewport[4], prevFramebuffer;
    GLboolean prevDepthTest, prevBlend, prevCullFace;
    GLfloat prevClearColor[4];

    glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVAO);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prevArrayBuffer);
    glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &prevElementBuffer);
    glGetIntegerv(GL_VIEWPORT, prevViewport);
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFramebuffer);
    glGetBooleanv(GL_DEPTH_TEST, &prevDepthTest);
    glGetBooleanv(GL_BLEND, &prevBlend);
    glGetBooleanv(GL_CULL_FACE, &prevCullFace);
    glGetFloatv(GL_COLOR_CLEAR_VALUE, prevClearColor);

    frameCounter_++;
    bool needsUpdate = forceUpdate_ || (frameCounter_ >= updateInterval_);

    if (needsUpdate) {
        updateBoundingBox();
        computeAllCrossSections();
        for (int i = 0; i < 3; i++) lastSlicePos_[i] = slicePos_[i];
        lastSlabThickness_ = slabThickness_;
        forceUpdate_ = false;
        frameCounter_ = 0;
    }

    for (int axis = 0; axis < 3; axis++) {
        if (windows_[axis] && !glfwWindowShouldClose(windows_[axis])) {
            renderAxisWindow(axis);
        }
    }

    glfwMakeContextCurrent(mainCtx);
    glUseProgram(prevProgram);
    glBindVertexArray(prevVAO);
    glBindBuffer(GL_ARRAY_BUFFER, prevArrayBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, prevElementBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, prevFramebuffer);
    glViewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3]);
    if (prevDepthTest) glEnable(GL_DEPTH_TEST); else glDisable(GL_DEPTH_TEST);
    if (prevBlend) glEnable(GL_BLEND); else glDisable(GL_BLEND);
    if (prevCullFace) glEnable(GL_CULL_FACE); else glDisable(GL_CULL_FACE);
    glClearColor(prevClearColor[0], prevClearColor[1], prevClearColor[2], prevClearColor[3]);
}

//=============================================================================
// computeAllCrossSections
//=============================================================================
void CRSlicerCrossSec::computeAllCrossSections() {
    auto totalStart = std::chrono::high_resolution_clock::now();

    lastProfile_.reset();
    atomicProfile_.reset();

    glm::vec3 size = boundsMax_ - boundsMin_;
    if (size.x < 0.001f || size.y < 0.001f || size.z < 0.001f) return;

    std::array<float, 3> slabMinArr, slabMaxArr, centerPosArr;
    for (int axis = 0; axis < 3; axis++) {
        centerPosArr[axis] = boundsMin_[axis] + size[axis] * slicePos_[axis];
        float halfThickness = size[axis] * slabThickness_ * 0.5f;
        slabMinArr[axis] = centerPosArr[axis] - halfThickness;
        slabMaxArr[axis] = centerPosArr[axis] + halfThickness;
        crossSections_[axis].clear();
    }

    if (!maskPoolInitialized_ || threadLocalMasks_[0].width != maskResolution_) {
        initializeMaskPool();
    }

    // メッシュデータ準備
    std::vector<PreparedMeshData> meshDataList;
    meshDataList.reserve(meshes_.size());

    for (const auto& m : meshes_) {
        if (!m.body) continue;

        const auto& tetIds = m.body->highResMeshData.tetIds;
        const std::vector<float>* positions;

        if (useSmoothMesh_ && !m.body->smoothedVertices.empty() && m.body->smoothDisplayMode) {
            positions = &m.body->smoothedVertices;
        } else {
            positions = &m.body->highRes_positions;
        }

        if (positions->empty() || tetIds.empty()) continue;

        PreparedMeshData data;
        data.body = m.body;
        data.positions = positions;
        data.tetIds = &tetIds;
        data.numVerts = static_cast<int>(positions->size() / 3);
        data.numTets = static_cast<int>(tetIds.size() / 4);
        data.color = m.color;
        data.colorComponents = 3;

        switch (colorMode_) {
        case ColorMode::OBJ_SEGMENT:
            if (m.body->useOBJSegmentColors_ && !m.body->objSegmentVertexColors_.empty()) {
                data.vertColors = m.body->objSegmentVertexColors_;
                data.colorComponents = 3;
            }
            break;
        case ColorMode::SKELETON:
            if (!m.body->vertexColors.empty()) {
                data.vertColors = m.body->vertexColors;
                data.colorComponents = 4;
            }
            break;
        default:
            break;
        }

        data.hasColors = !data.vertColors.empty() &&
                         (data.vertColors.size() / data.colorComponents >= static_cast<size_t>(data.numVerts));

        meshDataList.push_back(std::move(data));
    }

    if (meshDataList.empty()) return;

    // ★★★ 探索モードに応じて分岐 ★★★
    auto searchStart = std::chrono::high_resolution_clock::now();

#ifdef _OPENMP
    if (parallelMode_ != ParallelMode::SERIAL) {
        int numMeshes = static_cast<int>(meshDataList.size());

#pragma omp parallel for schedule(dynamic)
        for (int task = 0; task < numMeshes * 3; task++) {
            int mi = task / 3;
            int axis = task % 3;

            auto t0 = std::chrono::high_resolution_clock::now();

            if (searchMode_ == SearchMode::FULL_SCAN) {
                meshDataList[mi].intersectingTets[axis] = findIntersectingTetsFullScan(
                    meshDataList[mi].body, *meshDataList[mi].positions, *meshDataList[mi].tetIds,
                    axis, slabMinArr[axis], slabMaxArr[axis]);
            } else {
                meshDataList[mi].intersectingTets[axis] = findIntersectingTetsSpatialHash(
                    meshDataList[mi].body, *meshDataList[mi].positions, *meshDataList[mi].tetIds,
                    axis, slabMinArr[axis], slabMaxArr[axis]);
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();

            double old = atomicProfile_.timeBFS[axis].load();
            while (!atomicProfile_.timeBFS[axis].compare_exchange_weak(old, old + elapsed));

            atomicProfile_.numIntersectingTets[axis].fetch_add(
                static_cast<int>(meshDataList[mi].intersectingTets[axis].size()));
        }
    } else
#endif
    {
        for (auto& data : meshDataList) {
            for (int axis = 0; axis < 3; axis++) {
                auto t0 = std::chrono::high_resolution_clock::now();

                if (searchMode_ == SearchMode::FULL_SCAN) {
                    data.intersectingTets[axis] = findIntersectingTetsFullScan(
                        data.body, *data.positions, *data.tetIds,
                        axis, slabMinArr[axis], slabMaxArr[axis]);
                } else {
                    data.intersectingTets[axis] = findIntersectingTetsSpatialHash(
                        data.body, *data.positions, *data.tetIds,
                        axis, slabMinArr[axis], slabMaxArr[axis]);
                }

                auto t1 = std::chrono::high_resolution_clock::now();
                lastProfile_.timeBFS[axis] += std::chrono::duration<double, std::milli>(t1 - t0).count();
                lastProfile_.numIntersectingTets[axis] += static_cast<int>(data.intersectingTets[axis].size());
            }
        }
    }

    auto searchEnd = std::chrono::high_resolution_clock::now();
    double searchTime = std::chrono::duration<double, std::milli>(searchEnd - searchStart).count();

    if (debugTiming_) {
        std::cout << "[CRSlicer] " << getSearchModeName() << " time: "
                  << std::fixed << std::setprecision(2) << searchTime << "ms" << std::endl;
    }

    // モード別処理
    switch (parallelMode_) {
    case ParallelMode::SERIAL:
        computeCrossSectionsSerial(meshDataList, slabMinArr, slabMaxArr, centerPosArr);
        break;
    case ParallelMode::PARALLEL_AXES:
        computeCrossSectionsParallelAxes(meshDataList, slabMinArr, slabMaxArr, centerPosArr);
        break;
    case ParallelMode::PARALLEL_FLAT:
        computeCrossSectionsParallelFlat(meshDataList, slabMinArr, slabMaxArr, centerPosArr);
        break;
    case ParallelMode::PARALLEL_NESTED:
        computeCrossSectionsParallelNested(meshDataList, slabMinArr, slabMaxArr, centerPosArr);
        break;
    }

    // カッター断面
    if (cutterVertices_ && cutterIndices_ && !cutterVertices_->empty() && !cutterIndices_->empty()) {
        for (int axis = 0; axis < 3; axis++) {
            float planePos = boundsMin_[axis] + size[axis] * slicePos_[axis];
            auto cutterPoints = computeCutterIntersection(axis, planePos);
            if (cutterPoints.size() >= 3) {
                sortPointsByAngle(cutterPoints, axis);
                MaskBuffer cutterMask;
                cutterMask.resize(maskResolution_, maskResolution_);
                rasterizeCutterToMask(cutterMask, cutterPoints, axis);
                CRCrossSectionData cutterData = generatePolygonFromMaskForAxis(
                    cutterMask, axis, boundsMin_, boundsMax_, cutterColor_.a, slicePos_[axis]);
                if (!cutterData.empty()) {
                    for (size_t i = 0; i < cutterData.colors.size(); i += 4) {
                        cutterData.colors[i] = cutterColor_.r;
                        cutterData.colors[i+1] = cutterColor_.g;
                        cutterData.colors[i+2] = cutterColor_.b;
                        cutterData.colors[i+3] = cutterColor_.a;
                    }
                    crossSections_[axis].push_back(std::move(cutterData));
                }
            }
        }
    }

    if (parallelMode_ != ParallelMode::SERIAL) {
        atomicProfile_.copyTo(lastProfile_);
    }

    auto totalEnd = std::chrono::high_resolution_clock::now();
    lastProfile_.timeTotal = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    if (profilingEnabled_) {
        printProfilingReport();
    }
}

//=============================================================================
// モード別処理関数
//=============================================================================
void CRSlicerCrossSec::computeCrossSectionsSerial(
    std::vector<PreparedMeshData>& meshDataList,
    const std::array<float, 3>& slabMinArr,
    const std::array<float, 3>& slabMaxArr,
    const std::array<float, 3>& centerPosArr)
{
    for (int axis = 0; axis < 3; axis++) {
        threadLocalMasks_[axis].clear();
        for (auto& meshData : meshDataList) {
            if (meshData.intersectingTets[axis].empty()) continue;
            computeAxisSubSlices(axis, meshData, slabMinArr[axis], slabMaxArr[axis],
                                 centerPosArr[axis], threadLocalMasks_[axis], crossSections_[axis]);
        }
    }
}

void CRSlicerCrossSec::computeCrossSectionsParallelAxes(
    std::vector<PreparedMeshData>& meshDataList,
    const std::array<float, 3>& slabMinArr,
    const std::array<float, 3>& slabMaxArr,
    const std::array<float, 3>& centerPosArr)
{
    std::array<std::vector<CRCrossSectionData>, 3> axisResults;

#ifdef _OPENMP
#pragma omp parallel for num_threads(3) schedule(static)
#endif
    for (int axis = 0; axis < 3; axis++) {
        MaskBuffer& localMask = threadLocalMasks_[axis];
        localMask.clear();
        for (auto& meshData : meshDataList) {
            if (meshData.intersectingTets[axis].empty()) continue;
            computeAxisSubSlices(axis, meshData, slabMinArr[axis], slabMaxArr[axis],
                                 centerPosArr[axis], localMask, axisResults[axis]);
        }
    }

    for (int axis = 0; axis < 3; axis++) {
        crossSections_[axis] = std::move(axisResults[axis]);
    }
}

void CRSlicerCrossSec::computeCrossSectionsParallelFlat(
    std::vector<PreparedMeshData>& meshDataList,
    const std::array<float, 3>& slabMinArr,
    const std::array<float, 3>& slabMaxArr,
    const std::array<float, 3>& centerPosArr)
{
    int numMeshes = static_cast<int>(meshDataList.size());
    int totalTasks = 3 * numMeshes * numSubSlices_;

    struct TaskResult { int axis; int meshIdx; CRCrossSectionData data; };
    std::vector<TaskResult> allResults(totalTasks);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int taskIdx = 0; taskIdx < totalTasks; taskIdx++) {
        int axis = taskIdx / (numMeshes * numSubSlices_);
        int remainder = taskIdx % (numMeshes * numSubSlices_);
        int meshIdx = remainder / numSubSlices_;
        int subIdx = remainder % numSubSlices_;

        allResults[taskIdx].axis = axis;
        allResults[taskIdx].meshIdx = meshIdx;

        const auto& meshData = meshDataList[meshIdx];
        if (meshData.intersectingTets[axis].empty()) continue;

        float t_param = (numSubSlices_ == 1) ? 0.0f : (float)subIdx / (float)(numSubSlices_ - 1) - 0.5f;
        float halfThickness = (slabMaxArr[axis] - slabMinArr[axis]) * 0.5f;
        float planePos = centerPosArr[axis] + t_param * halfThickness * 2.0f;

        allResults[taskIdx].data = computeCrossSectionOptimized(
            *meshData.positions, *meshData.tetIds, meshData.intersectingTets[axis],
            axis, planePos, meshData.numVerts,
            meshData.vertColors, meshData.colorComponents, meshData.hasColors, meshData.color);
    }

    std::vector<std::vector<MaskBuffer>> meshMasks(3);
    for (int axis = 0; axis < 3; axis++) {
        meshMasks[axis].resize(numMeshes);
        for (int m = 0; m < numMeshes; m++) {
            meshMasks[axis][m].resize(maskResolution_, maskResolution_);
        }
    }

    for (const auto& result : allResults) {
        if (!result.data.empty()) {
            rasterizeCrossSectionToMaskOptimized(meshMasks[result.axis][result.meshIdx],
                                                 result.data, result.axis, boundsMin_, boundsMax_);
        }
    }

    for (int axis = 0; axis < 3; axis++) {
        for (int m = 0; m < numMeshes; m++) {
            CRCrossSectionData finalData = generatePolygonFromMaskForAxis(
                meshMasks[axis][m], axis, boundsMin_, boundsMax_,
                meshDataList[m].color.a, slicePos_[axis]);
            if (!finalData.empty()) {
                crossSections_[axis].push_back(std::move(finalData));
            }
        }
    }
}

void CRSlicerCrossSec::computeCrossSectionsParallelNested(
    std::vector<PreparedMeshData>& meshDataList,
    const std::array<float, 3>& slabMinArr,
    const std::array<float, 3>& slabMaxArr,
    const std::array<float, 3>& centerPosArr)
{
    std::array<std::vector<CRCrossSectionData>, 3> axisResults;

#ifdef _OPENMP
    omp_set_nested(1);
#pragma omp parallel for num_threads(3) schedule(static)
#endif
    for (int axis = 0; axis < 3; axis++) {
        MaskBuffer localMask;
        localMask.resize(maskResolution_, maskResolution_);

        for (auto& meshData : meshDataList) {
            if (meshData.intersectingTets[axis].empty()) continue;

            localMask.clear();
            std::vector<CRCrossSectionData> subResults(numSubSlices_);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int subIdx = 0; subIdx < numSubSlices_; subIdx++) {
                float t_param = (numSubSlices_ == 1) ? 0.0f : (float)subIdx / (float)(numSubSlices_ - 1) - 0.5f;
                float halfThickness = (slabMaxArr[axis] - slabMinArr[axis]) * 0.5f;
                float planePos = centerPosArr[axis] + t_param * halfThickness * 2.0f;

                subResults[subIdx] = computeCrossSectionOptimized(
                    *meshData.positions, *meshData.tetIds, meshData.intersectingTets[axis],
                    axis, planePos, meshData.numVerts,
                    meshData.vertColors, meshData.colorComponents, meshData.hasColors, meshData.color);
            }

            for (int subIdx = 0; subIdx < numSubSlices_; subIdx++) {
                if (!subResults[subIdx].empty()) {
                    rasterizeCrossSectionToMaskOptimized(localMask, subResults[subIdx], axis, boundsMin_, boundsMax_);
                }
            }

            CRCrossSectionData finalData = generatePolygonFromMaskForAxis(
                localMask, axis, boundsMin_, boundsMax_, meshData.color.a, slicePos_[axis]);
            if (!finalData.empty()) {
                axisResults[axis].push_back(std::move(finalData));
            }
        }
    }
#ifdef _OPENMP
    omp_set_nested(0);
#endif

    for (int axis = 0; axis < 3; axis++) {
        crossSections_[axis] = std::move(axisResults[axis]);
    }
}

void CRSlicerCrossSec::computeAxisSubSlices(
    int axis, const PreparedMeshData& meshData,
    float slabMin, float slabMax, float centerPos,
    MaskBuffer& mask, std::vector<CRCrossSectionData>& results)
{
    mask.clear();

    for (int subIdx = 0; subIdx < numSubSlices_; subIdx++) {
        float t_param = (numSubSlices_ == 1) ? 0.0f : (float)subIdx / (float)(numSubSlices_ - 1) - 0.5f;
        float halfThickness = (slabMax - slabMin) * 0.5f;
        float planePos = centerPos + t_param * halfThickness * 2.0f;

        CRCrossSectionData subData = computeCrossSectionOptimized(
            *meshData.positions, *meshData.tetIds, meshData.intersectingTets[axis],
            axis, planePos, meshData.numVerts,
            meshData.vertColors, meshData.colorComponents, meshData.hasColors, meshData.color);

        if (!subData.empty()) {
            rasterizeCrossSectionToMaskOptimized(mask, subData, axis, boundsMin_, boundsMax_);
        }
    }

    CRCrossSectionData finalData = generatePolygonFromMaskForAxis(
        mask, axis, boundsMin_, boundsMax_, meshData.color.a, slicePos_[axis]);

    if (!finalData.empty()) {
        results.push_back(std::move(finalData));
    }
}
//=============================================================================
// Part 3: 断面計算、ラスタライズ、カッター
//=============================================================================

CRCrossSectionData CRSlicerCrossSec::computeCrossSectionOptimized(
    const std::vector<float>& positions, const std::vector<int>& tetIds,
    const std::vector<int>& intersectingTets, int axis, float planePos, int numVerts,
    const std::vector<float>& vertexColors, int colorComponents, bool hasColors,
    const glm::vec4& baseColor) const
{
    CRCrossSectionData result;
    result.vertices.reserve(intersectingTets.size() * 12);
    result.colors.reserve(intersectingTets.size() * 16);
    result.indices.reserve(intersectingTets.size() * 6);

    for (int t : intersectingTets) {
        int i0 = tetIds[t*4+0], i1 = tetIds[t*4+1], i2 = tetIds[t*4+2], i3 = tetIds[t*4+3];

        glm::vec3 tetVerts[4] = {
            glm::vec3(positions[i0*3], positions[i0*3+1], positions[i0*3+2]),
            glm::vec3(positions[i1*3], positions[i1*3+1], positions[i1*3+2]),
            glm::vec3(positions[i2*3], positions[i2*3+1], positions[i2*3+2]),
            glm::vec3(positions[i3*3], positions[i3*3+1], positions[i3*3+2])
        };

        float minCoord = std::min({tetVerts[0][axis], tetVerts[1][axis], tetVerts[2][axis], tetVerts[3][axis]});
        float maxCoord = std::max({tetVerts[0][axis], tetVerts[1][axis], tetVerts[2][axis], tetVerts[3][axis]});
        if (planePos < minCoord || planePos > maxCoord) continue;

        glm::vec4 tetColors[4];
        if (hasColors) {
            if (colorComponents == 4) {
                tetColors[0] = glm::vec4(vertexColors[i0*4], vertexColors[i0*4+1], vertexColors[i0*4+2], vertexColors[i0*4+3]);
                tetColors[1] = glm::vec4(vertexColors[i1*4], vertexColors[i1*4+1], vertexColors[i1*4+2], vertexColors[i1*4+3]);
                tetColors[2] = glm::vec4(vertexColors[i2*4], vertexColors[i2*4+1], vertexColors[i2*4+2], vertexColors[i2*4+3]);
                tetColors[3] = glm::vec4(vertexColors[i3*4], vertexColors[i3*4+1], vertexColors[i3*4+2], vertexColors[i3*4+3]);
            } else {
                tetColors[0] = glm::vec4(vertexColors[i0*3], vertexColors[i0*3+1], vertexColors[i0*3+2], 1.0f);
                tetColors[1] = glm::vec4(vertexColors[i1*3], vertexColors[i1*3+1], vertexColors[i1*3+2], 1.0f);
                tetColors[2] = glm::vec4(vertexColors[i2*3], vertexColors[i2*3+1], vertexColors[i2*3+2], 1.0f);
                tetColors[3] = glm::vec4(vertexColors[i3*3], vertexColors[i3*3+1], vertexColors[i3*3+2], 1.0f);
            }
        } else {
            tetColors[0] = tetColors[1] = tetColors[2] = tetColors[3] = baseColor;
        }

        glm::vec3 outVerts[4];
        glm::vec4 outColors[4];
        int numIntersect = intersectTetrahedronWithPlane(tetVerts, tetColors, axis, planePos, outVerts, outColors);
        if (numIntersect >= 3) {
            triangulatePolygonLocal(result, outVerts, outColors, numIntersect);
        }
    }
    return result;
}

int CRSlicerCrossSec::intersectTetrahedronWithPlane(
    const glm::vec3 tetVerts[4], const glm::vec4 tetColors[4],
    int axis, float planePos, glm::vec3 outVerts[4], glm::vec4 outColors[4]) const
{
    float d[4];
    for (int i = 0; i < 4; i++) d[i] = tetVerts[i][axis] - planePos;

    int numIntersect = 0;
    static const int edges[6][2] = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};

    for (int e = 0; e < 6; e++) {
        int i = edges[e][0], j = edges[e][1];
        if (d[i] * d[j] > 0.0f) continue;

        if (std::abs(d[i]) < 0.0001f) {
            bool dup = false;
            for (int k = 0; k < numIntersect; k++) {
                if (glm::length(outVerts[k] - tetVerts[i]) < 0.0001f) { dup = true; break; }
            }
            if (!dup && numIntersect < 4) {
                outVerts[numIntersect] = tetVerts[i];
                outColors[numIntersect] = tetColors[i];
                numIntersect++;
            }
            continue;
        }
        if (std::abs(d[j]) < 0.0001f) {
            bool dup = false;
            for (int k = 0; k < numIntersect; k++) {
                if (glm::length(outVerts[k] - tetVerts[j]) < 0.0001f) { dup = true; break; }
            }
            if (!dup && numIntersect < 4) {
                outVerts[numIntersect] = tetVerts[j];
                outColors[numIntersect] = tetColors[j];
                numIntersect++;
            }
            continue;
        }

        float t = d[i] / (d[i] - d[j]);
        glm::vec3 p = tetVerts[i] + t * (tetVerts[j] - tetVerts[i]);
        glm::vec4 c = tetColors[i] + t * (tetColors[j] - tetColors[i]);
        if (numIntersect < 4) {
            outVerts[numIntersect] = p;
            outColors[numIntersect] = c;
            numIntersect++;
        }
    }

    if (numIntersect >= 3) {
        glm::vec3 center(0.0f);
        for (int i = 0; i < numIntersect; i++) center += outVerts[i];
        center /= (float)numIntersect;

        int uAxis = (axis == 0) ? 1 : 0;
        int vAxis = (axis == 2) ? 1 : 2;

        for (int i = 0; i < numIntersect - 1; i++) {
            for (int j = i + 1; j < numIntersect; j++) {
                float angle_i = std::atan2(outVerts[i][vAxis] - center[vAxis], outVerts[i][uAxis] - center[uAxis]);
                float angle_j = std::atan2(outVerts[j][vAxis] - center[vAxis], outVerts[j][uAxis] - center[uAxis]);
                if (angle_i > angle_j) {
                    std::swap(outVerts[i], outVerts[j]);
                    std::swap(outColors[i], outColors[j]);
                }
            }
        }
    }
    return numIntersect;
}

void CRSlicerCrossSec::triangulatePolygonLocal(
    CRCrossSectionData& data, const glm::vec3* verts, const glm::vec4* colors, int vertCount) const
{
    if (vertCount < 3) return;
    unsigned int baseIdx = static_cast<unsigned int>(data.vertices.size() / 3);
    for (int i = 0; i < vertCount; i++) data.addVertex(verts[i], colors[i]);
    for (int i = 1; i < vertCount - 1; i++) data.addTriangle(baseIdx, baseIdx + i, baseIdx + i + 1);
}

void CRSlicerCrossSec::rasterizeCrossSectionToMaskOptimized(
    MaskBuffer& buf, const CRCrossSectionData& cs, int axis,
    const glm::vec3& boundsMin, const glm::vec3& boundsMax) const
{
    if (cs.empty()) return;

    glm::vec3 size = boundsMax - boundsMin;
    int uAxis = (axis == 0) ? 2 : 0;
    int vAxis = (axis == 2) ? 1 : (axis == 0) ? 1 : 2;

    float scaleU = (buf.width - 1) / size[uAxis];
    float scaleV = (buf.height - 1) / size[vAxis];

    for (size_t i = 0; i < cs.indices.size(); i += 3) {
        unsigned int i0 = cs.indices[i], i1 = cs.indices[i+1], i2 = cs.indices[i+2];

        float u0 = (cs.vertices[i0*3+uAxis] - boundsMin[uAxis]) * scaleU;
        float v0 = (cs.vertices[i0*3+vAxis] - boundsMin[vAxis]) * scaleV;
        float u1 = (cs.vertices[i1*3+uAxis] - boundsMin[uAxis]) * scaleU;
        float v1 = (cs.vertices[i1*3+vAxis] - boundsMin[vAxis]) * scaleV;
        float u2 = (cs.vertices[i2*3+uAxis] - boundsMin[uAxis]) * scaleU;
        float v2 = (cs.vertices[i2*3+vAxis] - boundsMin[vAxis]) * scaleV;

        int x0 = static_cast<int>(u0), y0 = static_cast<int>(v0);
        int x1 = static_cast<int>(u1), y1 = static_cast<int>(v1);
        int x2 = static_cast<int>(u2), y2 = static_cast<int>(v2);

        glm::vec4 c0(cs.colors[i0*4], cs.colors[i0*4+1], cs.colors[i0*4+2], cs.colors[i0*4+3]);
        glm::vec4 c1(cs.colors[i1*4], cs.colors[i1*4+1], cs.colors[i1*4+2], cs.colors[i1*4+3]);
        glm::vec4 c2(cs.colors[i2*4], cs.colors[i2*4+1], cs.colors[i2*4+2], cs.colors[i2*4+3]);

        int minX = std::max(0, std::min({x0, x1, x2}));
        int maxX = std::min(buf.width - 1, std::max({x0, x1, x2}));
        int minY = std::max(0, std::min({y0, y1, y2}));
        int maxY = std::min(buf.height - 1, std::max({y0, y1, y2}));

        if (maxX - minX < 1 && maxY - minY < 1) {
            int idx = y0 * buf.width + x0;
            if (idx >= 0 && idx < buf.width * buf.height) {
                buf.mask[idx] = 255;
                buf.colorSum[idx] += (c0 + c1 + c2) / 3.0f;
                buf.colorCount[idx]++;
            }
            continue;
        }

        float denom = (float)((y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2));
        if (std::abs(denom) < 0.0001f) continue;
        float invDenom = 1.0f / denom;

        for (int py = minY; py <= maxY; py++) {
            for (int px = minX; px <= maxX; px++) {
                float w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) * invDenom;
                float w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) * invDenom;
                float w2 = 1.0f - w0 - w1;
                if (w0 >= -0.001f && w1 >= -0.001f && w2 >= -0.001f) {
                    int idx = py * buf.width + px;
                    buf.mask[idx] = 255;
                    buf.colorSum[idx] += c0 * w0 + c1 * w1 + c2 * w2;
                    buf.colorCount[idx]++;
                }
            }
        }
    }
}

CRCrossSectionData CRSlicerCrossSec::generatePolygonFromMaskForAxis(
    const MaskBuffer& buf, int axis, const glm::vec3& boundsMin, const glm::vec3& boundsMax,
    float alpha, float slicePosNormalized) const
{
    CRCrossSectionData result;

    glm::vec3 size = boundsMax - boundsMin;
    float sliceCoord = boundsMin[axis] + size[axis] * slicePosNormalized;

    int uAxis = (axis == 0) ? 2 : 0;
    int vAxis = (axis == 2) ? 1 : (axis == 0) ? 1 : 2;

    float pixelSizeU = size[uAxis] / buf.width;
    float pixelSizeV = size[vAxis] / buf.height;

    for (int py = 0; py < buf.height; py++) {
        int runStart = -1;
        glm::vec4 runColorSum(0.0f);
        int runColorCount = 0;

        for (int px = 0; px <= buf.width; px++) {
            int idx = py * buf.width + px;
            bool isSet = (px < buf.width) && (buf.mask[idx] != 0);

            if (isSet) {
                if (runStart < 0) {
                    runStart = px;
                    runColorSum = glm::vec4(0.0f);
                    runColorCount = 0;
                }
                if (buf.colorCount[idx] > 0) {
                    runColorSum += buf.colorSum[idx];
                    runColorCount += buf.colorCount[idx];
                }
            } else {
                if (runStart >= 0) {
                    glm::vec4 avgColor(0.8f, 0.3f, 0.3f, 1.0f);
                    if (runColorCount > 0) avgColor = runColorSum / (float)runColorCount;
                    avgColor.a = alpha;

                    float uStart = boundsMin[uAxis] + runStart * pixelSizeU;
                    float uEnd = boundsMin[uAxis] + px * pixelSizeU;
                    float vStart = boundsMin[vAxis] + py * pixelSizeV;
                    float vEnd = boundsMin[vAxis] + (py + 1) * pixelSizeV;

                    glm::vec3 p0, p1, p2, p3;
                    p0[axis] = p1[axis] = p2[axis] = p3[axis] = sliceCoord;
                    p0[uAxis] = uStart; p0[vAxis] = vStart;
                    p1[uAxis] = uEnd;   p1[vAxis] = vStart;
                    p2[uAxis] = uEnd;   p2[vAxis] = vEnd;
                    p3[uAxis] = uStart; p3[vAxis] = vEnd;

                    unsigned int baseIdx = static_cast<unsigned int>(result.vertices.size() / 3);
                    result.addVertex(p0, avgColor);
                    result.addVertex(p1, avgColor);
                    result.addVertex(p2, avgColor);
                    result.addVertex(p3, avgColor);
                    result.addTriangle(baseIdx, baseIdx + 1, baseIdx + 2);
                    result.addTriangle(baseIdx, baseIdx + 2, baseIdx + 3);

                    runStart = -1;
                }
            }
        }
    }
    return result;
}

//=============================================================================
// カッター
//=============================================================================
std::vector<glm::vec3> CRSlicerCrossSec::computeCutterIntersection(int axis, float planePos) const {
    std::vector<glm::vec3> intersections;
    if (!cutterVertices_ || !cutterIndices_) return intersections;

    const auto& verts = *cutterVertices_;
    const auto& indices = *cutterIndices_;

    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        unsigned int i0 = indices[i], i1 = indices[i+1], i2 = indices[i+2];
        if (i0*3+2 >= verts.size() || i1*3+2 >= verts.size() || i2*3+2 >= verts.size()) continue;

        glm::vec3 v0(verts[i0*3], verts[i0*3+1], verts[i0*3+2]);
        glm::vec3 v1(verts[i1*3], verts[i1*3+1], verts[i1*3+2]);
        glm::vec3 v2(verts[i2*3], verts[i2*3+1], verts[i2*3+2]);

        auto checkEdge = [&](const glm::vec3& a, const glm::vec3& b) {
            float aVal = a[axis], bVal = b[axis];
            if ((aVal <= planePos && bVal >= planePos) || (aVal >= planePos && bVal <= planePos)) {
                float denom = bVal - aVal;
                if (std::abs(denom) > 1e-6f) {
                    float t = (planePos - aVal) / denom;
                    if (t >= 0.0f && t <= 1.0f) intersections.push_back(a + t * (b - a));
                }
            }
        };
        checkEdge(v0, v1);
        checkEdge(v1, v2);
        checkEdge(v2, v0);
    }
    return intersections;
}

void CRSlicerCrossSec::sortPointsByAngle(std::vector<glm::vec3>& points, int axis) const {
    if (points.size() < 3) return;

    glm::vec3 center(0.0f);
    for (const auto& p : points) center += p;
    center /= static_cast<float>(points.size());

    std::sort(points.begin(), points.end(), [&center, axis](const glm::vec3& a, const glm::vec3& b) {
        float ax, ay, bx, by;
        if (axis == 0) { ax = a.y - center.y; ay = a.z - center.z; bx = b.y - center.y; by = b.z - center.z; }
        else if (axis == 1) { ax = a.x - center.x; ay = a.z - center.z; bx = b.x - center.x; by = b.z - center.z; }
        else { ax = a.x - center.x; ay = a.y - center.y; bx = b.x - center.x; by = b.y - center.y; }
        return std::atan2(ay, ax) < std::atan2(by, bx);
    });
}

void CRSlicerCrossSec::rasterizeCutterToMask(MaskBuffer& mask, const std::vector<glm::vec3>& points, int axis) const {
    if (points.size() < 3) return;

    glm::vec3 boundsSize = boundsMax_ - boundsMin_;
    auto to2D = [&](const glm::vec3& p) -> std::pair<float, float> {
        float u, v;
        if (axis == 0) {
            u = (boundsSize.z > 0.001f) ? (p.z - boundsMin_.z) / boundsSize.z : 0.5f;
            v = (boundsSize.y > 0.001f) ? (p.y - boundsMin_.y) / boundsSize.y : 0.5f;
        } else if (axis == 1) {
            u = (boundsSize.x > 0.001f) ? (p.x - boundsMin_.x) / boundsSize.x : 0.5f;
            v = (boundsSize.z > 0.001f) ? (p.z - boundsMin_.z) / boundsSize.z : 0.5f;
        } else {
            u = (boundsSize.x > 0.001f) ? (p.x - boundsMin_.x) / boundsSize.x : 0.5f;
            v = (boundsSize.y > 0.001f) ? (p.y - boundsMin_.y) / boundsSize.y : 0.5f;
        }
        return {u * (mask.width - 1), v * (mask.height - 1)};
    };

    std::vector<std::pair<float, float>> poly2D;
    for (const auto& p : points) poly2D.push_back(to2D(p));

    float minY = FLT_MAX, maxY = -FLT_MAX;
    for (const auto& p : poly2D) {
        minY = std::min(minY, p.second);
        maxY = std::max(maxY, p.second);
    }

    for (int y = std::max(0, (int)std::floor(minY)); y <= std::min(mask.height - 1, (int)std::ceil(maxY)); y++) {
        std::vector<float> xIntersections;
        int n = static_cast<int>(poly2D.size());
        for (int i = 0; i < n; i++) {
            int j = (i + 1) % n;
            float y0 = poly2D[i].second, y1 = poly2D[j].second;
            float x0 = poly2D[i].first, x1 = poly2D[j].first;
            if ((y0 <= y && y1 > y) || (y1 <= y && y0 > y)) {
                xIntersections.push_back(x0 + (y - y0) / (y1 - y0) * (x1 - x0));
            }
        }
        std::sort(xIntersections.begin(), xIntersections.end());

        for (size_t i = 0; i + 1 < xIntersections.size(); i += 2) {
            for (int x = std::max(0, (int)std::ceil(xIntersections[i]));
                 x <= std::min(mask.width - 1, (int)std::floor(xIntersections[i + 1])); x++) {
                int idx = y * mask.width + x;
                mask.mask[idx] = 1;
                mask.colorSum[idx] = cutterColor_;
                mask.colorCount[idx] = 1;
            }
        }
    }
}

void CRSlicerCrossSec::setCutterMesh(const std::vector<float>* vertices,
                                     const std::vector<unsigned int>* indices, const glm::vec4& color) {
    cutterVertices_ = vertices;
    cutterIndices_ = indices;
    cutterColor_ = color;
}

void CRSlicerCrossSec::clearCutterMesh() {
    cutterVertices_ = nullptr;
    cutterIndices_ = nullptr;
}
//=============================================================================
// Part 4: 描画、カメラ連動、プロファイリング
//=============================================================================

void CRSlicerCrossSec::renderAxisWindow(int axis) {
    GLFWwindow* win = windows_[axis];
    if (!win) return;

    glfwMakeContextCurrent(win);
    glViewport(0, 0, windowSize_, windowSize_);
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (crossSections_[axis].empty()) { glfwSwapBuffers(win); return; }

    glm::vec3 center = (boundsMin_ + boundsMax_) * 0.5f;
    glm::vec3 size = boundsMax_ - boundsMin_;
    float maxExt = std::max({size.x, size.y, size.z}) * 0.5f;
    float distance = maxExt * 2.0f;

    glm::vec3 eye, up;
    switch (axis) {
    case 0:
        eye = center + glm::vec3((cameraDirection_.x >= 0 ? -1 : 1) * distance, 0, 0);
        up = glm::normalize(glm::vec3(0, orbitCameraUp_.y, orbitCameraUp_.z));
        if (glm::length(up) < 0.01f) up = glm::vec3(0, 1, 0);
        break;
    case 1:
        eye = center + glm::vec3(0, (cameraDirection_.y >= 0 ? -1 : 1) * distance, 0);
        up = glm::normalize(glm::vec3(orbitCameraUp_.x, 0, orbitCameraUp_.z));
        if (glm::length(up) < 0.01f) up = glm::vec3(0, 0, 1);
        break;
    default:
        eye = center + glm::vec3(0, 0, (cameraDirection_.z >= 0 ? -1 : 1) * distance);
        up = glm::normalize(glm::vec3(orbitCameraUp_.x, orbitCameraUp_.y, 0));
        if (glm::length(up) < 0.01f) up = glm::vec3(0, 1, 0);
        break;
    }

    glm::mat4 view = glm::lookAt(eye, center, up);
    glm::mat4 proj = glm::ortho(-maxExt, maxExt, -maxExt, maxExt, 0.1f, distance * 3.0f);
    glm::mat4 mvp = proj * view;

    glUseProgram(shaderPrograms_[axis]);
    glUniformMatrix4fv(glGetUniformLocation(shaderPrograms_[axis], "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);
    glBindVertexArray(vaos_[axis]);

    for (const auto& cs : crossSections_[axis]) {
        if (cs.empty()) continue;

        glBindBuffer(GL_ARRAY_BUFFER, vbos_[axis]);
        glBufferData(GL_ARRAY_BUFFER, cs.vertices.size() * sizeof(float), cs.vertices.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, cbos_[axis]);
        glBufferData(GL_ARRAY_BUFFER, cs.colors.size() * sizeof(float), cs.colors.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(1);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebos_[axis]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, cs.indices.size() * sizeof(unsigned int), cs.indices.data(), GL_DYNAMIC_DRAW);

        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(cs.indices.size()), GL_UNSIGNED_INT, nullptr);
    }

    glBindVertexArray(0);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glfwSwapBuffers(win);
}

void CRSlicerCrossSec::drawBoundingBox(const glm::mat4& viewMatrix, const glm::mat4& projMatrix, const glm::vec4& color) {
    if (!createPreviewShader()) return;

    GLint prevProgram;
    glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);

    glUseProgram(previewShaderProgram_);
    glm::mat4 mvp = projMatrix * viewMatrix;
    glUniformMatrix4fv(glGetUniformLocation(previewShaderProgram_, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform4fv(glGetUniformLocation(previewShaderProgram_, "uColor"), 1, &color[0]);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    glLineWidth(1.5f);

    glm::vec3 c[8] = {
        {boundsMin_.x, boundsMin_.y, boundsMin_.z}, {boundsMax_.x, boundsMin_.y, boundsMin_.z},
        {boundsMax_.x, boundsMax_.y, boundsMin_.z}, {boundsMin_.x, boundsMax_.y, boundsMin_.z},
        {boundsMin_.x, boundsMin_.y, boundsMax_.z}, {boundsMax_.x, boundsMin_.y, boundsMax_.z},
        {boundsMax_.x, boundsMax_.y, boundsMax_.z}, {boundsMin_.x, boundsMax_.y, boundsMax_.z}
    };

    int edges[12][2] = {{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};
    GLfloat verts[72];
    for (int e = 0; e < 12; e++) {
        for (int k = 0; k < 3; k++) verts[e*6+k] = c[edges[e][0]][k];
        for (int k = 0; k < 3; k++) verts[e*6+3+k] = c[edges[e][1]][k];
    }

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);
    glDrawArrays(GL_LINES, 0, 24);
    glBindVertexArray(0);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glUseProgram(prevProgram);
}

void CRSlicerCrossSec::drawSlicePlanes(const glm::mat4& viewMatrix, const glm::mat4& projMatrix) {
    if (!createPreviewShader()) return;

    GLint prevProgram;
    glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);

    glUseProgram(previewShaderProgram_);
    glm::mat4 mvp = projMatrix * viewMatrix;
    glUniformMatrix4fv(glGetUniformLocation(previewShaderProgram_, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    glm::vec3 size = boundsMax_ - boundsMin_;
    glm::vec4 colors[3] = {{1, 0.3f, 0.3f, 0.6f}, {0.3f, 1, 0.3f, 0.6f}, {0.3f, 0.3f, 1, 0.6f}};

    for (int a = 0; a < 3; a++) {
        float pos = boundsMin_[a] + size[a] * slicePos_[a];
        std::vector<GLfloat> v;
        switch (a) {
        case 0: v = {pos, boundsMin_.y, boundsMin_.z, pos, boundsMax_.y, boundsMin_.z,
                 pos, boundsMax_.y, boundsMax_.z, pos, boundsMin_.y, boundsMax_.z}; break;
        case 1: v = {boundsMin_.x, pos, boundsMin_.z, boundsMax_.x, pos, boundsMin_.z,
                 boundsMax_.x, pos, boundsMax_.z, boundsMin_.x, pos, boundsMax_.z}; break;
        default: v = {boundsMin_.x, boundsMin_.y, pos, boundsMax_.x, boundsMin_.y, pos,
                 boundsMax_.x, boundsMax_.y, pos, boundsMin_.x, boundsMax_.y, pos}; break;
        }

        GLuint vao, vbo;
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, v.size() * sizeof(GLfloat), v.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);

        glLineWidth(currentAxis_ == a ? 3.0f : 1.5f);
        glUniform4fv(glGetUniformLocation(previewShaderProgram_, "uColor"), 1, &colors[a][0]);
        glDrawArrays(GL_LINE_LOOP, 0, 4);

        glBindVertexArray(0);
        glDeleteVertexArrays(1, &vao);
        glDeleteBuffers(1, &vbo);
    }

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glUseProgram(prevProgram);
}

//=============================================================================
// カメラ連動
//=============================================================================
void CRSlicerCrossSec::setSliceAxisFromCamera(const glm::vec3& cameraPos, const glm::vec3& cameraTarget, const glm::mat4& viewMatrix) {
    if (previewLocked_) return;

    cameraPos_ = cameraPos;
    cameraTarget_ = cameraTarget;
    viewMatrix_ = viewMatrix;
    cameraRight_ = glm::vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]);
    cameraUp_ = glm::vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);

    glm::vec3 viewDir = glm::normalize(cameraTarget - cameraPos);
    cameraDirection_ = viewDir;
    orbitCameraUp_ = cameraUp_;

    float dotX = std::abs(glm::dot(viewDir, glm::vec3(1, 0, 0)));
    float dotY = std::abs(glm::dot(viewDir, glm::vec3(0, 1, 0)));
    float dotZ = std::abs(glm::dot(viewDir, glm::vec3(0, 0, 1)));

    if (dotX >= dotY && dotX >= dotZ) autoSelectedAxis_ = 0;
    else if (dotY >= dotX && dotY >= dotZ) autoSelectedAxis_ = 1;
    else autoSelectedAxis_ = 2;

    currentAxis_ = (autoSelectedAxis_ + axisOffset_) % 3;
}

void CRSlicerCrossSec::cycleAxisOffset() {
    axisOffset_ = (axisOffset_ + 1) % 3;
    currentAxis_ = (autoSelectedAxis_ + axisOffset_) % 3;
    static const char* names[] = {"X", "Y", "Z"};
    std::cout << "[CRSlicer] Axis: " << names[currentAxis_] << std::endl;
}

void CRSlicerCrossSec::togglePreviewLock() {
    previewLocked_ = !previewLocked_;
    std::cout << "[CRSlicer] Preview " << (previewLocked_ ? "LOCKED" : "UNLOCKED") << std::endl;
}

void CRSlicerCrossSec::alignSlicesToPosition(const glm::vec3& worldPos) {
    glm::vec3 size = boundsMax_ - boundsMin_;
    if (size.x > 0.001f) slicePos_[0] = glm::clamp((worldPos.x - boundsMin_.x) / size.x, 0.0f, 1.0f);
    if (size.y > 0.001f) slicePos_[1] = glm::clamp((worldPos.y - boundsMin_.y) / size.y, 0.0f, 1.0f);
    if (size.z > 0.001f) slicePos_[2] = glm::clamp((worldPos.z - boundsMin_.z) / size.z, 0.0f, 1.0f);
    forceUpdate_ = true;
}

void CRSlicerCrossSec::alignSlicesToCutterCenter(const std::vector<float>* cutterVertices) {
    if (!cutterVertices || cutterVertices->empty()) return;
    glm::vec3 center(0.0f);
    int numVerts = static_cast<int>(cutterVertices->size() / 3);
    for (int i = 0; i < numVerts; i++) {
        center.x += (*cutterVertices)[i * 3];
        center.y += (*cutterVertices)[i * 3 + 1];
        center.z += (*cutterVertices)[i * 3 + 2];
    }
    alignSlicesToPosition(center / static_cast<float>(numVerts));
}

int CRSlicerCrossSec::getWindowUnderMouse() const {
    for (int i = 0; i < 3; i++) {
        if (windows_[i] && glfwGetWindowAttrib(windows_[i], GLFW_FOCUSED)) return i;
    }
    return -1;
}

//=============================================================================
// プロファイリング
//=============================================================================
void CRSlicerCrossSec::printProfilingReport() {
    for (int i = 0; i < 3; i++) {
        sumProfile_.timeBFS[i] += lastProfile_.timeBFS[i];
        sumProfile_.timeHash[i] += lastProfile_.timeHash[i];
        sumProfile_.timeScan[i] += lastProfile_.timeScan[i];
        sumProfile_.numIntersectingTets[i] += lastProfile_.numIntersectingTets[i];
        sumProfile_.numCandidateTets[i] += lastProfile_.numCandidateTets[i];
    }
    sumProfile_.timeTotal += lastProfile_.timeTotal;
    profileCounter_++;

    if (profileCounter_ >= 30) {
        std::cout << "\n[CRSlicer] ========== " << getSearchModeName() << " (" << getParallelModeName() << ") ==========" << std::endl;
        std::cout << std::fixed << std::setprecision(2);

        const char* names[] = {"X", "Y", "Z"};
        for (int i = 0; i < 3; i++) {
            double total = sumProfile_.timeBFS[i] / profileCounter_;
            int tets = sumProfile_.numIntersectingTets[i] / profileCounter_;

            if (searchMode_ == SearchMode::SPATIAL_HASH) {
                double hash = sumProfile_.timeHash[i] / profileCounter_;
                int cand = sumProfile_.numCandidateTets[i] / profileCounter_;
                std::cout << "  [" << names[i] << "] Hash:" << hash << "ms Total:" << total
                          << "ms (cand:" << cand << " -> tets:" << tets << ")" << std::endl;
            } else {
                double scan = sumProfile_.timeScan[i] / profileCounter_;
                std::cout << "  [" << names[i] << "] Scan:" << scan << "ms Total:" << total
                          << "ms (tets:" << tets << ")" << std::endl;
            }
        }
        std::cout << "  TOTAL: " << (sumProfile_.timeTotal / profileCounter_) << " ms" << std::endl;
        std::cout << "=============================================" << std::endl;

        profileCounter_ = 0;
        sumProfile_.reset();
    }
}
