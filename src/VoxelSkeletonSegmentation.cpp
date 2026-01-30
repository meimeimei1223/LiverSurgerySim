#include "VoxelSkeletonSegmentation.h"
#include "ShaderProgram.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <functional>
#include <set>
#include <chrono>
#include "VoronoiPathSegmentation.h"
#include "SoftBodyGPUDuo.h"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace VoxelSkeleton {

//=============================================================================
// 色パレット
//=============================================================================
glm::vec3 VesselSegmentation::getColorForIndex(int index) {
    static const std::vector<glm::vec3> palette = {
        glm::vec3(0.9f, 0.2f, 0.2f),
        glm::vec3(0.2f, 0.7f, 0.9f),
        glm::vec3(0.2f, 0.9f, 0.3f),
        glm::vec3(0.9f, 0.6f, 0.1f),
        glm::vec3(0.7f, 0.2f, 0.9f),
        glm::vec3(0.9f, 0.9f, 0.2f),
        glm::vec3(0.9f, 0.3f, 0.6f),
        glm::vec3(0.3f, 0.9f, 0.9f),
        glm::vec3(0.5f, 0.8f, 0.2f),
        glm::vec3(0.9f, 0.5f, 0.5f),
        glm::vec3(0.4f, 0.4f, 0.9f),
        glm::vec3(0.8f, 0.6f, 0.9f),
    };
    return palette[index % palette.size()];
}

//=============================================================================
// コンストラクタ・デストラクタ
//=============================================================================
VesselSegmentation::VesselSegmentation(int gridSize)
    : gridSize_(gridSize),
    rootSegmentId_(-1), skeletonVAO_(0), skeletonVBO_(0), buffersInitialized_(false) {
}

VesselSegmentation::~VesselSegmentation() {
    if (buffersInitialized_) {
        glDeleteVertexArrays(1, &skeletonVAO_);
        glDeleteBuffers(1, &skeletonVBO_);
    }
}

//=============================================================================
// OBJファイル読み込み
//=============================================================================
bool VesselSegmentation::loadOBJ(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << path << std::endl;
        return false;
    }

    meshVertices_.clear();
    meshIndices_.clear();

    std::vector<glm::vec3> vertices;
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "v") {
            float x, y, z;
            iss >> x >> y >> z;
            vertices.push_back(glm::vec3(x, y, z));
        }
        else if (type == "f") {
            std::vector<int> face;
            std::string vertex;
            while (iss >> vertex) {
                size_t pos = vertex.find('/');
                if (pos != std::string::npos) {
                    vertex = vertex.substr(0, pos);
                }
                face.push_back(std::stoi(vertex) - 1);
            }
            if (face.size() >= 3) {
                for (size_t i = 1; i < face.size() - 1; ++i) {
                    meshIndices_.push_back(static_cast<GLuint>(face[0]));
                    meshIndices_.push_back(static_cast<GLuint>(face[i]));
                    meshIndices_.push_back(static_cast<GLuint>(face[i + 1]));
                }
            }
        }
    }
    file.close();

    meshVertices_.reserve(vertices.size() * 3);
    for (const auto& v : vertices) {
        meshVertices_.push_back(v.x);
        meshVertices_.push_back(v.y);
        meshVertices_.push_back(v.z);
    }

    std::cout << "  Loaded: " << vertices.size() << " vertices, "
              << meshIndices_.size() / 3 << " triangles" << std::endl;

    return !meshVertices_.empty();
}

//=============================================================================
// OBJファイルから解析
//=============================================================================
bool VesselSegmentation::analyzeFromFile(const std::string& objPath) {
    std::cout << "=== Voxel-based Skeleton Extraction ===" << std::endl;
    std::cout << "Loading OBJ: " << objPath << std::endl;

    if (!loadOBJ(objPath)) {
        return false;
    }

    return analyze(meshVertices_, meshIndices_);
}

//=============================================================================
// 頂点・インデックスから解析
//=============================================================================
bool VesselSegmentation::analyze(const std::vector<GLfloat>& vertices,
                                 const std::vector<GLuint>& indices) {
    if (vertices.empty() || indices.empty()) {
        std::cerr << "Error: Empty mesh data" << std::endl;
        return false;
    }

    meshVertices_ = std::vector<GLfloat>(vertices.begin(), vertices.end());
    meshIndices_ = std::vector<GLuint>(indices.begin(), indices.end());

    std::cout << "Grid size: " << gridSize_ << "x" << gridSize_ << "x" << gridSize_ << std::endl;

    auto totalStart = std::chrono::high_resolution_clock::now();
    auto stepStart = totalStart;
    auto stepEnd = totalStart;

    // Step 1: BVH構築
    std::cout << "Step 1: Building BVH..." << std::flush;
    stepStart = std::chrono::high_resolution_clock::now();
    bvh_ = std::make_unique<SimpleBVH>(meshVertices_, meshIndices_);
    stepEnd = std::chrono::high_resolution_clock::now();
    std::cout << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(stepEnd - stepStart).count() << "ms)" << std::endl;

    // Step 2: ボクセルグリッド初期化
    std::cout << "Step 2: Initializing voxel grid..." << std::flush;
    stepStart = std::chrono::high_resolution_clock::now();
    initializeVoxelGrid();
    stepEnd = std::chrono::high_resolution_clock::now();
    std::cout << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(stepEnd - stepStart).count() << "ms)" << std::endl;

    // Step 3: 内部ボクセル判定
    std::cout << "Step 3: Classifying inside voxels..." << std::flush;
    stepStart = std::chrono::high_resolution_clock::now();
    classifyInsideVoxels();
    stepEnd = std::chrono::high_resolution_clock::now();
    std::cout << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(stepEnd - stepStart).count() << "ms)" << std::endl;

    // Step 4: 外部ボクセル削除
    std::cout << "Step 4: Carving external voxels..." << std::flush;
    stepStart = std::chrono::high_resolution_clock::now();
    carveExternalVoxels();
    stepEnd = std::chrono::high_resolution_clock::now();
    std::cout << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(stepEnd - stepStart).count() << "ms)" << std::endl;

    int insideCount = 0;
    for (int x = 0; x < gridSize_; x++) {
        for (int y = 0; y < gridSize_; y++) {
            for (int z = 0; z < gridSize_; z++) {
                if (insideVoxels_[x][y][z]) insideCount++;
            }
        }
    }
    std::cout << "  Inside voxels: " << insideCount << std::endl;

    if (insideCount == 0) {
        std::cerr << "Error: No inside voxels found" << std::endl;
        return false;
    }

    // Step 5: 距離変換
    std::cout << "Step 5: Computing distance transform..." << std::flush;
    stepStart = std::chrono::high_resolution_clock::now();
    computeDistanceTransform();
    stepEnd = std::chrono::high_resolution_clock::now();
    std::cout << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(stepEnd - stepStart).count() << "ms)" << std::endl;

    // Step 6: スケルトン抽出
    std::cout << "Step 6: Extracting skeleton..." << std::flush;
    stepStart = std::chrono::high_resolution_clock::now();
    extractSkeleton();
    stepEnd = std::chrono::high_resolution_clock::now();
    std::cout << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(stepEnd - stepStart).count() << "ms)" << std::endl;

    int skelCount = 0;
    for (int x = 0; x < gridSize_; x++) {
        for (int y = 0; y < gridSize_; y++) {
            for (int z = 0; z < gridSize_; z++) {
                if (skeletonVoxels_[x][y][z]) skelCount++;
            }
        }
    }
    std::cout << "  Skeleton voxels: " << skelCount << std::endl;

    // Step 7: スケルトングラフ構築
    std::cout << "Step 7: Building skeleton graph..." << std::flush;
    stepStart = std::chrono::high_resolution_clock::now();
    buildSkeletonGraph();
    stepEnd = std::chrono::high_resolution_clock::now();
    std::cout << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(stepEnd - stepStart).count() << "ms)" << std::endl;
    std::cout << "  Skeleton nodes: " << nodes_.size() << std::endl;

    // Step 7.5: グラフレベルでループ除去
    std::cout << "  Removing graph-level loops..." << std::flush;
    stepStart = std::chrono::high_resolution_clock::now();
    removeGraphLoops();
    stepEnd = std::chrono::high_resolution_clock::now();
    std::cout << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(stepEnd - stepStart).count() << "ms)" << std::endl;

    // Step 7.6: 不連続成分を接続
    stepStart = std::chrono::high_resolution_clock::now();
    int connected = connectDisconnectedComponents();
    stepEnd = std::chrono::high_resolution_clock::now();
    if (connected > 0) {
        std::cout << "  Connected " << connected << " disconnected components";
        std::cout << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(stepEnd - stepStart).count() << "ms)" << std::endl;
    }

    // Step 8: セグメント分割
    std::cout << "Step 8: Segmenting skeleton..." << std::flush;
    stepStart = std::chrono::high_resolution_clock::now();
    segmentSkeleton();
    stepEnd = std::chrono::high_resolution_clock::now();
    std::cout << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(stepEnd - stepStart).count() << "ms)" << std::endl;

    // Step 9: 三角形割り当て
    std::cout << "Step 9: Assigning triangles..." << std::flush;
    stepStart = std::chrono::high_resolution_clock::now();
    assignTrianglesToSegments();
    stepEnd = std::chrono::high_resolution_clock::now();
    std::cout << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(stepEnd - stepStart).count() << "ms)" << std::endl;

    // Step 10: 親子関係構築
    std::cout << "Step 10: Building hierarchy..." << std::flush;
    stepStart = std::chrono::high_resolution_clock::now();
    buildHierarchy();
    stepEnd = std::chrono::high_resolution_clock::now();
    std::cout << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(stepEnd - stepStart).count() << "ms)" << std::endl;

    // analyze() の最後（buildHierarchy()の後）に追加
    //autoExtendShortTerminalBranches(0.1f, 0.3f);  // 50%以下を80%まで延長


    assignColors();
    initSkeletonBuffers();

    // 木構造の検証
    verifyTreeStructure();

    // Voronoi 3Dセグメンテーションを構築
    buildVoronoi3D();

    // 距離ベースの結果をキャッシュ（現在の状態）
    triangleToSegment_Distance_ = triangleToSegment_;

    auto totalEnd = std::chrono::high_resolution_clock::now();
    std::cout << "\n=== Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart).count() << "ms ===" << std::endl;
    std::cout << "Analysis complete!" << std::endl;


    return !segments_.empty();
}

//=============================================================================
// ボクセルグリッド初期化
//=============================================================================
void VesselSegmentation::initializeVoxelGrid() {
    gridMin_ = glm::vec3(std::numeric_limits<float>::max());
    gridMax_ = glm::vec3(std::numeric_limits<float>::lowest());

    for (size_t i = 0; i < meshVertices_.size(); i += 3) {
        glm::vec3 v(meshVertices_[i], meshVertices_[i+1], meshVertices_[i+2]);
        gridMin_ = glm::min(gridMin_, v);
        gridMax_ = glm::max(gridMax_, v);
    }

    glm::vec3 size = gridMax_ - gridMin_;
    glm::vec3 padding = size * 0.1f;
    gridMin_ -= padding;
    gridMax_ += padding;

    voxelSize_ = (gridMax_ - gridMin_) / float(gridSize_);

    std::cout << "  Grid min: (" << gridMin_.x << ", " << gridMin_.y << ", " << gridMin_.z << ")" << std::endl;
    std::cout << "  Grid max: (" << gridMax_.x << ", " << gridMax_.y << ", " << gridMax_.z << ")" << std::endl;
    std::cout << "  Voxel size: " << voxelSize_.x << std::endl;

    insideVoxels_.resize(gridSize_);
    distanceField_.resize(gridSize_);
    skeletonVoxels_.resize(gridSize_);

    for (int x = 0; x < gridSize_; x++) {
        insideVoxels_[x].resize(gridSize_);
        distanceField_[x].resize(gridSize_);
        skeletonVoxels_[x].resize(gridSize_);
        for (int y = 0; y < gridSize_; y++) {
            insideVoxels_[x][y].resize(gridSize_, false);
            distanceField_[x][y].resize(gridSize_, 0.0f);
            skeletonVoxels_[x][y].resize(gridSize_, false);
        }
    }
}

//=============================================================================
// 内部ボクセル判定
//=============================================================================
void VesselSegmentation::classifyInsideVoxels() {
#ifdef _OPENMP
    std::cout << "  OpenMP threads: " << omp_get_max_threads() << std::endl;
#else
    std::cout << "  OpenMP: DISABLED" << std::endl;
#endif

    glm::vec3 rayDir(1.0f, 0.00001f, 0.0f);

#pragma omp parallel for collapse(3)
    for (int x = 0; x < gridSize_; x++) {
        for (int y = 0; y < gridSize_; y++) {
            for (int z = 0; z < gridSize_; z++) {
                glm::vec3 center = voxelCenter(x, y, z);
                int count = bvh_->countIntersections(center, rayDir);
                insideVoxels_[x][y][z] = (count % 2) == 1;
            }
        }
    }
}

//=============================================================================
// 外部ボクセル削除
//=============================================================================
void VesselSegmentation::carveExternalVoxels() {
    std::vector<std::vector<std::vector<bool>>> removed(gridSize_);
    for (int x = 0; x < gridSize_; x++) {
        removed[x].resize(gridSize_);
        for (int y = 0; y < gridSize_; y++) {
            removed[x][y].resize(gridSize_, false);
        }
    }

    bool changed = true;
    while (changed) {
        changed = false;

        for (int x = 0; x < gridSize_; x++) {
            for (int y = 0; y < gridSize_; y++) {
                for (int z = 0; z < gridSize_; z++) {
                    if (removed[x][y][z]) continue;
                    if (insideVoxels_[x][y][z]) continue;

                    bool exposed = (x == 0 || x == gridSize_-1 ||
                                    y == 0 || y == gridSize_-1 ||
                                    z == 0 || z == gridSize_-1);

                    if (!exposed) {
                        exposed = (x > 0 && removed[x-1][y][z]) ||
                                  (x < gridSize_-1 && removed[x+1][y][z]) ||
                                  (y > 0 && removed[x][y-1][z]) ||
                                  (y < gridSize_-1 && removed[x][y+1][z]) ||
                                  (z > 0 && removed[x][y][z-1]) ||
                                  (z < gridSize_-1 && removed[x][y][z+1]);
                    }

                    if (exposed) {
                        removed[x][y][z] = true;
                        changed = true;
                    }
                }
            }
        }
    }

    for (int x = 0; x < gridSize_; x++) {
        for (int y = 0; y < gridSize_; y++) {
            for (int z = 0; z < gridSize_; z++) {
                insideVoxels_[x][y][z] = !removed[x][y][z];
            }
        }
    }
}

//=============================================================================
// 距離変換
//=============================================================================
void VesselSegmentation::computeDistanceTransform() {
    const float INF = 1e10f;

    for (int x = 0; x < gridSize_; x++) {
        for (int y = 0; y < gridSize_; y++) {
            for (int z = 0; z < gridSize_; z++) {
                distanceField_[x][y][z] = insideVoxels_[x][y][z] ? INF : 0.0f;
            }
        }
    }

    // Forward pass
    for (int z = 0; z < gridSize_; z++) {
        for (int y = 0; y < gridSize_; y++) {
            for (int x = 0; x < gridSize_; x++) {
                if (!insideVoxels_[x][y][z]) continue;

                float minVal = distanceField_[x][y][z];

                for (int dz = -1; dz <= 0; dz++) {
                    for (int dy = (dz < 0 ? -1 : -1); dy <= (dz < 0 ? 1 : 0); dy++) {
                        for (int dx = -1; dx <= (dz < 0 || dy < 0 ? 1 : 0); dx++) {
                            if (dz == 0 && dy == 0 && dx >= 0) continue;

                            int nx = x + dx, ny = y + dy, nz = z + dz;
                            if (nx >= 0 && nx < gridSize_ && ny >= 0 && ny < gridSize_ && nz >= 0 && nz < gridSize_) {
                                float dist = std::sqrt(float(dx*dx + dy*dy + dz*dz));
                                minVal = std::min(minVal, distanceField_[nx][ny][nz] + dist);
                            }
                        }
                    }
                }
                distanceField_[x][y][z] = minVal;
            }
        }
    }

    // Backward pass
    for (int z = gridSize_ - 1; z >= 0; z--) {
        for (int y = gridSize_ - 1; y >= 0; y--) {
            for (int x = gridSize_ - 1; x >= 0; x--) {
                if (!insideVoxels_[x][y][z]) continue;

                float minVal = distanceField_[x][y][z];

                for (int dz = 0; dz <= 1; dz++) {
                    for (int dy = (dz > 0 ? -1 : 0); dy <= 1; dy++) {
                        for (int dx = (dz > 0 || dy > 0 ? -1 : 0); dx <= 1; dx++) {
                            if (dz == 0 && dy == 0 && dx <= 0) continue;

                            int nx = x + dx, ny = y + dy, nz = z + dz;
                            if (nx >= 0 && nx < gridSize_ && ny >= 0 && ny < gridSize_ && nz >= 0 && nz < gridSize_) {
                                float dist = std::sqrt(float(dx*dx + dy*dy + dz*dz));
                                minVal = std::min(minVal, distanceField_[nx][ny][nz] + dist);
                            }
                        }
                    }
                }
                distanceField_[x][y][z] = minVal;
            }
        }
    }
}

//=============================================================================
// ★★★ Voxel Thinning法 ★★★
//=============================================================================
void VesselSegmentation::extractSkeleton() {
    std::cout << "  [VOXEL_THINNING] Distance-based thinning..." << std::endl;

    // 内部ボクセルで初期化
    for (int x = 0; x < gridSize_; x++) {
        for (int y = 0; y < gridSize_; y++) {
            for (int z = 0; z < gridSize_; z++) {
                skeletonVoxels_[x][y][z] = insideVoxels_[x][y][z];
            }
        }
    }

    // 距離でソート
    struct VoxelDist {
        int x, y, z;
        float dist;
    };
    std::vector<VoxelDist> voxelsByDist;

    for (int x = 0; x < gridSize_; x++) {
        for (int y = 0; y < gridSize_; y++) {
            for (int z = 0; z < gridSize_; z++) {
                if (skeletonVoxels_[x][y][z]) {
                    voxelsByDist.push_back({x, y, z, distanceField_[x][y][z]});
                }
            }
        }
    }

    std::sort(voxelsByDist.begin(), voxelsByDist.end(),
              [](const VoxelDist& a, const VoxelDist& b) {
                  return a.dist < b.dist;
              });

    bool changed = true;
    int iteration = 0;
    int totalRemoved = 0;

    while (changed && iteration < 500) {
        changed = false;
        iteration++;

        for (const auto& v : voxelsByDist) {
            if (!skeletonVoxels_[v.x][v.y][v.z]) continue;

            if (canRemoveVoxel(v.x, v.y, v.z)) {
                skeletonVoxels_[v.x][v.y][v.z] = false;
                changed = true;
                totalRemoved++;
            }
        }
    }

    std::cout << "  Thinning iterations: " << iteration << std::endl;
    std::cout << "  Removed voxels: " << totalRemoved << std::endl;

    pruneShortBranches(3);
    connectNearEndpoints();

    // 孤立点除去
    for (int x = 0; x < gridSize_; x++) {
        for (int y = 0; y < gridSize_; y++) {
            for (int z = 0; z < gridSize_; z++) {
                if (skeletonVoxels_[x][y][z] && countNeighbors26(x, y, z) == 0) {
                    skeletonVoxels_[x][y][z] = false;
                }
            }
        }
    }

    // ★ ボクセルレベルでのループ除去は行わない ★
    // グラフ構築後にエッジレベルで除去する
}

//=============================================================================
// グラフレベルでのループ除去（ボクセルは削除しない、エッジだけ削除）
//=============================================================================
void VesselSegmentation::removeGraphLoops() {
    if (nodes_.empty()) return;

    std::cout << "  Removing graph-level loops..." << std::endl;

    int nodeCount = nodes_.size();

    // Union-Findでループを検出
    std::vector<int> parent(nodeCount);
    for (int i = 0; i < nodeCount; i++) parent[i] = i;

    std::function<int(int)> find = [&](int x) -> int {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    };

    // エッジを距離値の大きい順にソート（太いエッジを優先的に残す）
    struct Edge {
        int u, v;
        float minRadius;
    };
    std::vector<Edge> edges;

    for (int u = 0; u < nodeCount; u++) {
        for (int neighbor : nodes_[u].neighbors) {
            if (u < neighbor) {
                float r1 = nodes_[u].radius;
                float r2 = nodes_[neighbor].radius;
                edges.push_back({u, neighbor, std::min(r1, r2)});
            }
        }
    }

    // 太いエッジを優先（Kruskal's algorithm で最大全域木）
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        return a.minRadius > b.minRadius;
    });

    std::set<std::pair<int, int>> edgesToRemove;

    for (const auto& e : edges) {
        int pu = find(e.u);
        int pv = find(e.v);

        if (pu == pv) {
            // ループを形成するエッジ → 削除対象
            edgesToRemove.insert({std::min(e.u, e.v), std::max(e.u, e.v)});
        } else {
            // 新しい接続 → 保持
            parent[pu] = pv;
        }
    }

    // グラフからエッジを削除（ボクセルは削除しない）
    int removedCount = 0;
    for (const auto& ep : edgesToRemove) {
        int u = ep.first;
        int v = ep.second;

        // u -> v を削除
        auto& neighborsU = nodes_[u].neighbors;
        auto itU = std::find(neighborsU.begin(), neighborsU.end(), v);
        if (itU != neighborsU.end()) {
            neighborsU.erase(itU);
        }

        // v -> u を削除
        auto& neighborsV = nodes_[v].neighbors;
        auto itV = std::find(neighborsV.begin(), neighborsV.end(), u);
        if (itV != neighborsV.end()) {
            neighborsV.erase(itV);
        }

        removedCount++;
    }

    std::cout << "  Loop edges removed: " << removedCount << std::endl;
}

bool VesselSegmentation::canRemoveVoxel(int x, int y, int z) {
    if (!skeletonVoxels_[x][y][z]) return false;

    int neighbors = countNeighbors26(x, y, z);
    if (neighbors == 0) return true;
    if (neighbors == 1) return false;

    skeletonVoxels_[x][y][z] = false;

    std::vector<glm::ivec3> neighborVoxels;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                int nx = x + dx, ny = y + dy, nz = z + dz;
                if (isValidVoxel(nx, ny, nz) && skeletonVoxels_[nx][ny][nz]) {
                    neighborVoxels.push_back(glm::ivec3(nx, ny, nz));
                }
            }
        }
    }

    bool canRemove = true;

    if (neighborVoxels.size() >= 2) {
        std::vector<bool> visited(neighborVoxels.size(), false);
        std::queue<int> queue;
        queue.push(0);
        visited[0] = true;
        int visitedCount = 1;

        while (!queue.empty()) {
            int idx = queue.front();
            queue.pop();

            glm::ivec3 curr = neighborVoxels[idx];

            for (size_t i = 0; i < neighborVoxels.size(); i++) {
                if (visited[i]) continue;

                glm::ivec3 other = neighborVoxels[i];
                glm::ivec3 diff = other - curr;

                bool connected = false;

                if (std::abs(diff.x) <= 1 && std::abs(diff.y) <= 1 && std::abs(diff.z) <= 1) {
                    connected = true;
                }

                if (!connected) {
                    for (int dx = -1; dx <= 1 && !connected; dx++) {
                        for (int dy = -1; dy <= 1 && !connected; dy++) {
                            for (int dz = -1; dz <= 1 && !connected; dz++) {
                                if (dx == 0 && dy == 0 && dz == 0) continue;

                                int mx = curr.x + dx, my = curr.y + dy, mz = curr.z + dz;
                                if (mx == x && my == y && mz == z) continue;

                                if (isValidVoxel(mx, my, mz) && skeletonVoxels_[mx][my][mz]) {
                                    glm::ivec3 diff2 = other - glm::ivec3(mx, my, mz);
                                    if (std::abs(diff2.x) <= 1 && std::abs(diff2.y) <= 1 && std::abs(diff2.z) <= 1) {
                                        connected = true;
                                    }
                                }
                            }
                        }
                    }
                }

                if (connected) {
                    visited[i] = true;
                    visitedCount++;
                    queue.push(i);
                }
            }
        }

        canRemove = (visitedCount == static_cast<int>(neighborVoxels.size()));
    }

    skeletonVoxels_[x][y][z] = true;
    return canRemove;
}

int VesselSegmentation::countNeighbors26(int x, int y, int z) {
    int count = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                int nx = x + dx, ny = y + dy, nz = z + dz;
                if (isValidVoxel(nx, ny, nz) && skeletonVoxels_[nx][ny][nz]) {
                    count++;
                }
            }
        }
    }
    return count;
}

void VesselSegmentation::pruneShortBranches(int minLength) {
    bool changed = true;
    int totalPruned = 0;

    while (changed) {
        changed = false;

        std::vector<glm::ivec3> endpoints;
        for (int x = 1; x < gridSize_ - 1; x++) {
            for (int y = 1; y < gridSize_ - 1; y++) {
                for (int z = 1; z < gridSize_ - 1; z++) {
                    if (skeletonVoxels_[x][y][z] && countNeighbors26(x, y, z) == 1) {
                        endpoints.push_back(glm::ivec3(x, y, z));
                    }
                }
            }
        }

        for (const auto& start : endpoints) {
            if (!skeletonVoxels_[start.x][start.y][start.z]) continue;
            if (countNeighbors26(start.x, start.y, start.z) != 1) continue;

            std::vector<glm::ivec3> path;
            int cx = start.x, cy = start.y, cz = start.z;
            int px = -1, py = -1, pz = -1;

            while (true) {
                path.push_back(glm::ivec3(cx, cy, cz));
                if (path.size() > static_cast<size_t>(minLength + 1)) break;

                int nextX = -1, nextY = -1, nextZ = -1;
                int neighborCount = 0;

                for (int dx = -1; dx <= 1; dx++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dz = -1; dz <= 1; dz++) {
                            if (dx == 0 && dy == 0 && dz == 0) continue;
                            int nx = cx + dx, ny = cy + dy, nz = cz + dz;

                            if (isValidVoxel(nx, ny, nz) && skeletonVoxels_[nx][ny][nz]) {
                                if (nx != px || ny != py || nz != pz) {
                                    neighborCount++;
                                    nextX = nx; nextY = ny; nextZ = nz;
                                }
                            }
                        }
                    }
                }

                if (neighborCount == 0) break;
                if (neighborCount >= 2) break;

                px = cx; py = cy; pz = cz;
                cx = nextX; cy = nextY; cz = nextZ;
            }

            if (static_cast<int>(path.size()) <= minLength) {
                for (const auto& v : path) {
                    int nc = countNeighbors26(v.x, v.y, v.z);
                    if (nc <= 2) {
                        skeletonVoxels_[v.x][v.y][v.z] = false;
                        changed = true;
                        totalPruned++;
                    }
                }
            }
        }
    }

    if (totalPruned > 0) {
        std::cout << "  Pruned short branches: " << totalPruned << " voxels" << std::endl;
    }
}

void VesselSegmentation::connectNearEndpoints() {
    std::vector<glm::ivec3> endpoints;

    for (int x = 0; x < gridSize_; x++) {
        for (int y = 0; y < gridSize_; y++) {
            for (int z = 0; z < gridSize_; z++) {
                if (skeletonVoxels_[x][y][z] && countNeighbors26(x, y, z) == 1) {
                    endpoints.push_back(glm::ivec3(x, y, z));
                }
            }
        }
    }

    // グリッドサイズに比例した接続距離（100グリッドで8ボクセル相当）
    float maxConnectDist = 8.0f * (gridSize_ / 100.0f);

    for (size_t i = 0; i < endpoints.size(); i++) {
        for (size_t j = i + 1; j < endpoints.size(); j++) {
            glm::vec3 v1(endpoints[i]);
            glm::vec3 v2(endpoints[j]);
            float dist = glm::length(v2 - v1);

            if (dist < maxConnectDist && dist > 1.5f) {
                glm::vec3 dir = glm::normalize(v2 - v1);
                int steps = static_cast<int>(dist);

                bool canConnect = true;
                std::vector<glm::ivec3> linePath;

                for (int s = 1; s < steps; s++) {
                    glm::vec3 p = v1 + dir * float(s);
                    int px = static_cast<int>(p.x + 0.5f);
                    int py = static_cast<int>(p.y + 0.5f);
                    int pz = static_cast<int>(p.z + 0.5f);

                    if (!isValidVoxel(px, py, pz) || !insideVoxels_[px][py][pz]) {
                        canConnect = false;
                        break;
                    }

                    linePath.push_back(glm::ivec3(px, py, pz));
                }

                if (canConnect) {
                    for (const auto& p : linePath) {
                        skeletonVoxels_[p.x][p.y][p.z] = true;
                    }
                }
            }
        }
    }
}

//=============================================================================
// スケルトングラフ構築（26近傍接続）
//=============================================================================
void VesselSegmentation::buildSkeletonGraph() {
    nodes_.clear();
    voxelToNode_.clear();

    int nodeId = 0;
    for (int x = 0; x < gridSize_; x++) {
        for (int y = 0; y < gridSize_; y++) {
            for (int z = 0; z < gridSize_; z++) {
                if (skeletonVoxels_[x][y][z]) {
                    SkeletonNode node;
                    node.id = nodeId;
                    node.voxelIndex = glm::ivec3(x, y, z);
                    node.position = voxelCenter(x, y, z);
                    node.radius = distanceField_[x][y][z] * glm::length(voxelSize_);

                    voxelToNode_[voxelIndex1D(x, y, z)] = nodeId;
                    nodes_.push_back(node);
                    nodeId++;
                }
            }
        }
    }

    // 26近傍接続
    for (auto& node : nodes_) {
        int x = node.voxelIndex.x;
        int y = node.voxelIndex.y;
        int z = node.voxelIndex.z;

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    int nx = x + dx, ny = y + dy, nz = z + dz;

                    if (isValidVoxel(nx, ny, nz)) {
                        int idx1D = voxelIndex1D(nx, ny, nz);
                        auto it = voxelToNode_.find(idx1D);
                        if (it != voxelToNode_.end()) {
                            node.neighbors.push_back(it->second);
                        }
                    }
                }
            }
        }
    }
}

//=============================================================================
// セグメント分割
//=============================================================================
//=============================================================================
// セグメント分割
//=============================================================================
void VesselSegmentation::segmentSkeleton() {
    segments_.clear();
    if (nodes_.empty()) return;

    for (auto& node : nodes_) {
        node.segmentId = -1;
    }

    // 分岐点検出：3つ以上の隣接を持つノードは全て分岐点
    std::set<int> branchNodes;

    for (size_t i = 0; i < nodes_.size(); i++) {
        if (nodes_[i].neighbors.size() >= 3) {
            branchNodes.insert(i);
        }

        // ★ 人工接続の端点も分岐点として扱う
        for (int neighborId : nodes_[i].neighbors) {
            int minNode = std::min(static_cast<int>(i), neighborId);
            int maxNode = std::max(static_cast<int>(i), neighborId);
            if (artificialConnections_.count({minNode, maxNode}) > 0) {
                branchNodes.insert(i);
            }
        }
    }

    std::cout << "  Branch nodes (>=3 neighbors or artificial connection): " << branchNodes.size() << std::endl;

    int segmentId = 0;

    // 分岐点以外のノードをBFSでセグメント化
    for (size_t i = 0; i < nodes_.size(); i++) {
        if (nodes_[i].segmentId >= 0) continue;
        if (branchNodes.count(i) > 0) continue;

        Segment segment;
        segment.id = segmentId;

        std::queue<int> queue;
        queue.push(i);
        nodes_[i].segmentId = segmentId;

        float sumRadius = 0;

        while (!queue.empty()) {
            int curr = queue.front();
            queue.pop();

            segment.nodeIds.push_back(curr);
            sumRadius += nodes_[curr].radius;

            for (int neighborId : nodes_[curr].neighbors) {
                // ★ 人工接続は跨がない
                int minNode = std::min(curr, neighborId);
                int maxNode = std::max(curr, neighborId);
                if (artificialConnections_.count({minNode, maxNode}) > 0) {
                    continue;
                }

                // 分岐点は越えない、未割り当てのノードのみ追加
                if (nodes_[neighborId].segmentId < 0 && branchNodes.count(neighborId) == 0) {
                    nodes_[neighborId].segmentId = segmentId;
                    queue.push(neighborId);
                }
            }
        }

        segment.averageRadius = sumRadius / segment.nodeIds.size();
        segments_.push_back(segment);
        segmentId++;
    }

    // 分岐点を最も太いセグメントに割り当て、接続を記録
    segmentConnections_.clear();

    for (int bn : branchNodes) {
        std::map<int, float> connectedSegmentsRadius;

        for (int neighborId : nodes_[bn].neighbors) {
            // ★ 人工接続先は別セグメントとして扱う（同じセグメントに統合しない）
            int minNode = std::min(bn, neighborId);
            int maxNode = std::max(bn, neighborId);
            if (artificialConnections_.count({minNode, maxNode}) > 0) {
                continue;  // 人工接続先はスキップ
            }

            int segId = nodes_[neighborId].segmentId;
            if (segId >= 0) {
                if (connectedSegmentsRadius.find(segId) == connectedSegmentsRadius.end()) {
                    connectedSegmentsRadius[segId] = segments_[segId].averageRadius;
                }
            }
        }

        int bestSegment = -1;
        float maxRadius = -1;

        for (const auto& pair : connectedSegmentsRadius) {
            if (pair.second > maxRadius) {
                maxRadius = pair.second;
                bestSegment = pair.first;
            }
        }

        if (bestSegment >= 0) {
            nodes_[bn].segmentId = bestSegment;
            segments_[bestSegment].nodeIds.push_back(bn);

            for (const auto& pair : connectedSegmentsRadius) {
                int seg = pair.first;
                if (seg != bestSegment) {
                    segmentConnections_[bestSegment].insert(seg);
                    segmentConnections_[seg].insert(bestSegment);
                }
            }
        } else {
            // ★ 人工接続のみの分岐点は新しいセグメントとして作成
            Segment segment;
            segment.id = segmentId;
            segment.nodeIds.push_back(bn);
            segment.averageRadius = nodes_[bn].radius;
            nodes_[bn].segmentId = segmentId;
            segments_.push_back(segment);
            segmentId++;
        }
    }

    // ★ 人工接続によるセグメント間の接続を記録（階層構築用）
    for (const auto& conn : artificialConnections_) {
        int node1 = conn.first;
        int node2 = conn.second;
        int seg1 = nodes_[node1].segmentId;
        int seg2 = nodes_[node2].segmentId;

        if (seg1 >= 0 && seg2 >= 0 && seg1 != seg2) {
            segmentConnections_[seg1].insert(seg2);
            segmentConnections_[seg2].insert(seg1);
        }
    }

    mergeSmallSegments(3);

    std::cout << "  Final segments: " << segments_.size() << std::endl;
}

// void VesselSegmentation::segmentSkeleton() {
//     segments_.clear();
//     if (nodes_.empty()) return;

//     for (auto& node : nodes_) {
//         node.segmentId = -1;
//     }

//     // 分岐点検出：3つ以上の隣接を持つノードは全て分岐点
//     std::set<int> branchNodes;

//     for (size_t i = 0; i < nodes_.size(); i++) {
//         if (nodes_[i].neighbors.size() >= 3) {
//             branchNodes.insert(i);
//         }
//     }

//     std::cout << "  Branch nodes (>=3 neighbors): " << branchNodes.size() << std::endl;

//     int segmentId = 0;

//     // 分岐点以外のノードをBFSでセグメント化
//     for (size_t i = 0; i < nodes_.size(); i++) {
//         if (nodes_[i].segmentId >= 0) continue;
//         if (branchNodes.count(i) > 0) continue;

//         Segment segment;
//         segment.id = segmentId;

//         std::queue<int> queue;
//         queue.push(i);
//         nodes_[i].segmentId = segmentId;

//         float sumRadius = 0;

//         while (!queue.empty()) {
//             int curr = queue.front();
//             queue.pop();

//             segment.nodeIds.push_back(curr);
//             sumRadius += nodes_[curr].radius;

//             for (int neighborId : nodes_[curr].neighbors) {
//                 // 分岐点は越えない、未割り当てのノードのみ追加
//                 if (nodes_[neighborId].segmentId < 0 && branchNodes.count(neighborId) == 0) {
//                     nodes_[neighborId].segmentId = segmentId;
//                     queue.push(neighborId);
//                 }
//             }
//         }

//         segment.averageRadius = sumRadius / segment.nodeIds.size();
//         segments_.push_back(segment);
//         segmentId++;
//     }

//     // 分岐点を最も太いセグメントに割り当て、接続を記録
//     segmentConnections_.clear();

//     for (int bn : branchNodes) {
//         std::map<int, float> connectedSegmentsRadius;

//         for (int neighborId : nodes_[bn].neighbors) {
//             int segId = nodes_[neighborId].segmentId;
//             if (segId >= 0) {
//                 if (connectedSegmentsRadius.find(segId) == connectedSegmentsRadius.end()) {
//                     connectedSegmentsRadius[segId] = segments_[segId].averageRadius;
//                 }
//             }
//         }

//         int bestSegment = -1;
//         float maxRadius = -1;

//         for (const auto& pair : connectedSegmentsRadius) {
//             if (pair.second > maxRadius) {
//                 maxRadius = pair.second;
//                 bestSegment = pair.first;
//             }
//         }

//         if (bestSegment >= 0) {
//             nodes_[bn].segmentId = bestSegment;
//             segments_[bestSegment].nodeIds.push_back(bn);

//             for (const auto& pair : connectedSegmentsRadius) {
//                 int seg = pair.first;
//                 if (seg != bestSegment) {
//                     segmentConnections_[bestSegment].insert(seg);
//                     segmentConnections_[seg].insert(bestSegment);
//                 }
//             }
//         }
//     }

//     mergeSmallSegments(3);

//     std::cout << "  Final segments: " << segments_.size() << std::endl;
// }

void VesselSegmentation::mergeSmallSegments(int minNodes) {
    bool changed = true;

    while (changed) {
        changed = false;

        for (size_t i = 0; i < segments_.size(); i++) {
            if (segments_[i].nodeIds.size() <= static_cast<size_t>(minNodes) &&
                segments_[i].nodeIds.size() > 0) {

                int bestNeighbor = -1;
                float maxRadius = -1;

                for (int seg : segmentConnections_[i]) {
                    if (seg != static_cast<int>(i) && segments_[seg].nodeIds.size() > 0) {
                        if (segments_[seg].averageRadius > maxRadius) {
                            maxRadius = segments_[seg].averageRadius;
                            bestNeighbor = seg;
                        }
                    }
                }

                if (bestNeighbor >= 0) {
                    for (int nodeId : segments_[i].nodeIds) {
                        nodes_[nodeId].segmentId = bestNeighbor;
                        segments_[bestNeighbor].nodeIds.push_back(nodeId);
                    }

                    for (int neighbor : segmentConnections_[i]) {
                        if (neighbor != bestNeighbor) {
                            segmentConnections_[bestNeighbor].insert(neighbor);
                            segmentConnections_[neighbor].erase(i);
                            segmentConnections_[neighbor].insert(bestNeighbor);
                        }
                    }

                    segments_[i].nodeIds.clear();
                    segmentConnections_[i].clear();
                    changed = true;
                }
            }
        }
    }

    std::vector<Segment> newSegments;
    std::map<int, int> oldToNew;

    for (size_t i = 0; i < segments_.size(); i++) {
        if (!segments_[i].nodeIds.empty()) {
            int newId = newSegments.size();
            oldToNew[i] = newId;
            segments_[i].id = newId;
            newSegments.push_back(segments_[i]);
        }
    }

    for (auto& node : nodes_) {
        if (node.segmentId >= 0) {
            auto it = oldToNew.find(node.segmentId);
            if (it != oldToNew.end()) {
                node.segmentId = it->second;
            }
        }
    }

    std::map<int, std::set<int>> newConnections;
    for (const auto& pair : segmentConnections_) {
        auto itOld = oldToNew.find(pair.first);
        if (itOld != oldToNew.end()) {
            for (int neighbor : pair.second) {
                auto itNeighbor = oldToNew.find(neighbor);
                if (itNeighbor != oldToNew.end() && itOld->second != itNeighbor->second) {
                    newConnections[itOld->second].insert(itNeighbor->second);
                }
            }
        }
    }

    segments_ = newSegments;
    segmentConnections_ = newConnections;
}

void VesselSegmentation::assignTrianglesToSegments() {
    triangleToSegment_.clear();
    for (auto& seg : segments_) {
        seg.triangleIndices.clear();
    }

    int numTriangles = meshIndices_.size() / 3;

    // Step 1: 三角形の隣接関係を構築（エッジを共有する三角形）
    std::map<std::pair<int,int>, std::vector<int>> edgeToTriangles;

    auto makeEdge = [](int a, int b) {
        return std::make_pair(std::min(a, b), std::max(a, b));
    };

    for (int t = 0; t < numTriangles; t++) {
        int i0 = meshIndices_[t * 3];
        int i1 = meshIndices_[t * 3 + 1];
        int i2 = meshIndices_[t * 3 + 2];

        edgeToTriangles[makeEdge(i0, i1)].push_back(t);
        edgeToTriangles[makeEdge(i1, i2)].push_back(t);
        edgeToTriangles[makeEdge(i2, i0)].push_back(t);
    }

    // 三角形の隣接リスト
    std::vector<std::vector<int>> triangleNeighbors(numTriangles);
    for (const auto& pair : edgeToTriangles) {
        const auto& tris = pair.second;
        for (size_t i = 0; i < tris.size(); i++) {
            for (size_t j = i + 1; j < tris.size(); j++) {
                triangleNeighbors[tris[i]].push_back(tris[j]);
                triangleNeighbors[tris[j]].push_back(tris[i]);
            }
        }
    }

    // Step 2: 各三角形の中心を計算
    std::vector<glm::vec3> triangleCentroids(numTriangles);
    for (int t = 0; t < numTriangles; t++) {
        int i0 = meshIndices_[t * 3];
        int i1 = meshIndices_[t * 3 + 1];
        int i2 = meshIndices_[t * 3 + 2];

        glm::vec3 v0(meshVertices_[i0*3], meshVertices_[i0*3+1], meshVertices_[i0*3+2]);
        glm::vec3 v1(meshVertices_[i1*3], meshVertices_[i1*3+1], meshVertices_[i1*3+2]);
        glm::vec3 v2(meshVertices_[i2*3], meshVertices_[i2*3+1], meshVertices_[i2*3+2]);

        triangleCentroids[t] = (v0 + v1 + v2) / 3.0f;
    }

    // Step 3: 各三角形に最も近いスケルトンノードを見つけ、その距離とセグメントを記録
    std::vector<int> triangleSegment(numTriangles, -1);
    std::vector<float> triangleToSkeleton(numTriangles, std::numeric_limits<float>::max());

    for (int t = 0; t < numTriangles; t++) {
        int bestNode = -1;
        float minDist = std::numeric_limits<float>::max();

        for (const auto& node : nodes_) {
            float dist = glm::length(triangleCentroids[t] - node.position);
            if (dist < minDist) {
                minDist = dist;
                bestNode = node.id;
            }
        }

        if (bestNode >= 0 && nodes_[bestNode].segmentId >= 0) {
            triangleToSkeleton[t] = minDist;
            triangleSegment[t] = nodes_[bestNode].segmentId;
        }
    }

    // Step 4: Dijkstraライクな伝播（スケルトンに近い三角形を優先）
    // priority queue: (スケルトンまでの距離, 三角形ID)
    std::priority_queue<
        std::pair<float, int>,
        std::vector<std::pair<float, int>>,
        std::greater<std::pair<float, int>>
        > pq;

    std::vector<bool> finalized(numTriangles, false);

    // 全ての三角形をキューに追加
    for (int t = 0; t < numTriangles; t++) {
        if (triangleSegment[t] >= 0) {
            pq.push({triangleToSkeleton[t], t});
        }
    }

    while (!pq.empty()) {
        auto [dist, t] = pq.top();
        pq.pop();

        if (finalized[t]) continue;
        finalized[t] = true;

        int currSeg = triangleSegment[t];

        // 隣接三角形をチェック
        for (int neighbor : triangleNeighbors[t]) {
            if (finalized[neighbor]) continue;

            // 隣接三角形のスケルトンへの最短距離
            float neighborSkeletonDist = triangleToSkeleton[neighbor];

            // 現在の三角形のセグメントのスケルトンノードへの距離
            float distToCurrentSeg = std::numeric_limits<float>::max();
            for (const auto& node : nodes_) {
                if (node.segmentId == currSeg) {
                    float d = glm::length(triangleCentroids[neighbor] - node.position);
                    distToCurrentSeg = std::min(distToCurrentSeg, d);
                }
            }

            // 現在のセグメントからの距離が、他のどのセグメントよりも近い場合のみ伝播
            if (distToCurrentSeg <= neighborSkeletonDist * 1.2f) {  // 20%のマージン
                if (triangleSegment[neighbor] < 0 || distToCurrentSeg < triangleToSkeleton[neighbor]) {
                    triangleSegment[neighbor] = currSeg;
                    triangleToSkeleton[neighbor] = distToCurrentSeg;
                    pq.push({distToCurrentSeg, neighbor});
                }
            }
        }
    }

    // Step 5: まだ未割り当ての三角形は最も近いノードで割り当て
    for (int t = 0; t < numTriangles; t++) {
        if (triangleSegment[t] < 0) {
            int bestNode = -1;
            float minDist = std::numeric_limits<float>::max();

            for (const auto& node : nodes_) {
                float dist = glm::length(triangleCentroids[t] - node.position);
                if (dist < minDist) {
                    minDist = dist;
                    bestNode = node.id;
                }
            }

            if (bestNode >= 0 && nodes_[bestNode].segmentId >= 0) {
                triangleSegment[t] = nodes_[bestNode].segmentId;
            }
        }
    }

    // Step 6: 結果を格納
    for (int t = 0; t < numTriangles; t++) {
        int segId = triangleSegment[t];
        if (segId >= 0 && segId < static_cast<int>(segments_.size())) {
            triangleToSegment_[t] = segId;
            segments_[segId].triangleIndices.insert(t);
        }
    }
}

void VesselSegmentation::buildHierarchy() {
    if (segments_.empty()) return;

    for (auto& seg : segments_) {
        seg.parentId = -1;
        seg.childIds.clear();
        seg.hierarchyLevel = 0;
    }

    rootSegmentId_ = 0;
    float maxRadius = segments_[0].averageRadius;

    for (size_t i = 1; i < segments_.size(); i++) {
        if (segments_[i].averageRadius > maxRadius) {
            maxRadius = segments_[i].averageRadius;
            rootSegmentId_ = i;
        }
    }

    std::cout << "  Root segment: " << rootSegmentId_ << " (radius: " << maxRadius << ")" << std::endl;

    for (const auto& node : nodes_) {
        for (int neighborId : node.neighbors) {
            int seg1 = node.segmentId;
            int seg2 = nodes_[neighborId].segmentId;
            if (seg1 >= 0 && seg2 >= 0 && seg1 != seg2) {
                segmentConnections_[seg1].insert(seg2);
                segmentConnections_[seg2].insert(seg1);
            }
        }
    }

    std::vector<bool> visited(segments_.size(), false);
    std::queue<int> queue;

    queue.push(rootSegmentId_);
    visited[rootSegmentId_] = true;
    segments_[rootSegmentId_].parentId = -1;
    segments_[rootSegmentId_].hierarchyLevel = 0;

    while (!queue.empty()) {
        int currentId = queue.front();
        queue.pop();

        std::vector<std::pair<float, int>> neighborsByRadius;
        for (int neighborId : segmentConnections_[currentId]) {
            if (!visited[neighborId]) {
                neighborsByRadius.push_back({segments_[neighborId].averageRadius, neighborId});
            }
        }

        std::sort(neighborsByRadius.begin(), neighborsByRadius.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        for (const auto& pair : neighborsByRadius) {
            int neighborId = pair.second;
            if (!visited[neighborId]) {
                visited[neighborId] = true;
                segments_[neighborId].parentId = currentId;
                segments_[neighborId].hierarchyLevel = segments_[currentId].hierarchyLevel + 1;
                segments_[currentId].childIds.push_back(neighborId);
                queue.push(neighborId);
            }
        }
    }

    int unconnected = 0;
    for (size_t i = 0; i < segments_.size(); i++) {
        if (!visited[i]) {
            unconnected++;
            float minDist = std::numeric_limits<float>::max();
            int nearestSeg = -1;

            if (!segments_[i].nodeIds.empty()) {
                glm::vec3 center1(0);
                for (int nodeId : segments_[i].nodeIds) {
                    center1 += nodes_[nodeId].position;
                }
                center1 /= float(segments_[i].nodeIds.size());

                for (size_t j = 0; j < segments_.size(); j++) {
                    if (i == j || !visited[j] || segments_[j].nodeIds.empty()) continue;

                    glm::vec3 center2(0);
                    for (int nodeId : segments_[j].nodeIds) {
                        center2 += nodes_[nodeId].position;
                    }
                    center2 /= float(segments_[j].nodeIds.size());

                    float dist = glm::length(center1 - center2);
                    if (dist < minDist) {
                        minDist = dist;
                        nearestSeg = j;
                    }
                }

                if (nearestSeg >= 0) {
                    visited[i] = true;
                    segments_[i].parentId = nearestSeg;
                    segments_[i].hierarchyLevel = segments_[nearestSeg].hierarchyLevel + 1;
                    segments_[nearestSeg].childIds.push_back(i);
                }
            }
        }
    }

    if (unconnected > 0) {
        std::cout << "  Connected " << unconnected << " isolated segments" << std::endl;
    }

    int maxLevel = 0;
    int withChildren = 0;
    for (const auto& seg : segments_) {
        maxLevel = std::max(maxLevel, seg.hierarchyLevel);
        if (!seg.childIds.empty()) withChildren++;
    }
    std::cout << "  Max hierarchy level: " << maxLevel << std::endl;
    std::cout << "  Segments with children: " << withChildren << std::endl;
}

//=============================================================================
// ヘルパー関数
//=============================================================================
int VesselSegmentation::voxelIndex1D(int x, int y, int z) const {
    return x + y * gridSize_ + z * gridSize_ * gridSize_;
}

glm::vec3 VesselSegmentation::voxelCenter(int x, int y, int z) const {
    return gridMin_ + voxelSize_ * (glm::vec3(x, y, z) + 0.5f);
}

bool VesselSegmentation::isValidVoxel(int x, int y, int z) const {
    return x >= 0 && x < gridSize_ && y >= 0 && y < gridSize_ && z >= 0 && z < gridSize_;
}

void VesselSegmentation::assignColors() {
    for (size_t i = 0; i < segments_.size(); i++) {
        segments_[i].color = getColorForIndex(i);
    }
}

int VesselSegmentation::getSegmentByTriangle(int triangleIndex) const {
    auto it = triangleToSegment_.find(triangleIndex);
    return (it != triangleToSegment_.end()) ? it->second : -1;
}

void VesselSegmentation::selectByTriangle(int triangleIndex) {
    int segId = getSegmentByTriangle(triangleIndex);
    if (segId >= 0) selectSegment(segId);
}

void VesselSegmentation::selectSegment(int segmentId) {
    if (segmentId < 0 || segmentId >= (int)segments_.size()) {
        clearSelection();
        return;
    }

    selection_.selectedSegmentId = segmentId;
    selection_.selectedSegments.clear();
    selection_.selectedSegments.insert(segmentId);
    collectDownstream(segmentId, selection_.selectedSegments);

    std::cout << "Selected segment " << segmentId
              << " (" << segments_[segmentId].triangleIndices.size() << " triangles)"
              << " with " << (selection_.selectedSegments.size() - 1)
              << " downstream segments" << std::endl;
}

void VesselSegmentation::clearSelection() {
    selection_.clear();
}

void VesselSegmentation::collectDownstream(int segmentId, std::set<int>& result) const {
    if (segmentId < 0 || segmentId >= (int)segments_.size()) return;

    for (int childId : segments_[segmentId].childIds) {
        if (result.find(childId) == result.end()) {
            result.insert(childId);
            collectDownstream(childId, result);
        }
    }
}

//=============================================================================
// 描画
//=============================================================================

void VesselSegmentation::initSkeletonBuffers() {
    if (buffersInitialized_) {
        glDeleteVertexArrays(1, &skeletonVAO_);
        glDeleteBuffers(1, &skeletonVBO_);
    }

    glGenVertexArrays(1, &skeletonVAO_);
    glGenBuffers(1, &skeletonVBO_);
    buffersInitialized_ = true;

    updateSkeletonBuffers();
}

void VesselSegmentation::updateSkeletonBuffers() {
    if (!buffersInitialized_) return;

    std::vector<GLfloat> lineData;

    for (const auto& node : nodes_) {
        for (int neighborId : node.neighbors) {
            if (neighborId > node.id) {
                lineData.push_back(node.position.x);
                lineData.push_back(node.position.y);
                lineData.push_back(node.position.z);

                lineData.push_back(nodes_[neighborId].position.x);
                lineData.push_back(nodes_[neighborId].position.y);
                lineData.push_back(nodes_[neighborId].position.z);
            }
        }
    }

    glBindVertexArray(skeletonVAO_);
    glBindBuffer(GL_ARRAY_BUFFER, skeletonVBO_);
    glBufferData(GL_ARRAY_BUFFER, lineData.size() * sizeof(GLfloat), lineData.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
}


//=============================================================================
// デバッグ機能：トポロジー解析（ループと不連続の検出）
//=============================================================================
VesselSegmentation::DebugInfo VesselSegmentation::analyzeTopology() {
    DebugInfo info;
    info.loopCount = 0;
    info.componentCount = 0;
    info.hasDisconnection = false;

    if (nodes_.empty()) {
        std::cout << "\n=== Topology Analysis ===" << std::endl;
        std::cout << "  No nodes to analyze" << std::endl;
        return info;
    }

    std::cout << "\n=== Topology Analysis ===" << std::endl;
    std::cout << "  Total nodes: " << nodes_.size() << std::endl;

    // 1. 連結成分の検出（Union-Find）
    std::vector<int> parent(nodes_.size());
    std::vector<int> rank(nodes_.size(), 0);

    // 初期化
    for (size_t i = 0; i < nodes_.size(); i++) {
        parent[i] = i;
    }

    // Find with path compression
    std::function<int(int)> find = [&](int x) -> int {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    };

    // Union by rank
    auto unite = [&](int x, int y) {
        int px = find(x);
        int py = find(y);
        if (px == py) return false; // 既に同じ成分

        if (rank[px] < rank[py]) std::swap(px, py);
        parent[py] = px;
        if (rank[px] == rank[py]) rank[px]++;
        return true;
    };

    // 2. ループ検出（DFSでバックエッジを探す）
    std::vector<bool> visited(nodes_.size(), false);
    std::vector<bool> inStack(nodes_.size(), false);
    std::vector<int> parentNode(nodes_.size(), -1);

    std::function<void(int)> detectLoopDFS = [&](int node) {
        visited[node] = true;
        inStack[node] = true;

        for (int neighbor : nodes_[node].neighbors) {
            if (neighbor == parentNode[node]) continue; // 直前の親は無視

            if (!visited[neighbor]) {
                parentNode[neighbor] = node;
                detectLoopDFS(neighbor);
            }
            else if (inStack[neighbor]) {
                // ループ検出！
                info.loopEdges.push_back({node, neighbor});
                info.loopCount++;
            }
        }

        inStack[node] = false;
    };

    // すべてのエッジを処理してUnion-Find
    for (const auto& node : nodes_) {
        for (int neighbor : node.neighbors) {
            if (node.id < neighbor) { // 各エッジを1回だけ処理
                unite(node.id, neighbor);
            }
        }
    }

    // DFSでループ検出
    for (size_t i = 0; i < nodes_.size(); i++) {
        if (!visited[i]) {
            detectLoopDFS(i);
        }
    }

    // 3. 連結成分を収集
    std::map<int, std::set<int>> components;
    for (size_t i = 0; i < nodes_.size(); i++) {
        int root = find(i);
        components[root].insert(i);
    }

    for (const auto& pair : components) {
        info.connectedComponents.push_back(pair.second);
    }
    info.componentCount = info.connectedComponents.size();
    info.hasDisconnection = (info.componentCount > 1);

    // 4. 孤立ノード（隣接なし）を検出
    for (const auto& node : nodes_) {
        if (node.neighbors.empty()) {
            info.isolatedNodes.push_back(node.id);
        }
    }

    // 結果出力
    std::cout << "  Connected components: " << info.componentCount << std::endl;

    if (info.hasDisconnection) {
        std::cout << "  ⚠ WARNING: Disconnected skeleton detected!" << std::endl;
        std::cout << "  Component sizes: ";
        for (size_t i = 0; i < info.connectedComponents.size(); i++) {
            std::cout << info.connectedComponents[i].size();
            if (i < info.connectedComponents.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "  ✓ Skeleton is fully connected" << std::endl;
    }

    std::cout << "  Loops detected: " << info.loopCount << std::endl;
    if (info.loopCount > 0) {
        std::cout << "  ⚠ WARNING: Loop edges found:" << std::endl;
        for (const auto& edge : info.loopEdges) {
            glm::vec3 p1 = nodes_[edge.first].position;
            glm::vec3 p2 = nodes_[edge.second].position;
            std::cout << "    Node " << edge.first << " (" << p1.x << ", " << p1.y << ", " << p1.z << ")"
                      << " <-> Node " << edge.second << " (" << p2.x << ", " << p2.y << ", " << p2.z << ")" << std::endl;
        }
    } else {
        std::cout << "  ✓ No loops (tree structure)" << std::endl;
    }

    if (!info.isolatedNodes.empty()) {
        std::cout << "  Isolated nodes: " << info.isolatedNodes.size() << std::endl;
    }

    std::cout << "========================\n" << std::endl;

    return info;
}

//=============================================================================
// デバッグ描画（ループと不連続を視覚化）
//=============================================================================
void VesselSegmentation::drawDebug(ShaderProgram& shader,
                                   const glm::mat4& model,
                                   const glm::mat4& view,
                                   const glm::mat4& projection,
                                   const DebugInfo& debugInfo) {
    if (nodes_.empty()) return;

    shader.use();
    shader.setUniform("model", model);
    shader.setUniform("view", view);
    shader.setUniform("projection", projection);

    // ループエッジを赤で描画
    if (!debugInfo.loopEdges.empty()) {
        std::vector<GLfloat> loopLineData;

        for (const auto& edge : debugInfo.loopEdges) {
            const auto& p1 = nodes_[edge.first].position;
            const auto& p2 = nodes_[edge.second].position;

            loopLineData.push_back(p1.x);
            loopLineData.push_back(p1.y);
            loopLineData.push_back(p1.z);
            loopLineData.push_back(p2.x);
            loopLineData.push_back(p2.y);
            loopLineData.push_back(p2.z);
        }

        GLuint loopVAO, loopVBO;
        glGenVertexArrays(1, &loopVAO);
        glGenBuffers(1, &loopVBO);

        glBindVertexArray(loopVAO);
        glBindBuffer(GL_ARRAY_BUFFER, loopVBO);
        glBufferData(GL_ARRAY_BUFFER, loopLineData.size() * sizeof(GLfloat), loopLineData.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);

        // 赤色でループを描画
        shader.setUniform("vertColor", glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
        glLineWidth(5.0f);
        glDrawArrays(GL_LINES, 0, loopLineData.size() / 3);

        glDeleteBuffers(1, &loopVBO);
        glDeleteVertexArrays(1, &loopVAO);
    }

    // 不連続部分を異なる色で描画（成分ごとに色分け）
    if (debugInfo.hasDisconnection) {
        static const std::vector<glm::vec3> componentColors = {
            glm::vec3(0.0f, 1.0f, 0.0f),  // 緑
            glm::vec3(0.0f, 0.0f, 1.0f),  // 青
            glm::vec3(1.0f, 0.0f, 1.0f),  // マゼンタ
            glm::vec3(0.0f, 1.0f, 1.0f),  // シアン
            glm::vec3(1.0f, 1.0f, 0.0f),  // 黄
        };

        for (size_t c = 0; c < debugInfo.connectedComponents.size(); c++) {
            std::vector<GLfloat> compLineData;
            const auto& component = debugInfo.connectedComponents[c];

            for (int nodeId : component) {
                const auto& node = nodes_[nodeId];
                for (int neighborId : node.neighbors) {
                    if (nodeId < neighborId && component.count(neighborId) > 0) {
                        const auto& p1 = node.position;
                        const auto& p2 = nodes_[neighborId].position;

                        compLineData.push_back(p1.x);
                        compLineData.push_back(p1.y);
                        compLineData.push_back(p1.z);
                        compLineData.push_back(p2.x);
                        compLineData.push_back(p2.y);
                        compLineData.push_back(p2.z);
                    }
                }
            }

            if (!compLineData.empty()) {
                GLuint compVAO, compVBO;
                glGenVertexArrays(1, &compVAO);
                glGenBuffers(1, &compVBO);

                glBindVertexArray(compVAO);
                glBindBuffer(GL_ARRAY_BUFFER, compVBO);
                glBufferData(GL_ARRAY_BUFFER, compLineData.size() * sizeof(GLfloat), compLineData.data(), GL_STATIC_DRAW);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
                glEnableVertexAttribArray(0);

                glm::vec3 color = componentColors[c % componentColors.size()];
                shader.setUniform("vertColor", glm::vec4(color, 1.0f));
                glLineWidth(3.0f);
                glDrawArrays(GL_LINES, 0, compLineData.size() / 3);

                glDeleteBuffers(1, &compVBO);
                glDeleteVertexArrays(1, &compVAO);
            }
        }
    }

    // 孤立ノードを大きな点で描画
    if (!debugInfo.isolatedNodes.empty()) {
        std::vector<GLfloat> pointData;

        for (int nodeId : debugInfo.isolatedNodes) {
            const auto& p = nodes_[nodeId].position;
            pointData.push_back(p.x);
            pointData.push_back(p.y);
            pointData.push_back(p.z);
        }

        GLuint pointVAO, pointVBO;
        glGenVertexArrays(1, &pointVAO);
        glGenBuffers(1, &pointVBO);

        glBindVertexArray(pointVAO);
        glBindBuffer(GL_ARRAY_BUFFER, pointVBO);
        glBufferData(GL_ARRAY_BUFFER, pointData.size() * sizeof(GLfloat), pointData.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);

        shader.setUniform("vertColor", glm::vec4(1.0f, 0.5f, 0.0f, 1.0f)); // オレンジ
        glPointSize(10.0f);
        glDrawArrays(GL_POINTS, 0, pointData.size() / 3);

        glDeleteBuffers(1, &pointVBO);
        glDeleteVertexArrays(1, &pointVAO);
    }

    glLineWidth(1.0f);
    glPointSize(1.0f);
    glBindVertexArray(0);
}

//=============================================================================
// ループ除去機能（最小全域木を構築してバックエッジを削除）
//=============================================================================
int VesselSegmentation::removeLoops() {
    if (nodes_.empty()) return 0;

    std::cout << "\n=== Removing Loops ===" << std::endl;

    // 現在のループを検出
    DebugInfo info = analyzeTopology();

    if (info.loopCount == 0) {
        std::cout << "  No loops to remove" << std::endl;
        return 0;
    }

    int removedCount = 0;

    // ループエッジを削除（距離値が小さい方のノードからエッジを削除）
    for (const auto& edge : info.loopEdges) {
        int node1 = edge.first;
        int node2 = edge.second;

        // 両ノードの距離値（半径）を比較
        float r1 = nodes_[node1].radius;
        float r2 = nodes_[node2].radius;

        // 半径が小さい方のノードからエッジを削除
        int removeFrom = (r1 < r2) ? node1 : node2;
        int removeTo = (r1 < r2) ? node2 : node1;

        // エッジを削除
        auto& neighbors = nodes_[removeFrom].neighbors;
        auto it = std::find(neighbors.begin(), neighbors.end(), removeTo);
        if (it != neighbors.end()) {
            neighbors.erase(it);

            // 双方向なので反対側も削除
            auto& neighbors2 = nodes_[removeTo].neighbors;
            auto it2 = std::find(neighbors2.begin(), neighbors2.end(), removeFrom);
            if (it2 != neighbors2.end()) {
                neighbors2.erase(it2);
            }

            removedCount++;
            std::cout << "  Removed edge: Node " << removeFrom << " <-> Node " << removeTo << std::endl;
        }
    }

    std::cout << "  Total removed: " << removedCount << " edges" << std::endl;

    // ループ除去後の確認
    DebugInfo infoAfter = analyzeTopology();

    // 再度セグメント化と階層構築
    if (removedCount > 0) {
        std::cout << "  Rebuilding segments..." << std::endl;
        segmentSkeleton();
        assignTrianglesToSegments();
        buildHierarchy();
        assignColors();
        updateSkeletonBuffers();
        std::cout << "  Rebuild complete. Final segments: " << segments_.size() << std::endl;
    }

    std::cout << "========================\n" << std::endl;

    return removedCount;
}

// //=============================================================================
// // 不連続成分の接続
// //=============================================================================
// int VesselSegmentation::connectDisconnectedComponents() {
//     if (nodes_.empty()) return 0;

//     // Union-Findで連結成分を検出
//     std::vector<int> parent(nodes_.size());
//     for (size_t i = 0; i < nodes_.size(); i++) {
//         parent[i] = i;
//     }

//     std::function<int(int)> find = [&](int x) -> int {
//         if (parent[x] != x) parent[x] = find(parent[x]);
//         return parent[x];
//     };

//     auto unite = [&](int x, int y) {
//         int px = find(x), py = find(y);
//         if (px != py) parent[px] = py;
//     };

//     // 既存のエッジで接続
//     for (const auto& node : nodes_) {
//         for (int neighbor : node.neighbors) {
//             unite(node.id, neighbor);
//         }
//     }

//     // 連結成分ごとにノードを分類
//     std::map<int, std::vector<int>> components;
//     for (size_t i = 0; i < nodes_.size(); i++) {
//         components[find(i)].push_back(i);
//     }

//     if (components.size() <= 1) {
//         return 0;  // 既に連結
//     }

//     std::cout << "  Found " << components.size() << " disconnected components" << std::endl;

//     // 最大の連結成分を「メイン」とする
//     int mainRoot = -1;
//     size_t maxSize = 0;
//     for (const auto& pair : components) {
//         if (pair.second.size() > maxSize) {
//             maxSize = pair.second.size();
//             mainRoot = pair.first;
//         }
//     }

//     int connectionsAdded = 0;

//     // 各小さい成分をメイン成分に接続
//     for (const auto& pair : components) {
//         if (pair.first == mainRoot) continue;

//         const auto& smallComponent = pair.second;
//         const auto& mainComponent = components[mainRoot];

//         // 小さい成分の各ノードから、メイン成分の最も近いノードを探す
//         int bestSmallNode = -1;
//         int bestMainNode = -1;
//         float minDist = std::numeric_limits<float>::max();

//         for (int smallNodeId : smallComponent) {
//             for (int mainNodeId : mainComponent) {
//                 float dist = glm::length(nodes_[smallNodeId].position - nodes_[mainNodeId].position);
//                 if (dist < minDist) {
//                     minDist = dist;
//                     bestSmallNode = smallNodeId;
//                     bestMainNode = mainNodeId;
//                 }
//             }
//         }

//         // 接続を追加
//         if (bestSmallNode >= 0 && bestMainNode >= 0) {
//             // 双方向接続
//             nodes_[bestSmallNode].neighbors.push_back(bestMainNode);
//             nodes_[bestMainNode].neighbors.push_back(bestSmallNode);

//             // Union-Findも更新
//             unite(bestSmallNode, bestMainNode);

//             connectionsAdded++;

//             std::cout << "    Connected node " << bestSmallNode << " to " << bestMainNode
//                       << " (distance: " << minDist << ")" << std::endl;
//         }
//     }

//     return connectionsAdded;
// }

//=============================================================================
// 親子関係の検証（木構造の一意性チェック）
//=============================================================================
bool VesselSegmentation::verifyTreeStructure() {
    std::cout << "\n=== Tree Structure Verification ===" << std::endl;

    bool isValid = true;

    // 1. 各セグメントが1つの親しか持たないことを確認
    std::cout << "  Checking parent uniqueness..." << std::endl;
    int rootCount = 0;
    for (const auto& seg : segments_) {
        if (seg.parentId == -1) {
            rootCount++;
        }
    }

    if (rootCount != 1) {
        std::cout << "  ⚠ WARNING: Found " << rootCount << " root segments (should be 1)" << std::endl;
        isValid = false;
    } else {
        std::cout << "  ✓ Single root segment" << std::endl;
    }

    // 2. 子が複数の親を持っていないことを確認
    std::cout << "  Checking for multiple parents..." << std::endl;
    std::map<int, std::vector<int>> childToParents;

    for (const auto& seg : segments_) {
        for (int childId : seg.childIds) {
            childToParents[childId].push_back(seg.id);
        }
    }

    int multipleParentCount = 0;
    for (const auto& pair : childToParents) {
        if (pair.second.size() > 1) {
            multipleParentCount++;
            std::cout << "  ⚠ Segment " << pair.first << " has " << pair.second.size() << " parents: ";
            for (int p : pair.second) {
                std::cout << p << " ";
            }
            std::cout << std::endl;
            isValid = false;
        }
    }

    if (multipleParentCount == 0) {
        std::cout << "  ✓ No segment has multiple parents" << std::endl;
    }

    // 3. parentIdとchildIdsの整合性を確認
    std::cout << "  Checking parent-child consistency..." << std::endl;
    int inconsistentCount = 0;

    for (const auto& seg : segments_) {
        // 親が設定されている場合、その親のchildIdsに自分がいるか確認
        if (seg.parentId >= 0 && seg.parentId < static_cast<int>(segments_.size())) {
            const auto& parent = segments_[seg.parentId];
            bool foundInParent = false;
            for (int childId : parent.childIds) {
                if (childId == seg.id) {
                    foundInParent = true;
                    break;
                }
            }
            if (!foundInParent) {
                std::cout << "  ⚠ Segment " << seg.id << " has parent " << seg.parentId
                          << " but not in parent's childIds" << std::endl;
                inconsistentCount++;
                isValid = false;
            }
        }

        // 子が設定されている場合、その子のparentIdが自分か確認
        for (int childId : seg.childIds) {
            if (childId >= 0 && childId < static_cast<int>(segments_.size())) {
                if (segments_[childId].parentId != seg.id) {
                    std::cout << "  ⚠ Segment " << seg.id << " has child " << childId
                              << " but child's parent is " << segments_[childId].parentId << std::endl;
                    inconsistentCount++;
                    isValid = false;
                }
            }
        }
    }

    if (inconsistentCount == 0) {
        std::cout << "  ✓ Parent-child relationships are consistent" << std::endl;
    }

    // 4. ノードレベルでの確認（各ノードが1つのセグメントにのみ属する）
    std::cout << "  Checking node uniqueness..." << std::endl;
    std::map<int, std::vector<int>> nodeToSegments;

    for (const auto& seg : segments_) {
        for (int nodeId : seg.nodeIds) {
            nodeToSegments[nodeId].push_back(seg.id);
        }
    }

    int duplicateNodeCount = 0;
    for (const auto& pair : nodeToSegments) {
        if (pair.second.size() > 1) {
            duplicateNodeCount++;
            if (duplicateNodeCount <= 5) {  // 最初の5つだけ表示
                std::cout << "  ⚠ Node " << pair.first << " belongs to " << pair.second.size() << " segments: ";
                for (int s : pair.second) {
                    std::cout << s << " ";
                }
                std::cout << std::endl;
            }
            isValid = false;
        }
    }

    if (duplicateNodeCount > 5) {
        std::cout << "  ... and " << (duplicateNodeCount - 5) << " more duplicate nodes" << std::endl;
    }

    if (duplicateNodeCount == 0) {
        std::cout << "  ✓ Each node belongs to exactly one segment" << std::endl;
    } else {
        std::cout << "  ⚠ Found " << duplicateNodeCount << " nodes belonging to multiple segments" << std::endl;
    }

    // 5. サマリー
    std::cout << "\n  === Summary ===" << std::endl;
    std::cout << "  Total segments: " << segments_.size() << std::endl;
    std::cout << "  Total nodes: " << nodes_.size() << std::endl;

    int totalNodesInSegments = 0;
    for (const auto& seg : segments_) {
        totalNodesInSegments += seg.nodeIds.size();
    }
    std::cout << "  Nodes in segments: " << totalNodesInSegments << std::endl;

    if (isValid) {
        std::cout << "\n  ✓✓✓ Tree structure is VALID ✓✓✓" << std::endl;
    } else {
        std::cout << "\n  ⚠⚠⚠ Tree structure has ISSUES ⚠⚠⚠" << std::endl;
    }

    std::cout << "===================================\n" << std::endl;

    return isValid;
}


void VesselSegmentation::draw(ShaderProgram& shader,
                              const std::vector<GLfloat>& vertices,
                              const std::vector<GLuint>& indices,
                              const glm::mat4& model,
                              const glm::mat4& view,
                              const glm::mat4& projection,
                              const glm::vec3& cameraPos) {
    shader.use();
    shader.setUniform("model", model);
    shader.setUniform("view", view);
    shader.setUniform("projection", projection);
    shader.setUniform("lightPos", cameraPos);

    glUniform1i(glGetUniformLocation(shader.getProgram(), "useVertexColor"), 0);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    GLuint tempVAO, tempVBO, tempEBO, tempNormalVBO;
    glGenVertexArrays(1, &tempVAO);
    glGenBuffers(1, &tempVBO);
    glGenBuffers(1, &tempNormalVBO);
    glGenBuffers(1, &tempEBO);

    glBindVertexArray(tempVAO);

    // 頂点位置
    glBindBuffer(GL_ARRAY_BUFFER, tempVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GLfloat), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    // 法線を計算
    std::vector<GLfloat> normals(vertices.size(), 0.0f);
    size_t numVerts = vertices.size() / 3;

    for (size_t i = 0; i < indices.size(); i += 3) {
        int i0 = indices[i], i1 = indices[i+1], i2 = indices[i+2];

        glm::vec3 v0(vertices[i0*3], vertices[i0*3+1], vertices[i0*3+2]);
        glm::vec3 v1(vertices[i1*3], vertices[i1*3+1], vertices[i1*3+2]);
        glm::vec3 v2(vertices[i2*3], vertices[i2*3+1], vertices[i2*3+2]);

        glm::vec3 normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));

        for (int idx : {i0, i1, i2}) {
            normals[idx*3+0] += normal.x;
            normals[idx*3+1] += normal.y;
            normals[idx*3+2] += normal.z;
        }
    }

    for (size_t i = 0; i < numVerts; i++) {
        glm::vec3 n(normals[i*3], normals[i*3+1], normals[i*3+2]);
        float len = glm::length(n);
        if (len > 0.001f) {
            normals[i*3+0] /= len;
            normals[i*3+1] /= len;
            normals[i*3+2] /= len;
        }
    }

    glBindBuffer(GL_ARRAY_BUFFER, tempNormalVBO);
    glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(GLfloat), normals.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, tempEBO);

    // ★ 選択モード用の色定義
    const glm::vec3 selectedColor = glm::vec3(1.0f, 0.85f, 0.2f);    // ゴールド
    const glm::vec3 unselectedColor = glm::vec3(1.0f, 0.6f, 0.8f);   // ピンク

    // ★ 選択状態があるかどうかをチェック
    bool hasSelection = !selection_.selectedSegments.empty();

    for (const auto& segment : segments_) {
        if (segment.triangleIndices.empty()) continue;

        glm::vec3 color;

        if (hasSelection) {
            // ★ 選択モード：2色のみ
            if (selection_.selectedSegments.count(segment.id) > 0) {
                color = selectedColor;   // 選択された部分はゴールド
            } else {
                color = unselectedColor; // 選択されていない部分はピンク
            }
        } else {
            // ★ 通常モード：セグメントごとの色
            color = glm::clamp(segment.color * 1.3f, 0.0f, 1.0f);
        }

        shader.setUniform("objectColor", color);

        std::vector<GLuint> segIndices;
        for (int triIdx : segment.triangleIndices) {
            segIndices.push_back(indices[triIdx * 3]);
            segIndices.push_back(indices[triIdx * 3 + 1]);
            segIndices.push_back(indices[triIdx * 3 + 2]);
        }

        glBufferData(GL_ELEMENT_ARRAY_BUFFER, segIndices.size() * sizeof(GLuint), segIndices.data(), GL_DYNAMIC_DRAW);
        glDrawElements(GL_TRIANGLES, segIndices.size(), GL_UNSIGNED_INT, nullptr);
    }

    glDeleteBuffers(1, &tempEBO);
    glDeleteBuffers(1, &tempNormalVBO);
    glDeleteBuffers(1, &tempVBO);
    glDeleteVertexArrays(1, &tempVAO);

    glDisable(GL_BLEND);
}

void VesselSegmentation::drawSkeleton(ShaderProgram& shader,
                                      const glm::mat4& model,
                                      const glm::mat4& view,
                                      const glm::mat4& projection) {
    if (!buffersInitialized_ || nodes_.empty()) return;

    shader.use();
    shader.setUniform("model", model);
    shader.setUniform("view", view);
    shader.setUniform("projection", projection);

    // useVertexColorをオフに
    glUniform1i(glGetUniformLocation(shader.getProgram(), "useVertexColor"), 0);

    // objectColor (vec3) を使用
    shader.setUniform("objectColor", glm::vec3(1.0f, 1.0f, 0.0f));

    glBindVertexArray(skeletonVAO_);

    int lineCount = 0;
    for (const auto& node : nodes_) {
        for (int neighborId : node.neighbors) {
            if (neighborId > node.id) lineCount++;
        }
    }

    glLineWidth(2.0f);
    glDrawArrays(GL_LINES, 0, lineCount * 2);
    glLineWidth(1.0f);

    glBindVertexArray(0);
}


//=============================================================================
// portal SoftBodyにスケルトンをバインド
//=============================================================================
void VesselSegmentation::bindToPortalSoftBody(
    const std::vector<float>& positions,
    const std::vector<int>& tetIds,
    size_t numTets)
{
    nodeBindings.clear();
    nodeBindings.resize(nodes_.size());

    std::cout << "Binding skeleton to portal SoftBody..." << std::endl;

    for (size_t i = 0; i < nodes_.size(); i++) {
        const glm::vec3& nodePos = nodes_[i].position;
        nodeBindings[i].initialPos = nodePos;

        int bestTet = -1;
        float minDist = std::numeric_limits<float>::max();
        glm::vec4 bestBary(0.25f);

        // 最も近い四面体を探す
        for (size_t t = 0; t < numTets; t++) {
            glm::vec3 v[4];
            for (int j = 0; j < 4; j++) {
                int vid = tetIds[t * 4 + j];
                v[j] = glm::vec3(
                    positions[vid * 3 + 0],
                    positions[vid * 3 + 1],
                    positions[vid * 3 + 2]
                    );
            }

            // 四面体の中心
            glm::vec3 center = (v[0] + v[1] + v[2] + v[3]) * 0.25f;
            float dist = glm::length(nodePos - center);

            if (dist < minDist) {
                minDist = dist;
                bestTet = (int)t;

                // バリセントリック座標を計算
                glm::vec3 d0 = v[1] - v[0];
                glm::vec3 d1 = v[2] - v[0];
                glm::vec3 d2 = v[3] - v[0];
                glm::vec3 d3 = nodePos - v[0];

                float det = glm::dot(d0, glm::cross(d1, d2));
                if (std::abs(det) > 1e-10f) {
                    float b1 = glm::dot(d3, glm::cross(d1, d2)) / det;
                    float b2 = glm::dot(d0, glm::cross(d3, d2)) / det;
                    float b3 = glm::dot(d0, glm::cross(d1, d3)) / det;
                    float b0 = 1.0f - b1 - b2 - b3;
                    bestBary = glm::vec4(b0, b1, b2, b3);
                }
            }
        }

        nodeBindings[i].tetIdx = bestTet;
        nodeBindings[i].baryCoords = bestBary;
    }

    isBoundToSoftBody = true;

    int boundCount = 0;
    for (const auto& b : nodeBindings) {
        if (b.tetIdx >= 0) boundCount++;
    }
    std::cout << "  Bound " << boundCount << "/" << nodes_.size() << " nodes" << std::endl;
}

//=============================================================================
// portal SoftBodyの変形に追従してノード位置を更新
//=============================================================================
void VesselSegmentation::updateNodesFromPortal(
    const std::vector<float>& positions,
    const std::vector<int>& tetIds)
{
    if (!isBoundToSoftBody) return;

    for (size_t i = 0; i < nodes_.size(); i++) {
        const auto& binding = nodeBindings[i];

        if (binding.tetIdx < 0) continue;

        // 四面体の頂点位置を取得
        glm::vec3 v[4];
        for (int j = 0; j < 4; j++) {
            int vid = tetIds[binding.tetIdx * 4 + j];
            v[j] = glm::vec3(
                positions[vid * 3 + 0],
                positions[vid * 3 + 1],
                positions[vid * 3 + 2]
                );
        }

        // バリセントリック座標で新しい位置を計算
        nodes_[i].position =
            v[0] * binding.baryCoords[0] +
            v[1] * binding.baryCoords[1] +
            v[2] * binding.baryCoords[2] +
            v[3] * binding.baryCoords[3];
    }

    // スケルトン描画バッファを更新
    updateSkeletonBuffers();
}

//=============================================================================
// バインディング解除
//=============================================================================
void VesselSegmentation::unbindFromSoftBody() {
    // 初期位置に戻す
    for (size_t i = 0; i < nodes_.size(); i++) {
        if (i < nodeBindings.size()) {
            nodes_[i].position = nodeBindings[i].initialPos;
        }
    }

    nodeBindings.clear();
    isBoundToSoftBody = false;
    updateSkeletonBuffers();
}



//=============================================================================
// スケルトンスムージング
//=============================================================================
void VesselSegmentation::smoothSkeletonGraph(int iterations, float factor) {
    if (nodes_.empty()) return;

    std::cout << "  Smoothing skeleton graph (" << iterations << " iterations)..." << std::endl;

    for (int iter = 0; iter < iterations; iter++) {
        std::vector<glm::vec3> newPositions(nodes_.size());

        for (size_t i = 0; i < nodes_.size(); i++) {
            const auto& node = nodes_[i];

            // 隣接が2つ（直線部分）のノードのみスムージング
            // 分岐点（3以上）と端点（1以下）は動かさない
            if (node.neighbors.size() == 2) {
                int n1 = node.neighbors[0];
                int n2 = node.neighbors[1];

                glm::vec3 midpoint = (nodes_[n1].position + nodes_[n2].position) * 0.5f;
                newPositions[i] = glm::mix(node.position, midpoint, factor);
            } else {
                newPositions[i] = node.position;
            }
        }

        // 位置を更新
        for (size_t i = 0; i < nodes_.size(); i++) {
            nodes_[i].position = newPositions[i];
        }
    }
}

//=============================================================================
// バックアップ・リストア
//=============================================================================
void VesselSegmentation::backupCurrentSkeleton(SkeletonBackup& backup) {
    backup.nodes = nodes_;
    backup.segments = segments_;
    backup.triangleToSegment = triangleToSegment_;
    backup.hasBackup = true;
}

void VesselSegmentation::restoreSkeletonFromBackup(const SkeletonBackup& backup) {
    if (!backup.hasBackup) return;

    nodes_ = backup.nodes;
    segments_ = backup.segments;
    triangleToSegment_ = backup.triangleToSegment;

    updateSkeletonBuffers();
}

//=============================================================================
// スケルトン切り替え
//=============================================================================
void VesselSegmentation::toggleExtendedSkeleton() {
    if (!originalSkeleton_.hasBackup || !extendedSkeleton_.hasBackup) {
        std::cout << "Extended skeleton not available. Run autoExtendShortTerminalBranches() first." << std::endl;
        return;
    }

    useExtendedSkeleton_ = !useExtendedSkeleton_;

    if (useExtendedSkeleton_) {
        restoreSkeletonFromBackup(extendedSkeleton_);
        std::cout << "Switched to EXTENDED skeleton" << std::endl;
    } else {
        restoreSkeletonFromBackup(originalSkeleton_);
        std::cout << "Switched to ORIGINAL skeleton" << std::endl;
    }
}


//=============================================================================
// 最終枝の検出（端点ノードを持つセグメント）
//=============================================================================
std::vector<int> VesselSegmentation::findTerminalSegments() {
    std::vector<int> terminals;

    for (const auto& seg : segments_) {
        if (seg.nodeIds.empty()) continue;

        // セグメント内に端点ノード（neighbors==1）があれば最終枝
        bool hasEndpoint = false;
        for (int nodeId : seg.nodeIds) {
            if (nodeId >= 0 && nodeId < static_cast<int>(nodes_.size())) {
                if (nodes_[nodeId].neighbors.size() == 1) {
                    hasEndpoint = true;
                    break;
                }
            }
        }

        if (hasEndpoint) {
            terminals.push_back(seg.id);
        }
    }

    std::cout << "  Found " << terminals.size() << " terminal segments" << std::endl;
    return terminals;
}

//=============================================================================
// 最終枝の判定
//=============================================================================
bool VesselSegmentation::isTerminalSegment(int segmentId) const {
    if (segmentId < 0 || segmentId >= static_cast<int>(segments_.size())) {
        return false;
    }

    const auto& seg = segments_[segmentId];

    for (int nodeId : seg.nodeIds) {
        if (nodeId >= 0 && nodeId < static_cast<int>(nodes_.size())) {
            if (nodes_[nodeId].neighbors.size() == 1) {
                return true;
            }
        }
    }

    return false;
}

//=============================================================================
//=============================================================================
// ノード間の平均距離
//=============================================================================
float VesselSegmentation::getAverageNodeSpacing(int segmentId) {
    if (segmentId < 0 || segmentId >= static_cast<int>(segments_.size())) {
        return glm::length(voxelSize_);
    }

    const auto& seg = segments_[segmentId];
    if (seg.nodeIds.size() < 2) {
        return glm::length(voxelSize_);
    }

    float totalDist = 0.0f;
    int count = 0;

    for (int nodeId : seg.nodeIds) {
        if (nodeId < 0 || nodeId >= static_cast<int>(nodes_.size())) continue;

        for (int neighborId : nodes_[nodeId].neighbors) {
            if (neighborId >= 0 && neighborId < static_cast<int>(nodes_.size())) {
                if (nodes_[neighborId].segmentId == segmentId && nodeId < neighborId) {
                    totalDist += glm::length(nodes_[nodeId].position - nodes_[neighborId].position);
                    count++;
                }
            }
        }
    }

    return (count > 0) ? totalDist / count : glm::length(voxelSize_);
}

//=============================================================================
// セグメントの長さ計算
//=============================================================================
float VesselSegmentation::calculateSegmentLength(int segmentId) {
    if (segmentId < 0 || segmentId >= static_cast<int>(segments_.size())) {
        return 0.0f;
    }

    const auto& seg = segments_[segmentId];
    if (seg.nodeIds.size() < 2) return 0.0f;

    int endpointNodeId = findEndpointNode(segmentId);
    if (endpointNodeId < 0) return 0.0f;

    float totalLength = 0.0f;
    std::set<int> visited;
    int currentNode = endpointNodeId;
    visited.insert(currentNode);

    while (true) {
        if (nodes_[currentNode].neighbors.size() >= 3) {
            break;
        }

        int nextNode = -1;

        for (int neighborId : nodes_[currentNode].neighbors) {
            if (visited.find(neighborId) == visited.end() &&
                nodes_[neighborId].segmentId == segmentId) {
                nextNode = neighborId;
                break;
            }
        }

        if (nextNode < 0) break;

        totalLength += glm::length(nodes_[nextNode].position - nodes_[currentNode].position);
        visited.insert(nextNode);
        currentNode = nextNode;
    }

    return totalLength;
}


//=============================================================================
// 手動延長前にバックアップ
//=============================================================================
void VesselSegmentation::backupBeforeManualExtension() {
    if (!originalSkeleton_.hasBackup) {
        backupCurrentSkeleton(originalSkeleton_);
    }
}

//=============================================================================
// 手動延長後に保存
//=============================================================================
void VesselSegmentation::saveManualExtension() {
    backupCurrentSkeleton(extendedSkeleton_);
    manualExtended_ = true;
    useExtendedSkeleton_ = true;
    std::cout << "Manual extended skeleton saved" << std::endl;
}

//=============================================================================
// 自動延長に戻す
//=============================================================================
void VesselSegmentation::revertToAutoExtended() {
    if (!autoExtendedSkeleton_.hasBackup) {
        std::cout << "No auto-extended skeleton available" << std::endl;
        return;
    }

    extendedSkeleton_ = autoExtendedSkeleton_;

    if (useExtendedSkeleton_) {
        restoreSkeletonFromBackup(autoExtendedSkeleton_);
    }

    manualExtended_ = false;
    std::cout << "Reverted to auto-extended skeleton" << std::endl;
}


//=============================================================================
// ノードの進行方向を取得
//=============================================================================
glm::vec3 VesselSegmentation::getNodeDirection(int nodeId) const {
    if (nodeId < 0 || nodeId >= static_cast<int>(nodes_.size())) {
        return glm::vec3(0.0f, 1.0f, 0.0f);
    }

    const auto& node = nodes_[nodeId];

    if (node.neighbors.empty()) {
        return glm::vec3(0.0f, 1.0f, 0.0f);
    }

    // 隣接ノードの逆方向の平均 = このノードの進行方向
    glm::vec3 avgDir(0.0f);
    for (int neighborId : node.neighbors) {
        if (neighborId >= 0 && neighborId < static_cast<int>(nodes_.size())) {
            avgDir += node.position - nodes_[neighborId].position;
        }
    }

    if (glm::length(avgDir) < 0.001f) {
        return glm::vec3(0.0f, 1.0f, 0.0f);
    }

    return glm::normalize(avgDir);
}


//=============================================================================
// 点から線分への最短距離を計算
//=============================================================================
float VesselSegmentation::pointToSegmentDistance(
    const glm::vec3& point,
    const glm::vec3& segStart,
    const glm::vec3& segEnd,
    glm::vec3& closestPoint) const
{
    glm::vec3 seg = segEnd - segStart;
    float segLenSq = glm::dot(seg, seg);

    if (segLenSq < 1e-10f) {
        closestPoint = segStart;
        return glm::length(point - segStart);
    }

    float t = glm::clamp(glm::dot(point - segStart, seg) / segLenSq, 0.0f, 1.0f);
    closestPoint = segStart + t * seg;
    return glm::length(point - closestPoint);
}

//=============================================================================
// 端点の伸長方向を計算（親の方向を考慮）
//=============================================================================
glm::vec3 VesselSegmentation::getExtendDirection(int endpointId) const {
    if (endpointId < 0 || endpointId >= static_cast<int>(nodes_.size())) {
        return glm::vec3(0.0f);
    }

    const auto& endpoint = nodes_[endpointId];

    // 端点でない場合は通常の方向
    if (endpoint.neighbors.size() != 1) {
        return getNodeDirection(endpointId);
    }

    // 端点から親方向にノードを辿る
    const int lookbackDepth = 5;

    std::vector<glm::vec3> positions;
    positions.push_back(endpoint.position);

    int current = endpointId;
    int prev = -1;

    for (int i = 0; i < lookbackDepth; i++) {
        int next = -1;
        for (int neighbor : nodes_[current].neighbors) {
            if (neighbor != prev) {
                next = neighbor;
                break;
            }
        }
        if (next < 0) break;

        positions.push_back(nodes_[next].position);
        prev = current;
        current = next;
    }

    if (positions.size() < 2) {
        return getNodeDirection(endpointId);
    }

    // 末端の方向（端点 → 隣接ノード、逆向きにして外向きに）
    glm::vec3 localDir = glm::normalize(positions[0] - positions[1]);

    if (positions.size() < 3) {
        return localDir;
    }

    // 親の方向（上流の流れ）
    glm::vec3 parentDir = glm::normalize(
        positions[positions.size() - 2] - positions[positions.size() - 1]
        );

    // 加重平均（末端方向:親方向 = 1:1）
    glm::vec3 blendedDir = glm::normalize(localDir + parentDir);

    return blendedDir;
}

//=============================================================================
// 不連続成分の接続（点-線分距離版 + 方向考慮）
//=============================================================================
int VesselSegmentation::connectDisconnectedComponents() {
    if (nodes_.empty()) return 0;

    artificialConnections_.clear();

    int connectionsAdded = 0;
    float maxConnectionDistance = glm::length(voxelSize_) * 15.0f;
    float veryCloseThreshold = glm::length(voxelSize_) * 5.0f;
    float directionThreshold = 0.2f;

    std::cout << "  Max connection distance: " << maxConnectionDistance << std::endl;
    std::cout << "  Very close threshold: " << veryCloseThreshold << std::endl;

    bool changed = true;

    while (changed) {
        changed = false;

        // Union-Find
        std::vector<int> parent(nodes_.size());
        for (size_t i = 0; i < nodes_.size(); i++) {
            parent[i] = static_cast<int>(i);
        }

        std::function<int(int)> find = [&](int x) -> int {
            if (parent[x] != x) parent[x] = find(parent[x]);
            return parent[x];
        };

        auto unite = [&](int x, int y) {
            int px = find(x), py = find(y);
            if (px != py) parent[px] = py;
        };

        for (const auto& node : nodes_) {
            for (int neighbor : node.neighbors) {
                unite(node.id, neighbor);
            }
        }

        std::map<int, std::vector<int>> components;
        for (size_t i = 0; i < nodes_.size(); i++) {
            components[find(static_cast<int>(i))].push_back(static_cast<int>(i));
        }

        if (components.size() <= 1) {
            break;
        }

        if (connectionsAdded == 0) {
            std::cout << "\n  Found " << components.size() << " disconnected components" << std::endl;
            for (const auto& pair : components) {
                std::cout << "    Component " << pair.first << ": " << pair.second.size() << " nodes" << std::endl;
            }
        }

        // 各成分の端点とエッジを抽出
        std::map<int, std::vector<int>> componentEndpoints;
        std::map<int, std::vector<std::pair<int, int>>> componentEdges;

        for (const auto& pair : components) {
            int root = pair.first;
            for (int nodeId : pair.second) {
                if (nodes_[nodeId].neighbors.size() == 1) {
                    componentEndpoints[root].push_back(nodeId);
                }
                // エッジを収集（重複を避けるため nodeId < neighbor のみ）
                for (int neighbor : nodes_[nodeId].neighbors) {
                    if (nodeId < neighbor) {
                        componentEdges[root].push_back({nodeId, neighbor});
                    }
                }
            }
        }

        // 最良ペアを探す
        int bestNode1 = -1, bestNode2 = -1;
        float bestDist = std::numeric_limits<float>::max();
        float bestScore = std::numeric_limits<float>::max();
        bool bestIsVeryClose = false;

        std::vector<int> compRoots;
        for (const auto& pair : components) {
            compRoots.push_back(pair.first);
        }

        for (size_t ci = 0; ci < compRoots.size(); ci++) {
            for (size_t cj = ci + 1; cj < compRoots.size(); cj++) {
                int root1 = compRoots[ci];
                int root2 = compRoots[cj];

                const auto& endpoints1 = componentEndpoints[root1];
                const auto& edges2 = componentEdges[root2];
                const auto& endpoints2 = componentEndpoints[root2];
                const auto& edges1 = componentEdges[root1];

                // ★ 成分1の端点 → 成分2のエッジ（線分）
                for (int endpoint : endpoints1) {
                    glm::vec3 endpointPos = nodes_[endpoint].position;
                    glm::vec3 endpointDir = getExtendDirection(endpoint);

                    for (const auto& edge : edges2) {
                        glm::vec3 closestPoint;
                        float dist = pointToSegmentDistance(
                            endpointPos,
                            nodes_[edge.first].position,
                            nodes_[edge.second].position,
                            closestPoint
                            );

                        if (dist > maxConnectionDistance) continue;

                        // 接続先ノードを決定
                        float d1 = glm::length(closestPoint - nodes_[edge.first].position);
                        float d2 = glm::length(closestPoint - nodes_[edge.second].position);
                        int targetNode = (d1 < d2) ? edge.first : edge.second;

                        // 非常に近い場合は方向チェックなし
                        if (dist < veryCloseThreshold) {
                            if (dist < bestScore || !bestIsVeryClose) {
                                bestScore = dist;
                                bestDist = dist;
                                bestNode1 = endpoint;
                                bestNode2 = targetNode;
                                bestIsVeryClose = true;
                            }
                        }
                        // そうでなければ方向もチェック
                        else if (!bestIsVeryClose) {
                            glm::vec3 connectionDir = glm::normalize(closestPoint - endpointPos);
                            float dot = glm::dot(endpointDir, connectionDir);

                            if (dot > directionThreshold) {
                                float score = dist / (dot + 0.1f);
                                if (score < bestScore) {
                                    bestScore = score;
                                    bestDist = dist;
                                    bestNode1 = endpoint;
                                    bestNode2 = targetNode;
                                }
                            }
                        }
                    }
                }

                // ★ 成分2の端点 → 成分1のエッジ（線分）
                for (int endpoint : endpoints2) {
                    glm::vec3 endpointPos = nodes_[endpoint].position;
                    glm::vec3 endpointDir = getExtendDirection(endpoint);

                    for (const auto& edge : edges1) {
                        glm::vec3 closestPoint;
                        float dist = pointToSegmentDistance(
                            endpointPos,
                            nodes_[edge.first].position,
                            nodes_[edge.second].position,
                            closestPoint
                            );

                        if (dist > maxConnectionDistance) continue;

                        // 接続先ノードを決定
                        float d1 = glm::length(closestPoint - nodes_[edge.first].position);
                        float d2 = glm::length(closestPoint - nodes_[edge.second].position);
                        int targetNode = (d1 < d2) ? edge.first : edge.second;

                        // 非常に近い場合は方向チェックなし
                        if (dist < veryCloseThreshold) {
                            if (dist < bestScore || !bestIsVeryClose) {
                                bestScore = dist;
                                bestDist = dist;
                                bestNode1 = targetNode;
                                bestNode2 = endpoint;
                                bestIsVeryClose = true;
                            }
                        }
                        // そうでなければ方向もチェック
                        else if (!bestIsVeryClose) {
                            glm::vec3 connectionDir = glm::normalize(closestPoint - endpointPos);
                            float dot = glm::dot(endpointDir, connectionDir);

                            if (dot > directionThreshold) {
                                float score = dist / (dot + 0.1f);
                                if (score < bestScore) {
                                    bestScore = score;
                                    bestDist = dist;
                                    bestNode1 = targetNode;
                                    bestNode2 = endpoint;
                                }
                            }
                        }
                    }
                }
            }
        }

        // 接続
        if (bestNode1 >= 0 && bestNode2 >= 0) {
            nodes_[bestNode1].neighbors.push_back(bestNode2);
            nodes_[bestNode2].neighbors.push_back(bestNode1);

            int minNode = std::min(bestNode1, bestNode2);
            int maxNode = std::max(bestNode1, bestNode2);
            artificialConnections_.insert({minNode, maxNode});

            connectionsAdded++;
            changed = true;

            std::cout << "    Connected: " << bestNode1 << " <-> " << bestNode2
                      << " (dist: " << bestDist;
            if (bestIsVeryClose) {
                std::cout << ", VERY_CLOSE";
            }
            std::cout << ")" << std::endl;
        } else {
            std::cout << "    No valid connection within distance limit" << std::endl;
            break;
        }
    }

    std::cout << "  Total artificial connections: " << artificialConnections_.size() << std::endl;

    return connectionsAdded;
}

//=============================================================================
// 肝臓メッシュを設定
//=============================================================================
void VesselSegmentation::setLiverMesh(const std::vector<float>& vertices,
                                      const std::vector<int>& indices) {
    liverVertices_.assign(vertices.begin(), vertices.end());
    liverIndices_.assign(indices.begin(), indices.end());
    hasLiverMesh_ = true;

    std::cout << "Liver mesh set: " << (vertices.size() / 3) << " vertices, "
              << (indices.size() / 3) << " triangles" << std::endl;
}

//=============================================================================
// 延長方向の計算（親セグメントの方向を考慮）
//=============================================================================
glm::vec3 VesselSegmentation::calculateExtensionDirection(int endpointNodeId, float repulsionWeight) {
    if (endpointNodeId < 0 || endpointNodeId >= static_cast<int>(nodes_.size())) {
        return glm::vec3(0.0f, 1.0f, 0.0f);
    }

    const auto& endpoint = nodes_[endpointNodeId];
    int segmentId = endpoint.segmentId;

    // このセグメント内のノード位置を収集（端点から分岐点まで）
    std::vector<glm::vec3> positions;
    positions.push_back(endpoint.position);

    int currentNode = endpointNodeId;
    int prevNode = -1;

    while (true) {
        int nextNode = -1;
        for (int neighborId : nodes_[currentNode].neighbors) {
            if (neighborId != prevNode) {
                nextNode = neighborId;
                break;
            }
        }
        if (nextNode < 0) break;

        positions.push_back(nodes_[nextNode].position);
        prevNode = currentNode;
        currentNode = nextNode;

        if (nodes_[nextNode].segmentId != segmentId ||
            nodes_[nextNode].neighbors.size() >= 3) {
            break;
        }
    }

    // 基本方向：分岐点から端点への方向
    glm::vec3 baseDirection(0.0f);

    if (positions.size() >= 2) {
        baseDirection = glm::normalize(positions[0] - positions.back());
    } else {
        baseDirection = glm::vec3(0.0f, 1.0f, 0.0f);
    }

    // 親セグメントの方向を取得
    glm::vec3 parentDirection(0.0f);

    if (segmentId >= 0 && segmentId < static_cast<int>(segments_.size())) {
        int parentSegId = segments_[segmentId].parentId;

        if (parentSegId >= 0 && parentSegId < static_cast<int>(segments_.size())) {
            int junctionNodeId = -1;

            for (int neighborId : nodes_[currentNode].neighbors) {
                if (nodes_[neighborId].segmentId == parentSegId) {
                    junctionNodeId = neighborId;
                    break;
                }
            }

            if (junctionNodeId >= 0) {
                std::vector<glm::vec3> parentPositions;
                parentPositions.push_back(nodes_[junctionNodeId].position);

                int pCurrent = junctionNodeId;
                int pPrev = currentNode;

                for (int i = 0; i < 5; i++) {
                    int pNext = -1;
                    for (int neighborId : nodes_[pCurrent].neighbors) {
                        if (neighborId != pPrev && nodes_[neighborId].segmentId == parentSegId) {
                            pNext = neighborId;
                            break;
                        }
                    }
                    if (pNext < 0) break;

                    parentPositions.push_back(nodes_[pNext].position);
                    pPrev = pCurrent;
                    pCurrent = pNext;
                }

                if (parentPositions.size() >= 2) {
                    parentDirection = glm::normalize(parentPositions[0] - parentPositions.back());
                }
            }
        }
    }

    // 方向のブレンド
    glm::vec3 blendedDirection;

    if (glm::length(parentDirection) > 0.001f) {
        blendedDirection = glm::normalize(baseDirection + parentDirection);
    } else {
        blendedDirection = baseDirection;
    }

    // 反発方向
    glm::vec3 repulsionDirection(0.0f);
    float repulsionRadius = glm::length(voxelSize_) * 10.0f;

    for (const auto& node : nodes_) {
        if (node.segmentId == segmentId) continue;

        glm::vec3 toEndpoint = endpoint.position - node.position;
        float dist = glm::length(toEndpoint);

        if (dist < repulsionRadius && dist > 0.001f) {
            float strength = 1.0f - (dist / repulsionRadius);
            repulsionDirection += glm::normalize(toEndpoint) * strength;
        }
    }

    if (glm::length(repulsionDirection) > 0.001f) {
        repulsionDirection = glm::normalize(repulsionDirection);
    }

    glm::vec3 finalDirection = blendedDirection * (1.0f - repulsionWeight)
                               + repulsionDirection * repulsionWeight;

    if (glm::length(finalDirection) < 0.001f) {
        return blendedDirection;
    }

    return glm::normalize(finalDirection);
}

//=============================================================================
// 適応的な方向計算（延長中に逐次更新）
//=============================================================================
glm::vec3 VesselSegmentation::calculateAdaptiveDirection(int endpointNodeId, const glm::vec3& baseDirection) {
    if (endpointNodeId < 0 || endpointNodeId >= static_cast<int>(nodes_.size())) {
        return baseDirection;
    }

    const auto& endpoint = nodes_[endpointNodeId];
    int segmentId = endpoint.segmentId;

    glm::vec3 repulsionDirection(0.0f);
    float repulsionRadius = glm::length(voxelSize_) * 15.0f;
    float totalWeight = 0.0f;

    for (const auto& node : nodes_) {
        if (node.segmentId == segmentId) continue;

        glm::vec3 toEndpoint = endpoint.position - node.position;
        float dist = glm::length(toEndpoint);

        if (dist < repulsionRadius && dist > 0.001f) {
            float strength = (1.0f - dist / repulsionRadius);
            strength *= strength;
            repulsionDirection += glm::normalize(toEndpoint) * strength;
            totalWeight += strength;
        }
    }

    if (totalWeight > 0.001f) {
        repulsionDirection /= totalWeight;
        if (glm::length(repulsionDirection) > 0.001f) {
            repulsionDirection = glm::normalize(repulsionDirection);
        }
    }

    float repulsionWeight = 0.3f;
    glm::vec3 finalDirection = baseDirection * (1.0f - repulsionWeight)
                               + repulsionDirection * repulsionWeight;

    if (glm::length(finalDirection) < 0.001f) {
        return baseDirection;
    }

    return glm::normalize(finalDirection);
}

//=============================================================================
// 端点ノードを見つける（最も先端のノードを返す）
//=============================================================================
int VesselSegmentation::findEndpointNode(int segmentId) {
    if (segmentId < 0 || segmentId >= static_cast<int>(segments_.size())) {
        return -1;
    }

    const auto& seg = segments_[segmentId];

    std::vector<int> endpoints;

    for (int nodeId : seg.nodeIds) {
        if (nodes_[nodeId].neighbors.size() == 1) {
            endpoints.push_back(nodeId);
        }
    }

    if (endpoints.empty()) {
        return -1;
    }

    if (endpoints.size() == 1) {
        return endpoints[0];
    }

    int branchNodeId = -1;

    for (int nodeId : seg.nodeIds) {
        if (nodes_[nodeId].neighbors.size() >= 3) {
            branchNodeId = nodeId;
            break;
        }
        for (int neighborId : nodes_[nodeId].neighbors) {
            if (nodes_[neighborId].segmentId != segmentId) {
                branchNodeId = nodeId;
                break;
            }
        }
        if (branchNodeId >= 0) break;
    }

    if (branchNodeId < 0 && !seg.nodeIds.empty()) {
        branchNodeId = seg.nodeIds[0];
    }

    int bestEndpoint = endpoints[0];
    float maxDist = 0.0f;

    glm::vec3 branchPos = nodes_[branchNodeId].position;

    for (int endpointId : endpoints) {
        float dist = glm::length(nodes_[endpointId].position - branchPos);
        if (dist > maxDist) {
            maxDist = dist;
            bestEndpoint = endpointId;
        }
    }

    return bestEndpoint;
}

//=============================================================================
// 手動延長（指定セグメントを延長）- 肝臓内部チェック付き
//=============================================================================
bool VesselSegmentation::extendTerminalBranch(int segmentId, int nodesToAdd) {
    if (segmentId < 0 || segmentId >= static_cast<int>(segments_.size())) {
        std::cout << "Invalid segment ID: " << segmentId << std::endl;
        return false;
    }

    if (!isTerminalSegment(segmentId)) {
        std::cout << "Segment " << segmentId << " is not a terminal branch" << std::endl;
        return false;
    }

    auto& seg = segments_[segmentId];

    int endpointNodeId = findEndpointNode(segmentId);
    if (endpointNodeId < 0) {
        std::cout << "Could not find endpoint for segment " << segmentId << std::endl;
        return false;
    }

    glm::vec3 direction = calculateExtensionDirection(endpointNodeId, 0.3f);
    float spacing = getAverageNodeSpacing(segmentId);

    std::cout << "\n=== Extending Terminal Branch ===" << std::endl;
    std::cout << "  Segment: " << segmentId << std::endl;
    std::cout << "  Endpoint node: " << endpointNodeId << std::endl;
    std::cout << "  Node spacing: " << spacing << std::endl;
    std::cout << "  Nodes to add: " << nodesToAdd << std::endl;
    std::cout << "  Initial direction: (" << direction.x << ", " << direction.y << ", " << direction.z << ")" << std::endl;

    glm::vec3 currentPos = nodes_[endpointNodeId].position;
    int prevNodeId = endpointNodeId;
    int nodesAdded = 0;
    glm::vec3 currentDirection = direction;

    for (int i = 0; i < nodesToAdd; i++) {
        currentDirection = calculateAdaptiveDirection(prevNodeId, currentDirection);
        glm::vec3 newPos = currentPos + currentDirection * spacing;

        // 肝臓メッシュ内部チェック
        if (!isInsideLiver(newPos)) {
            std::cout << "  Stopped: position outside liver at node " << (nodesAdded + 1) << std::endl;
            break;
        }

        // 他ノードとの衝突チェック
        bool tooClose = false;
        float minDistThreshold = spacing * 0.5f;

        for (const auto& node : nodes_) {
            if (node.id == prevNodeId) continue;
            if (node.segmentId == segmentId) continue;

            float dist = glm::length(node.position - newPos);
            if (dist < minDistThreshold) {
                tooClose = true;
                std::cout << "  Stopped: too close to node " << node.id << std::endl;
                break;
            }
        }

        if (tooClose) break;

        // 新しいノードを作成
        SkeletonNode newNode;
        newNode.id = static_cast<int>(nodes_.size());
        newNode.position = newPos;
        newNode.radius = nodes_[endpointNodeId].radius;
        newNode.segmentId = segmentId;

        glm::vec3 gridCoord = (newPos - gridMin_) / voxelSize_;
        newNode.voxelIndex = glm::ivec3(
            static_cast<int>(gridCoord.x),
            static_cast<int>(gridCoord.y),
            static_cast<int>(gridCoord.z)
            );

        newNode.neighbors.push_back(prevNodeId);
        nodes_[prevNodeId].neighbors.push_back(newNode.id);

        nodes_.push_back(newNode);
        seg.nodeIds.push_back(newNode.id);

        currentPos = newPos;
        prevNodeId = newNode.id;
        nodesAdded++;
    }

    std::cout << "  Nodes added: " << nodesAdded << std::endl;
    std::cout << "=================================\n" << std::endl;

    if (nodesAdded > 0) {
        updateSkeletonBuffers();
        assignTrianglesToSegments();
        return true;
    }

    return false;
}

//=============================================================================
// セグメントの長さを計算（安全版）
//=============================================================================
float VesselSegmentation::calculateSegmentLength(int segmentId) const {
    if (segmentId < 0 || segmentId >= static_cast<int>(segments_.size())) {
        return 0.0f;
    }

    const auto& seg = segments_[segmentId];

    if (seg.nodeIds.size() < 2) {
        return 0.0f;
    }

    float totalLength = 0.0f;

    for (size_t i = 1; i < seg.nodeIds.size(); i++) {
        int prevNodeId = seg.nodeIds[i - 1];
        int currNodeId = seg.nodeIds[i];

        // ノードID範囲チェック
        if (prevNodeId < 0 || prevNodeId >= static_cast<int>(nodes_.size()) ||
            currNodeId < 0 || currNodeId >= static_cast<int>(nodes_.size())) {
            continue;
        }

        totalLength += glm::length(nodes_[currNodeId].position - nodes_[prevNodeId].position);
    }

    return totalLength;
}


//=============================================================================
// ルートセグメントからの累積距離を計算（安全版）
//=============================================================================
float VesselSegmentation::calculateDistanceFromRoot(int segmentId) const {
    if (segmentId < 0 || segmentId >= static_cast<int>(segments_.size())) {
        return 0.0f;
    }

    float totalDistance = 0.0f;
    int currentSegId = segmentId;

    // 無限ループ防止
    std::set<int> visited;
    int maxIterations = static_cast<int>(segments_.size()) + 1;
    int iterations = 0;

    while (currentSegId >= 0 && currentSegId < static_cast<int>(segments_.size())) {
        // 無限ループ検出
        if (visited.count(currentSegId) > 0) {
            std::cerr << "WARNING: Loop in hierarchy at segment " << currentSegId << std::endl;
            break;
        }
        visited.insert(currentSegId);

        if (++iterations > maxIterations) {
            std::cerr << "WARNING: Too many iterations in calculateDistanceFromRoot" << std::endl;
            break;
        }

        totalDistance += calculateSegmentLength(currentSegId);

        int parentId = segments_[currentSegId].parentId;

        // ルートに達したら終了
        if (parentId < 0 || parentId == currentSegId) {
            break;
        }

        currentSegId = parentId;
    }

    return totalDistance;
}

//=============================================================================
// 位置が肝臓メッシュ内部かどうかを判定（最適化版）
//=============================================================================
bool VesselSegmentation::isInsideLiver(const glm::vec3& position) const {
    if (!hasLiverMesh_ || liverIndices_.empty() || liverVertices_.empty()) {
        return true;  // 肝臓メッシュがない場合は制限なし
    }

    // バウンディングボックスを計算（初回のみ）
    static glm::vec3 liverMin(FLT_MAX), liverMax(-FLT_MAX);
    static bool boundsComputed = false;

    if (!boundsComputed) {
        for (size_t i = 0; i < liverVertices_.size(); i += 3) {
            liverMin.x = std::min(liverMin.x, liverVertices_[i]);
            liverMin.y = std::min(liverMin.y, liverVertices_[i + 1]);
            liverMin.z = std::min(liverMin.z, liverVertices_[i + 2]);
            liverMax.x = std::max(liverMax.x, liverVertices_[i]);
            liverMax.y = std::max(liverMax.y, liverVertices_[i + 1]);
            liverMax.z = std::max(liverMax.z, liverVertices_[i + 2]);
        }
        boundsComputed = true;
    }

    // バウンディングボックス外なら即座にfalse
    if (position.x < liverMin.x || position.x > liverMax.x ||
        position.y < liverMin.y || position.y > liverMax.y ||
        position.z < liverMin.z || position.z > liverMax.z) {
        return false;
    }

    // レイキャスティング（+X方向）
    glm::vec3 rayDir(1.0f, 0.0f, 0.0f);
    int intersectionCount = 0;

    int numTriangles = static_cast<int>(liverIndices_.size()) / 3;
    int numVertices = static_cast<int>(liverVertices_.size()) / 3;

    for (int tri = 0; tri < numTriangles; tri++) {
        int idx0 = liverIndices_[tri * 3];
        int idx1 = liverIndices_[tri * 3 + 1];
        int idx2 = liverIndices_[tri * 3 + 2];

        // インデックス範囲チェック
        if (idx0 < 0 || idx0 >= numVertices ||
            idx1 < 0 || idx1 >= numVertices ||
            idx2 < 0 || idx2 >= numVertices) {
            continue;
        }

        glm::vec3 v0(liverVertices_[idx0 * 3],
                     liverVertices_[idx0 * 3 + 1],
                     liverVertices_[idx0 * 3 + 2]);
        glm::vec3 v1(liverVertices_[idx1 * 3],
                     liverVertices_[idx1 * 3 + 1],
                     liverVertices_[idx1 * 3 + 2]);
        glm::vec3 v2(liverVertices_[idx2 * 3],
                     liverVertices_[idx2 * 3 + 1],
                     liverVertices_[idx2 * 3 + 2]);

        // 三角形のバウンディングボックスで早期スキップ
        float triMinY = std::min({v0.y, v1.y, v2.y});
        float triMaxY = std::max({v0.y, v1.y, v2.y});
        float triMinZ = std::min({v0.z, v1.z, v2.z});
        float triMaxZ = std::max({v0.z, v1.z, v2.z});
        float triMaxX = std::max({v0.x, v1.x, v2.x});

        // レイが三角形のY-Z範囲外ならスキップ
        if (position.y < triMinY || position.y > triMaxY ||
            position.z < triMinZ || position.z > triMaxZ) {
            continue;
        }
        // レイの始点が三角形より右ならスキップ
        if (position.x > triMaxX) {
            continue;
        }

        // Möller–Trumbore intersection algorithm
        glm::vec3 e1 = v1 - v0;
        glm::vec3 e2 = v2 - v0;
        glm::vec3 h = glm::cross(rayDir, e2);
        float a = glm::dot(e1, h);

        if (std::abs(a) < 1e-7f) continue;

        float f = 1.0f / a;
        glm::vec3 s = position - v0;
        float u = f * glm::dot(s, h);

        if (u < 0.0f || u > 1.0f) continue;

        glm::vec3 q = glm::cross(s, e1);
        float v = f * glm::dot(rayDir, q);

        if (v < 0.0f || u + v > 1.0f) continue;

        float t = f * glm::dot(e2, q);

        if (t > 0.001f) {
            intersectionCount++;
        }
    }

    return (intersectionCount % 2) == 1;
}

//=============================================================================
// 自動延長（短い最終枝を延長）- 親方向考慮 + 肝臓内部チェック + ルート距離制限
//=============================================================================

void VesselSegmentation::autoExtendShortTerminalBranches(float shortThreshold,
                                                         float targetRatio,
                                                         float distanceRatioThreshold) {
    std::cout << "\n=== Auto Extending Short Terminal Branches ===" << std::endl;

    backupCurrentSkeleton(originalSkeleton_);
    std::cout << "  Original skeleton backed up" << std::endl;

    std::vector<int> terminalSegments = findTerminalSegments();

    if (terminalSegments.empty()) {
        std::cout << "  No terminal segments found" << std::endl;
        extendedSkeleton_ = originalSkeleton_;
        return;
    }

    std::cout << "  Found " << terminalSegments.size() << " terminal segments" << std::endl;
    std::cout << "  Calculating segment lengths and distances..." << std::endl;

    // 各最終枝の長さとルートからの距離を計算
    float maxLength = 0.0f;
    float maxDistanceFromRoot = 0.0f;
    std::map<int, float> segmentLengths;
    std::map<int, float> distancesFromRoot;

    int count = 0;
    for (int segId : terminalSegments) {
        count++;

        // ★ デバッグ: 各セグメントの処理を出力
        std::cout << "    [" << count << "/" << terminalSegments.size() << "] Segment " << segId << "..." << std::flush;

        // セグメントID範囲チェック
        if (segId < 0 || segId >= static_cast<int>(segments_.size())) {
            std::cout << " INVALID ID, skipped" << std::endl;
            continue;
        }

        std::cout << " length..." << std::flush;
        float length = calculateSegmentLength(segId);

        std::cout << " distFromRoot..." << std::flush;
        float distFromRoot = calculateDistanceFromRoot(segId);

        std::cout << " OK (len=" << length << ", dist=" << distFromRoot << ")" << std::endl;

        segmentLengths[segId] = length;
        distancesFromRoot[segId] = distFromRoot;

        maxLength = std::max(maxLength, length);
        maxDistanceFromRoot = std::max(maxDistanceFromRoot, distFromRoot);
    }

    std::cout << "  Segment analysis complete" << std::endl;

    float targetLength = maxLength * targetRatio;
    float shortLengthThreshold = maxLength * shortThreshold;
    float distanceThreshold = maxDistanceFromRoot * distanceRatioThreshold;

    std::cout << "  Max terminal branch length: " << maxLength << std::endl;
    std::cout << "  Short threshold (" << (shortThreshold * 100) << "%): " << shortLengthThreshold << std::endl;
    std::cout << "  Target length (" << (targetRatio * 100) << "%): " << targetLength << std::endl;
    std::cout << "  Max distance from root: " << maxDistanceFromRoot << std::endl;
    std::cout << "  Distance threshold (" << (distanceRatioThreshold * 100) << "%): " << distanceThreshold << std::endl;

    int extendedCount = 0;
    int totalNodesAdded = 0;
    int skippedByDistance = 0;
    int skippedByLength = 0;

    std::cout << "  Starting extension..." << std::endl;

    count = 0;
    for (int segId : terminalSegments) {
        count++;

        // セグメントID範囲チェック
        if (segId < 0 || segId >= static_cast<int>(segments_.size())) {
            continue;
        }

        float currentLength = segmentLengths[segId];
        float distFromRoot = distancesFromRoot[segId];

        // 長い枝はスキップ
        if (currentLength > shortLengthThreshold) {
            skippedByLength++;
            continue;
        }

        // ルートから十分遠い枝はスキップ
        if (distFromRoot > distanceThreshold) {
            skippedByDistance++;
            continue;
        }

        float lengthToAdd = targetLength - currentLength;
        if (lengthToAdd <= 0) continue;

        int endpointNodeId = findEndpointNode(segId);
        if (endpointNodeId < 0 || endpointNodeId >= static_cast<int>(nodes_.size())) {
            continue;
        }

        // 親方向を考慮した初期方向
        glm::vec3 baseDirection = calculateExtensionDirection(endpointNodeId, 0.3f);
        if (glm::length(baseDirection) < 0.5f) continue;

        float spacing = getAverageNodeSpacing(segId);
        if (spacing < 0.001f) {
            spacing = glm::length(voxelSize_);
        }

        int nodesToAdd = static_cast<int>(std::ceil(lengthToAdd / spacing));
        nodesToAdd = std::max(1, std::min(nodesToAdd, 50));

        auto& seg = segments_[segId];

        glm::vec3 currentPos = nodes_[endpointNodeId].position;
        int prevNodeId = endpointNodeId;
        int nodesAdded = 0;
        glm::vec3 currentDirection = baseDirection;
        float addedLength = 0.0f;

        for (int i = 0; i < nodesToAdd && addedLength < lengthToAdd; i++) {
            currentDirection = calculateAdaptiveDirection(prevNodeId, currentDirection);
            glm::vec3 newPos = currentPos + currentDirection * spacing;

            // 肝臓メッシュ内部チェック
            if (!isInsideLiver(newPos)) {
                break;
            }

            // 他ノードとの衝突チェック
            bool tooClose = false;
            float minDistThreshold = spacing * 0.5f;

            for (size_t n = 0; n < nodes_.size(); n++) {
                if (static_cast<int>(n) == prevNodeId) continue;
                if (nodes_[n].segmentId == segId) continue;

                float dist = glm::length(nodes_[n].position - newPos);
                if (dist < minDistThreshold) {
                    tooClose = true;
                    break;
                }
            }

            if (tooClose) break;

            // 新しいノードを作成
            SkeletonNode newNode;
            newNode.id = static_cast<int>(nodes_.size());
            newNode.position = newPos;
            newNode.radius = nodes_[endpointNodeId].radius;
            newNode.segmentId = segId;

            glm::vec3 gridCoord = (newPos - gridMin_) / voxelSize_;
            newNode.voxelIndex = glm::ivec3(
                static_cast<int>(gridCoord.x),
                static_cast<int>(gridCoord.y),
                static_cast<int>(gridCoord.z)
                );

            newNode.neighbors.push_back(prevNodeId);
            nodes_[prevNodeId].neighbors.push_back(newNode.id);

            nodes_.push_back(newNode);
            seg.nodeIds.push_back(newNode.id);

            addedLength += spacing;
            currentPos = newPos;
            prevNodeId = newNode.id;
            nodesAdded++;
        }

        if (nodesAdded > 0) {
            extendedCount++;
            totalNodesAdded += nodesAdded;
        }
    }

    std::cout << "\n  === Extension Summary ===" << std::endl;
    std::cout << "  Extended branches: " << extendedCount << "/" << terminalSegments.size() << std::endl;
    std::cout << "  Skipped (already long): " << skippedByLength << std::endl;
    std::cout << "  Skipped (far from root): " << skippedByDistance << std::endl;
    std::cout << "  Total nodes added: " << totalNodesAdded << std::endl;

    assignTrianglesToSegments();
    backupCurrentSkeleton(extendedSkeleton_);
    useExtendedSkeleton_ = true;

    updateSkeletonBuffers();

    std::cout << "  Extended skeleton saved" << std::endl;
    std::cout << "=============================================\n" << std::endl;
}


//=============================================================================

//=============================================================================
// 複数のOBJセグメントを一括読み込み
//=============================================================================
bool VesselSegmentation::loadOBJSegments(const std::vector<std::string>& objPaths) {
    objSegments_.clear();

    std::cout << "\n=== Loading OBJ Segments ===" << std::endl;

    for (size_t i = 0; i < objPaths.size(); i++) {
        std::string name = "S" + std::to_string(i + 1);
        if (!loadOBJSegment(objPaths[i], static_cast<int>(i + 1), name)) {
            std::cerr << "Failed to load: " << objPaths[i] << std::endl;
        }
    }

    std::cout << "Loaded " << objSegments_.size() << " OBJ segments" << std::endl;
    std::cout << "==========================\n" << std::endl;

    return !objSegments_.empty();
}

//=============================================================================
// OBJセグメンテーションと通常セグメンテーションを切り替え
//=============================================================================
void VesselSegmentation::toggleOBJSegmentation() {
    if (objSegments_.empty()) {
        std::cout << "No OBJ segments loaded - cannot toggle" << std::endl;
        return;
    }

    useOBJSegmentation_ = !useOBJSegmentation_;

    std::cout << "Segmentation mode: "
              << (useOBJSegmentation_ ? "OBJ-based (S1-S8)" : "Skeleton-based (detailed)")
              << std::endl;
}

//=============================================================================
// OBJセグメントを選択
//=============================================================================
void VesselSegmentation::selectOBJSegment(int segmentId) {
    selectedOBJSegment_ = segmentId;

    if (segmentId < 0) {
        std::cout << "OBJ segment selection cleared" << std::endl;
        return;
    }

    // 選択されたセグメント内のノード数をカウント
    int nodeCount = 0;
    for (int id : nodeToOBJSegmentId_) {
        if (id == segmentId) nodeCount++;
    }

    std::cout << "Selected OBJ segment: S" << segmentId << " (" << nodeCount << " nodes)" << std::endl;
}

//=============================================================================
// OBJセグメント選択をクリア
//=============================================================================
void VesselSegmentation::clearOBJSegmentSelection() {
    selectedOBJSegment_ = -1;
    std::cout << "OBJ segment selection cleared" << std::endl;
}

//=============================================================================
// OBJセグメンテーションをスケルトンに適用
//=============================================================================
void VesselSegmentation::applyOBJSegmentation() {
    if (objSegments_.empty()) {
        std::cerr << "No OBJ segments loaded" << std::endl;
        return;
    }

    std::cout << "\n=== Applying OBJ Segmentation ===" << std::endl;

    // ノードをOBJセグメントに割り当て
    nodeToOBJSegmentId_.resize(nodes_.size(), -1);

    std::vector<int> segmentNodeCount(objSegments_.size(), 0);

    for (size_t i = 0; i < nodes_.size(); i++) {
        const glm::vec3& pos = nodes_[i].position;

        // 各OBJセグメントに対して内部判定
        for (size_t s = 0; s < objSegments_.size(); s++) {
            if (isInsideOBJSegment(pos, static_cast<int>(s))) {
                nodeToOBJSegmentId_[i] = objSegments_[s].id;
                segmentNodeCount[s]++;
                break;
            }
        }
    }

    // 三角形をOBJセグメントに割り当て（三角形の重心で判定）
    // ★ triangleToSegment_ を使用
    triangleToOBJSegmentId_.resize(triangleToSegment_.size(), -1);

    for (size_t i = 0; i < triangleToSegment_.size(); i++) {
        // 三角形の重心を計算
        if (i * 3 + 2 >= meshIndices_.size()) continue;

        int idx0 = meshIndices_[i * 3];
        int idx1 = meshIndices_[i * 3 + 1];
        int idx2 = meshIndices_[i * 3 + 2];

        if (idx0 * 3 + 2 >= static_cast<int>(meshVertices_.size()) ||
            idx1 * 3 + 2 >= static_cast<int>(meshVertices_.size()) ||
            idx2 * 3 + 2 >= static_cast<int>(meshVertices_.size())) {
            continue;
        }

        glm::vec3 v0(meshVertices_[idx0 * 3], meshVertices_[idx0 * 3 + 1], meshVertices_[idx0 * 3 + 2]);
        glm::vec3 v1(meshVertices_[idx1 * 3], meshVertices_[idx1 * 3 + 1], meshVertices_[idx1 * 3 + 2]);
        glm::vec3 v2(meshVertices_[idx2 * 3], meshVertices_[idx2 * 3 + 1], meshVertices_[idx2 * 3 + 2]);

        glm::vec3 centroid = (v0 + v1 + v2) / 3.0f;

        for (size_t s = 0; s < objSegments_.size(); s++) {
            if (isInsideOBJSegment(centroid, static_cast<int>(s))) {
                triangleToOBJSegmentId_[i] = objSegments_[s].id;
                break;
            }
        }
    }

    // 結果を出力
    std::cout << "  Node assignment:" << std::endl;
    for (size_t s = 0; s < objSegments_.size(); s++) {
        std::cout << "    " << objSegments_[s].name << ": " << segmentNodeCount[s] << " nodes" << std::endl;
    }

    int unassignedNodes = std::count(nodeToOBJSegmentId_.begin(), nodeToOBJSegmentId_.end(), -1);
    std::cout << "    Unassigned: " << unassignedNodes << " nodes" << std::endl;

    useOBJSegmentation_ = true;

    std::cout << "=================================\n" << std::endl;
}
//=============================================================================
// OBJセグメントの色を取得
//=============================================================================
glm::vec3 VesselSegmentation::getOBJSegmentColor(int segmentId) const {
    for (const auto& seg : objSegments_) {
        if (seg.id == segmentId) {
            return seg.color;
        }
    }
    return glm::vec3(0.5f, 0.5f, 0.5f);  // デフォルト：グレー
}


// ファイルの先頭付近に追加（既存のincludeの後）

//=============================================================================
// OBJSegmentBVH 実装
//=============================================================================

void OBJSegmentBVH::build(const std::vector<float>& vertices, const std::vector<int>& indices) {
    // データをコピー
    vertices_ = vertices;
    indices_ = indices;
    nodes_.clear();

    if (indices_.empty()) return;

    std::vector<size_t> allTriangles;
    allTriangles.reserve(indices_.size() / 3);
    for (size_t i = 0; i < indices_.size(); i += 3) {
        allTriangles.push_back(i);
    }
    nodes_.reserve(allTriangles.size() * 2);
    buildRecursive(allTriangles, 0);
}

int OBJSegmentBVH::countIntersections(const glm::vec3& origin, const glm::vec3& direction) const {
    if (nodes_.empty()) return 0;
    int count = 0;
    traverseAndCount(0, origin, direction, count);
    return count;
}

int OBJSegmentBVH::buildRecursive(std::vector<size_t>& triangles, int depth) {
    BVHNode node;
    node.leftChild = -1;
    node.rightChild = -1;
    node.min = glm::vec3(std::numeric_limits<float>::max());
    node.max = glm::vec3(std::numeric_limits<float>::lowest());

    for (size_t triIdx : triangles) {
        glm::vec3 v0 = getVertex(triIdx, 0);
        glm::vec3 v1 = getVertex(triIdx, 1);
        glm::vec3 v2 = getVertex(triIdx, 2);
        node.min = glm::min(node.min, glm::min(glm::min(v0, v1), v2));
        node.max = glm::max(node.max, glm::max(glm::max(v0, v1), v2));
    }

    if (triangles.size() <= MAX_TRIANGLES_PER_LEAF || depth > 20) {
        node.triangles = triangles;
        int nodeIdx = static_cast<int>(nodes_.size());
        nodes_.push_back(node);
        return nodeIdx;
    }

    glm::vec3 extent = node.max - node.min;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent[axis]) axis = 2;
    float splitPos = (node.min[axis] + node.max[axis]) * 0.5f;

    std::vector<size_t> leftTriangles;
    std::vector<size_t> rightTriangles;
    for (size_t triIdx : triangles) {
        glm::vec3 center = getTriangleCenter(triIdx);
        if (center[axis] < splitPos) {
            leftTriangles.push_back(triIdx);
        } else {
            rightTriangles.push_back(triIdx);
        }
    }

    if (leftTriangles.empty() || rightTriangles.empty()) {
        node.triangles = triangles;
        int nodeIdx = static_cast<int>(nodes_.size());
        nodes_.push_back(node);
        return nodeIdx;
    }

    int nodeIdx = static_cast<int>(nodes_.size());
    nodes_.push_back(node);
    int leftIdx = buildRecursive(leftTriangles, depth + 1);
    int rightIdx = buildRecursive(rightTriangles, depth + 1);
    nodes_[nodeIdx].leftChild = leftIdx;
    nodes_[nodeIdx].rightChild = rightIdx;
    return nodeIdx;
}

void OBJSegmentBVH::traverseAndCount(int nodeIdx, const glm::vec3& origin,
                                     const glm::vec3& direction, int& count) const {
    const BVHNode& node = nodes_[nodeIdx];
    if (!rayBoxIntersect(origin, direction, node.min, node.max)) {
        return;
    }

    if (node.leftChild == -1) {
        for (size_t triIdx : node.triangles) {
            if (rayTriangleIntersect(origin, direction, triIdx)) {
                count++;
            }
        }
        return;
    }

    traverseAndCount(node.leftChild, origin, direction, count);
    traverseAndCount(node.rightChild, origin, direction, count);
}

bool OBJSegmentBVH::rayBoxIntersect(const glm::vec3& origin, const glm::vec3& direction,
                                    const glm::vec3& boxMin, const glm::vec3& boxMax) const {
    glm::vec3 invDir = 1.0f / direction;

    float t1 = (boxMin.x - origin.x) * invDir.x;
    float t2 = (boxMax.x - origin.x) * invDir.x;
    float tmin = std::min(t1, t2);
    float tmax = std::max(t1, t2);

    t1 = (boxMin.y - origin.y) * invDir.y;
    t2 = (boxMax.y - origin.y) * invDir.y;
    tmin = std::max(tmin, std::min(t1, t2));
    tmax = std::min(tmax, std::max(t1, t2));

    t1 = (boxMin.z - origin.z) * invDir.z;
    t2 = (boxMax.z - origin.z) * invDir.z;
    tmin = std::max(tmin, std::min(t1, t2));
    tmax = std::min(tmax, std::max(t1, t2));

    return tmax >= std::max(0.0f, tmin);
}

bool OBJSegmentBVH::rayTriangleIntersect(const glm::vec3& origin, const glm::vec3& direction,
                                         size_t triIdx) const {
    glm::vec3 v0 = getVertex(triIdx, 0);
    glm::vec3 v1 = getVertex(triIdx, 1);
    glm::vec3 v2 = getVertex(triIdx, 2);

    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 h = glm::cross(direction, edge2);
    float a = glm::dot(edge1, h);

    if (std::abs(a) < 1e-7f) return false;

    float f = 1.0f / a;
    glm::vec3 s = origin - v0;
    float u = f * glm::dot(s, h);

    if (u < 0.0f || u > 1.0f) return false;

    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(direction, q);

    if (v < 0.0f || u + v > 1.0f) return false;

    float t = f * glm::dot(edge2, q);
    return t > 1e-6f;
}

glm::vec3 OBJSegmentBVH::getVertex(size_t triIdx, int vertexIdx) const {
    int idx = indices_[triIdx + vertexIdx];
    return glm::vec3(vertices_[idx * 3], vertices_[idx * 3 + 1], vertices_[idx * 3 + 2]);
}

glm::vec3 OBJSegmentBVH::getTriangleCenter(size_t triIdx) const {
    glm::vec3 v0 = getVertex(triIdx, 0);
    glm::vec3 v1 = getVertex(triIdx, 1);
    glm::vec3 v2 = getVertex(triIdx, 2);
    return (v0 + v1 + v2) / 3.0f;
}

bool VesselSegmentation::loadOBJSegment(const std::string& objPath, int segmentId, const std::string& name) {
    std::ifstream file(objPath);
    if (!file.is_open()) {
        std::cerr << "Failed to open OBJ file: " << objPath << std::endl;
        return false;
    }

    OBJSegment seg;
    seg.id = segmentId;
    seg.name = name;
    seg.boundMin = glm::vec3(FLT_MAX);
    seg.boundMax = glm::vec3(-FLT_MAX);

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "v") {
            float x, y, z;
            iss >> x >> y >> z;
            seg.vertices.push_back(x);
            seg.vertices.push_back(y);
            seg.vertices.push_back(z);

            seg.boundMin = glm::min(seg.boundMin, glm::vec3(x, y, z));
            seg.boundMax = glm::max(seg.boundMax, glm::vec3(x, y, z));
        } else if (type == "f") {
            std::string v1, v2, v3;
            iss >> v1 >> v2 >> v3;

            auto parseIndex = [](const std::string& s) -> int {
                size_t pos = s.find('/');
                if (pos != std::string::npos) {
                    return std::stoi(s.substr(0, pos)) - 1;
                }
                return std::stoi(s) - 1;
            };

            seg.indices.push_back(parseIndex(v1));
            seg.indices.push_back(parseIndex(v2));
            seg.indices.push_back(parseIndex(v3));
        }
    }

    file.close();

    // 色を設定
    seg.color = getColorForIndex(segmentId);

    // ★ BVHを構築
    seg.bvh = std::make_unique<OBJSegmentBVH>();
    seg.bvh->build(seg.vertices, seg.indices);

    std::cout << "Loaded OBJ segment " << name << " (ID=" << segmentId
              << "): " << seg.vertices.size()/3 << " vertices, "
              << seg.indices.size()/3 << " triangles" << std::endl;

    objSegments_.push_back(std::move(seg));

    return true;
}

bool VesselSegmentation::isInsideOBJSegment(const glm::vec3& position, int segmentIndex) const {
    if (segmentIndex < 0 || segmentIndex >= static_cast<int>(objSegments_.size())) {
        return false;
    }

    const auto& seg = objSegments_[segmentIndex];

    // バウンディングボックスで早期判定
    if (position.x < seg.boundMin.x || position.x > seg.boundMax.x ||
        position.y < seg.boundMin.y || position.y > seg.boundMax.y ||
        position.z < seg.boundMin.z || position.z > seg.boundMax.z) {
        return false;
    }

    // ★ BVHを使ったレイキャスティング
    if (seg.bvh && !seg.bvh->isEmpty()) {
        glm::vec3 rayDir(1.0f, 0.0f, 0.0f);  // +X方向
        int intersectionCount = seg.bvh->countIntersections(position, rayDir);
        return (intersectionCount % 2) == 1;
    }

    // フォールバック：BVHがない場合（通常は到達しない）
    return false;
}

//=============================================================================
// 位置からOBJセグメントIDを取得
//=============================================================================
int VesselSegmentation::getOBJSegmentAtPosition(const glm::vec3& position) const {
    if (objSegments_.empty()) {
        return -1;
    }

    // 各OBJセグメントで内部判定
    for (size_t s = 0; s < objSegments_.size(); s++) {
        if (isInsideOBJSegment(position, static_cast<int>(s))) {
            return objSegments_[s].id;
        }
    }

    // 内部でなければ最も近いセグメントを探す
    float minDist = FLT_MAX;
    int nearestId = -1;

    for (const auto& seg : objSegments_) {
        glm::vec3 segCenter = (seg.boundMin + seg.boundMax) * 0.5f;
        float dist = glm::length(position - segCenter);

        glm::vec3 segSize = seg.boundMax - seg.boundMin;
        float avgSize = (segSize.x + segSize.y + segSize.z) / 3.0f;

        // セグメントサイズの範囲内なら候補に
        if (dist < minDist && dist < avgSize * 1.5f) {
            minDist = dist;
            nearestId = seg.id;
        }
    }

    return nearestId;
}

//=============================================================================
// セグメンテーションモード切り替え
//=============================================================================


int VesselSegmentation::getVoronoiBranchAtPosition(const glm::vec3& pos) const {
    if (!hasVoronoi3D()) return -1;
    return voronoiSegmenter_->getBranchAtPosition(pos);
}

int VesselSegmentation::getVoronoiSegmentAtPosition(const glm::vec3& pos) const {
    if (!hasVoronoi3D()) return -1;
    return voronoiSegmenter_->getTerminalSegmentAtPosition(pos);
}

//=============================================================================
// セグメントの三角形リストを再構築（ヘルパー）
//=============================================================================

void VesselSegmentation::rebuildSegmentTriangleLists() {
    for (auto& seg : segments_) {
        seg.triangleIndices.clear();
    }

    for (const auto& pair : triangleToSegment_) {
        int triIdx = pair.first;
        int segId = pair.second;
        if (segId >= 0 && segId < static_cast<int>(segments_.size())) {
            segments_[segId].triangleIndices.insert(triIdx);
        }
    }
}

//=============================================================================
// 描画支援：現在のモードに応じた三角形の色を取得
//=============================================================================

glm::vec3 VesselSegmentation::getTriangleColorByCurrentMode(int triangleIndex) const {
    switch (currentMode_) {
    case SegmentationMode::OBJ: {
        // OBJセグメントの色
        if (triangleIndex >= 0 &&
            triangleIndex < static_cast<int>(triangleToOBJSegmentId_.size())) {
            int objId = triangleToOBJSegmentId_[triangleIndex];
            if (objId >= 0 && objId < static_cast<int>(objSegments_.size())) {
                return objSegments_[objId].color;
            }
        }
        return glm::vec3(0.5f);
    }

    case SegmentationMode::SkeletonDistance: {
        // 通常のセグメントの色
        auto it = triangleToSegment_.find(triangleIndex);
        if (it != triangleToSegment_.end()) {
            int segId = it->second;
            if (segId >= 0 && segId < static_cast<int>(segments_.size())) {
                return segments_[segId].color;
            }
        }
        return glm::vec3(0.5f);
    }

    case SegmentationMode::Voronoi3D: {
        // Voronoiブランチの色
        if (voronoiSegmenter_ &&
            triangleIndex >= 0 &&
            triangleIndex < static_cast<int>(triangleToVoronoiSegment_.size())) {

            // ブランチIDを取得するために末端セグメントIDからマッピング
            int terminalSeg = triangleToVoronoiSegment_[triangleIndex];
            const auto& branches = voronoiSegmenter_->getBranches();

            for (const auto& branch : branches) {
                if (branch.terminalSegmentId == terminalSeg) {
                    return branch.color;
                }
            }
        }
        return glm::vec3(0.5f);
    }

    default:
        return glm::vec3(0.5f);
    }
}



VesselSegmentation* VesselSegmentation::create(
    const std::string& portalPath,
    const std::vector<std::string>& objPaths,
    SoftBodyGPUDuo* liver,
    SoftBodyGPUDuo* portal,
    int resolution)
{
    std::cout << "\n=== Analyzing Portal Vein Skeleton ===" << std::endl;

    auto* skeleton = new VesselSegmentation(resolution);

    if (!skeleton->analyzeFromFile(portalPath)) {
        std::cerr << "Failed to analyze portal vein" << std::endl;
        return skeleton;
    }

    std::cout << "Portal skeleton analyzed:" << std::endl;
    std::cout << "  Segments: " << skeleton->getSegmentCount() << std::endl;
    std::cout << "  Nodes: " << skeleton->getNodes().size() << std::endl;

    // トポロジー修正
    auto debugInfo = skeleton->analyzeTopology();
    std::cout << "  Loop edges: " << debugInfo.loopCount << std::endl;
    std::cout << "  Connected components: " << debugInfo.componentCount << std::endl;

    if (debugInfo.loopCount > 0) {
        int removed = skeleton->removeLoops();
        std::cout << "  Removed " << removed << " loop edges" << std::endl;
    }

    if (debugInfo.componentCount > 1) {
        int connected = skeleton->connectDisconnectedComponents();
        std::cout << "  Connected " << connected << " components" << std::endl;
    }

    if (skeleton->verifyTreeStructure()) {
        std::cout << "  Tree structure: VALID" << std::endl;
    } else {
        std::cout << "  Tree structure: INVALID (some issues remain)" << std::endl;
    }

    // 肝臓メッシュ設定
    if (liver) {
        skeleton->setLiverMesh(liver->highRes_positions, liver->highResMeshData.tetSurfaceTriIds);
    }

    // OBJセグメント読み込み・適用
    if (!objPaths.empty() && skeleton->loadOBJSegments(objPaths)) {
        skeleton->applyOBJSegmentation();
        if (liver) liver->bindOBJSegments(*skeleton);
        if (portal) portal->bindOBJSegments(*skeleton);
    }

    // スケルトンにバインド
    if (liver) liver->bindToSkeleton(*skeleton);
    if (portal) portal->bindToSkeleton(*skeleton);

    // 色更新
    if (liver) liver->updateOBJSegmentColors(*skeleton);
    if (portal) portal->updateOBJSegmentColors(*skeleton);

    // 選択連動コールバック
    if (portal && liver) {
        portal->skeletonBinding.onSelectionChanged = [liver](const std::set<int>& selected) {
            liver->selectSegments(selected);
            if (selected.empty()) {
                std::cout << "Selection cleared" << std::endl;
            } else {
                std::cout << "Selected " << selected.size() << " segments: ";
                for (int id : selected) std::cout << id << " ";
                std::cout << std::endl;
            }
        };
    }

    if (liver) {
        liver->skeletonBinding.onSelectionChanged = [](const std::set<int>& selected) {
            std::cout << "Selection changed: " << selected.size() << " segments" << std::endl;
        };
    }

    // デバッグ出力
    std::cout << "\n=== Binding Skeleton to SoftBody ===" << std::endl;
    std::cout << "Skeleton binding completed" << std::endl;
    if (liver) liver->printSkeletonBindingStats();

    return skeleton;
}

VesselSegmentation* VesselSegmentation::create(
    const std::string& portalPath,
    SoftBodyGPUDuo* liver,
    SoftBodyGPUDuo* portal,
    int resolution)
{
    return create(portalPath, {}, liver, portal, resolution);
}





//=============================================================================
// VoxelSkeletonSegmentation.cpp - GEOGRAM オプショナル対応版
//
// 以下の4つの関数を既存のVoxelSkeletonSegmentation.cppの該当箇所に
// コピー＆ペーストで置き換えてください
//
// 置き換える関数:
//   1. setSegmentationMode()      - 約4268行目
//   2. cycleSegmentationMode()    - 約4318行目
//   3. applyVoronoi3DSegmentation() - 約4342行目
//   4. buildVoronoi3D()           - 約4484行目
//=============================================================================

//=============================================================================
// 1. setSegmentationMode() - 約4268行目を置き換え
//=============================================================================
void VesselSegmentation::setSegmentationMode(SegmentationMode mode) {
    if (mode == currentMode_) {
        return;
    }

    // ★ 追加: Voronoi3Dモードの場合、利用可能かチェック
    if (mode == SegmentationMode::Voronoi3D && !isVoronoi3DAvailable()) {
        std::cout << "\n========================================" << std::endl;
        std::cout << " WARNING: Voronoi3D mode not available" << std::endl;
        std::cout << "   GEOGRAM is not linked in this build." << std::endl;
        std::cout << "   Available modes: OBJ, SkeletonDistance" << std::endl;
        std::cout << "========================================" << std::endl;
        return;  // モード変更しない
    }

    SegmentationMode prevMode = currentMode_;
    currentMode_ = mode;

    std::cout << "\n========================================" << std::endl;
    std::cout << " Segmentation Mode Changed" << std::endl;
    std::cout << "  From: " << getSegmentationModeName(prevMode) << std::endl;
    std::cout << "  To:   " << getSegmentationModeName(mode) << std::endl;
    std::cout << "========================================" << std::endl;

    switch (mode) {
    case SegmentationMode::OBJ:
        // OBJベースのセグメンテーション
        if (objSegments_.empty()) {
            std::cout << "  Warning: No OBJ segments loaded" << std::endl;
        } else {
            applyOBJSegmentation();
            std::cout << "  Applied OBJ-based segmentation ("
                      << objSegments_.size() << " segments)" << std::endl;
        }
        break;

    case SegmentationMode::SkeletonDistance:
        // 従来の距離ベースセグメンテーション
        if (triangleToSegment_Distance_.empty()) {
            // 初回は計算
            assignTrianglesToSegments();
            triangleToSegment_Distance_ = triangleToSegment_;
        } else {
            // キャッシュから復元
            triangleToSegment_ = triangleToSegment_Distance_;
            rebuildSegmentTriangleLists();
        }
        std::cout << "  Applied Skeleton Distance-based segmentation" << std::endl;
        break;

    case SegmentationMode::Voronoi3D:
        // Voronoi 3D（パスサンプリング）
        applyVoronoi3DSegmentation();
        std::cout << "  Applied Voronoi 3D (Path-based) segmentation" << std::endl;
        break;
    }

    std::cout << std::endl;
}

//=============================================================================
// 2. cycleSegmentationMode() - 約4318行目を置き換え
//=============================================================================
void VesselSegmentation::cycleSegmentationMode() {
    // ★ 変更: 利用可能なモードのみを循環
    std::vector<SegmentationMode> availableModes = getAvailableModes();

    if (availableModes.empty()) {
        std::cerr << "[cycleSegmentationMode] No available modes!" << std::endl;
        return;
    }

    // 現在のモードのインデックスを見つける
    size_t currentIdx = 0;
    for (size_t i = 0; i < availableModes.size(); ++i) {
        if (availableModes[i] == currentMode_) {
            currentIdx = i;
            break;
        }
    }

    // 次のモードへ（循環）
    size_t nextIdx = (currentIdx + 1) % availableModes.size();
    SegmentationMode nextMode = availableModes[nextIdx];

    setSegmentationMode(nextMode);
}

//=============================================================================
// 3. applyVoronoi3DSegmentation() - 約4342行目を置き換え
//=============================================================================
void VesselSegmentation::applyVoronoi3DSegmentation() {
    std::cout << "\n=== Applying Voronoi 3D Segmentation ===" << std::endl;

    // ★ 追加: GEOGRAMが利用不可の場合は早期リターン
    if (!isVoronoi3DAvailable()) {
        std::cerr << "  ERROR: Voronoi3D not available (GEOGRAM not linked)" << std::endl;
        std::cerr << "  Please use OBJ or SkeletonDistance mode." << std::endl;
        return;
    }

    // 既にVoronoiがビルド済みで、キャッシュがあれば再構築しない
    if (hasVoronoi3D() && !triangleToSegment_Voronoi_.empty()) {
        triangleToSegment_ = triangleToSegment_Voronoi_;
        rebuildSegmentTriangleLists();
        std::cout << "  Restored from cache (no rebuild)" << std::endl;
        std::cout << "  Branches: " << voronoiSegmenter_->getNumBranches() << std::endl;
        return;
    }

    // Voronoiがビルド済みだがキャッシュがない場合（三角形割り当てのみ再実行）
    if (hasVoronoi3D() && triangleToSegment_Voronoi_.empty()) {
        std::cout << "  Voronoi already built, assigning triangles..." << std::endl;
    } else {
        // 初回：Voronoiを構築
        std::cout << "  Building Voronoi 3D (Path-based)..." << std::endl;
        if (!buildVoronoi3D()) {
            std::cerr << "  Failed to build Voronoi 3D" << std::endl;
            return;
        }
    }

    int numTriangles = meshIndices_.size() / 3;
    if (numTriangles == 0) {
        std::cerr << "  No triangles" << std::endl;
        return;
    }

    // 頂点データを準備
    std::vector<float> vertices(meshVertices_.begin(), meshVertices_.end());
    std::vector<int> indices(meshIndices_.begin(), meshIndices_.end());

    // Voronoiで三角形を割り当て
    voronoiSegmenter_->assignTrianglesToSegments(
        vertices, indices, triangleToVoronoiSegment_
        );

    // 結果をtriangleToSegment_に適用
    triangleToSegment_.clear();
    for (int t = 0; t < numTriangles; ++t) {
        int segId = triangleToVoronoiSegment_[t];
        if (segId >= 0) {
            triangleToSegment_[t] = segId;
        }
    }

    // セグメントの三角形リストを再構築
    rebuildSegmentTriangleLists();

    // キャッシュに保存
    triangleToSegment_Voronoi_ = triangleToSegment_;

    std::cout << "  Applied to " << numTriangles << " triangles" << std::endl;
    std::cout << "  Branches: " << voronoiSegmenter_->getNumBranches() << std::endl;
    std::cout << "=== Voronoi 3D Segmentation Complete ===" << std::endl;
}

//=============================================================================
// 4. buildVoronoi3D() - 約4484行目を置き換え
//=============================================================================
bool VesselSegmentation::buildVoronoi3D(float samplingInterval) {
    // ★ 追加: GEOGRAMが利用不可の場合
    if (!isVoronoi3DAvailable()) {
        std::cerr << "[buildVoronoi3D] ERROR: GEOGRAM not available" << std::endl;
        std::cerr << "  Voronoi3D mode requires GEOGRAM library." << std::endl;
        std::cerr << "  Please use OBJ or SkeletonDistance mode instead." << std::endl;
        return false;
    }

    std::cout << "\n=== Building Voronoi 3D (Node-based) ===" << std::endl;

    if (nodes_.empty() || segments_.empty()) {
        std::cerr << "[VesselSegmentation] Error: No skeleton data" << std::endl;
        return false;
    }

    // Voronoiセグメンターを作成
    if (!voronoiSegmenter_) {
        voronoiSegmenter_ = std::make_unique<VoronoiPathSegmenter>();
    }

    // ノード位置とセグメントIDを準備
    std::vector<glm::vec3> nodePositions;
    std::vector<int> nodeToSegmentId;

    nodePositions.reserve(nodes_.size());
    nodeToSegmentId.reserve(nodes_.size());

    for (size_t i = 0; i < nodes_.size(); ++i) {
        nodePositions.push_back(nodes_[i].position);
        nodeToSegmentId.push_back(nodes_[i].segmentId);
    }

    std::cout << "  Nodes: " << nodePositions.size() << std::endl;

    // Voronoiを構築
    bool success = voronoiSegmenter_->build(nodes_, segments_, rootSegmentId_, samplingInterval);

    if (success) {
        std::cout << "  Voronoi 3D built successfully" << std::endl;
    } else {
        std::cerr << "  Voronoi 3D build failed" << std::endl;
    }

    return success;
}

} // namespace VoxelSkeleton
