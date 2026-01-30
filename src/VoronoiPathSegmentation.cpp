// VoronoiPathSegmentation.cpp
// WASM/Emscripten & Native 両対応版

#include "VoronoiPathSegmentation.h"
#include <cmath>
#include <limits>

#ifdef _OPENMP
#ifndef __EMSCRIPTEN__
#include <omp.h>
#endif
#endif

#if GEOGRAM_AVAILABLE
#include <geogram/basic/logger.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#endif

namespace VoxelSkeleton {

//=============================================================================
// 静的メンバ
//=============================================================================
bool VoronoiPathSegmenter::geogramInitialized_ = false;

//=============================================================================
// コンストラクタ・デストラクタ
//=============================================================================
VoronoiPathSegmenter::VoronoiPathSegmenter()
    : numSamples_(0)
    , isBuilt_(false)
#if GEOGRAM_AVAILABLE
    , delaunay_(nullptr)
#endif
{
}

VoronoiPathSegmenter::~VoronoiPathSegmenter() {
}

//=============================================================================
// Geogram初期化
//=============================================================================
void VoronoiPathSegmenter::initializeGeogram() {
#if GEOGRAM_AVAILABLE
    if (geogramInitialized_) {
        return;
    }

    try {
        GEO::initialize();
        GEO::Logger::instance()->set_quiet(true);
        GEO::CmdLine::import_arg_group("algo");
        geogramInitialized_ = true;
        std::cout << "[VoronoiPathSegmenter] Geogram initialized" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "[VoronoiPathSegmenter] Geogram init error: " << e.what() << std::endl;
    }
#else
    // GEOGRAMなし
    static bool warned = false;
    if (!warned) {
        std::cout << "[VoronoiPathSegmenter] GEOGRAM not available on "
                  << getPlatformName() << std::endl;
        warned = true;
    }
#endif
}

//=============================================================================
// Voronoi構築
//=============================================================================
bool VoronoiPathSegmenter::buildVoronoi() {
#if GEOGRAM_AVAILABLE
    if (numSamples_ == 0) {
        return false;
    }

    initializeGeogram();

    try {
        std::cout << "  Building GEOGRAM Delaunay with " << numSamples_ << " points..." << std::endl;

        delaunay_ = GEO::Delaunay::create(3, "BDEL");

        if (!delaunay_) {
            std::cerr << "[VoronoiPathSegmenter] Failed to create Delaunay" << std::endl;
            return false;
        }

        delaunay_->set_vertices(
            static_cast<GEO::index_t>(numSamples_),
            sampleCoords_.data()
            );

        std::cout << "  GEOGRAM Delaunay built: " << delaunay_->nb_cells() << " cells" << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[VoronoiPathSegmenter] Voronoi build error: " << e.what() << std::endl;
        return false;
    }
#else
    std::cerr << "[VoronoiPathSegmenter] Cannot build Voronoi: GEOGRAM not available" << std::endl;
    return false;
#endif
}

//=============================================================================
// 色を割り当て
//=============================================================================
void VoronoiPathSegmenter::assignColors() {
    static const std::vector<glm::vec3> palette = {
        glm::vec3(0.9f, 0.2f, 0.2f),   // 赤
        glm::vec3(0.2f, 0.7f, 0.9f),   // シアン
        glm::vec3(0.2f, 0.9f, 0.3f),   // 緑
        glm::vec3(0.9f, 0.6f, 0.1f),   // オレンジ
        glm::vec3(0.7f, 0.2f, 0.9f),   // 紫
        glm::vec3(0.9f, 0.9f, 0.2f),   // 黄
        glm::vec3(0.9f, 0.3f, 0.6f),   // ピンク
        glm::vec3(0.3f, 0.9f, 0.9f),   // 水色
    };

    for (size_t i = 0; i < branches_.size(); ++i) {
        branches_[i].color = palette[i % palette.size()];
    }
}

//=============================================================================
// buildFromNodes
//=============================================================================
bool VoronoiPathSegmenter::buildFromNodes(
    const std::vector<glm::vec3>& nodePositions,
    const std::vector<int>& nodeToSegmentId)
{
#if !GEOGRAM_AVAILABLE
    (void)nodePositions; (void)nodeToSegmentId;
    std::cerr << "[VoronoiPathSegmenter] Cannot build: GEOGRAM not available" << std::endl;
    return false;
#else
    if (nodePositions.empty() || nodePositions.size() != nodeToSegmentId.size()) {
        std::cerr << "[VoronoiPathSegmenter] Error: Invalid input" << std::endl;
        return false;
    }

    std::cout << "\n=== Building Voronoi from Skeleton Nodes ===" << std::endl;

    sampleCoords_.clear();
    sampleToBranch_.clear();
    sampleToTerminal_.clear();
    branches_.clear();
    isBuilt_ = false;

    for (size_t i = 0; i < nodePositions.size(); ++i) {
        const glm::vec3& pos = nodePositions[i];
        int segId = nodeToSegmentId[i];

        sampleCoords_.push_back(static_cast<double>(pos.x));
        sampleCoords_.push_back(static_cast<double>(pos.y));
        sampleCoords_.push_back(static_cast<double>(pos.z));

        sampleToBranch_.push_back(segId);
        sampleToTerminal_.push_back(segId);
    }

    numSamples_ = nodePositions.size();

    if (!buildVoronoi()) {
        return false;
    }

    isBuilt_ = true;
    std::cout << "=== Voronoi Build Complete ===" << std::endl;
    return true;
#endif
}

//=============================================================================
// サンプリング点生成
//=============================================================================
void VoronoiPathSegmenter::generateSamplePoints(float interval) {
    sampleCoords_.clear();
    sampleToBranches_.clear();
    sampleToTerminals_.clear();

    const float cellSize = interval * 0.5f;
    std::map<std::tuple<int,int,int>, size_t> positionToSampleIdx;

    for (const auto& branch : branches_) {
        if (branch.pathPoints.size() < 2) continue;

        for (size_t i = 0; i < branch.pathPoints.size() - 1; ++i) {
            const glm::vec3& p0 = branch.pathPoints[i];
            const glm::vec3& p1 = branch.pathPoints[i + 1];

            glm::vec3 dir = p1 - p0;
            float segLen = glm::length(dir);
            if (segLen < 1e-6f) continue;
            dir /= segLen;

            int numSamples = std::max(1, static_cast<int>(std::ceil(segLen / interval)));
            float step = segLen / numSamples;

            for (int s = 0; s <= numSamples; ++s) {
                if (s == numSamples && i < branch.pathPoints.size() - 2) continue;

                glm::vec3 samplePos = p0 + dir * (step * s);

                int cx = static_cast<int>(std::floor(samplePos.x / cellSize));
                int cy = static_cast<int>(std::floor(samplePos.y / cellSize));
                int cz = static_cast<int>(std::floor(samplePos.z / cellSize));
                auto key = std::make_tuple(cx, cy, cz);

                auto it = positionToSampleIdx.find(key);
                if (it != positionToSampleIdx.end()) {
                    size_t idx = it->second;
                    auto& branchList = sampleToBranches_[idx];
                    if (std::find(branchList.begin(), branchList.end(), branch.branchId) == branchList.end()) {
                        branchList.push_back(branch.branchId);
                        sampleToTerminals_[idx].push_back(branch.terminalSegmentId);
                    }
                } else {
                    size_t idx = sampleToBranches_.size();
                    positionToSampleIdx[key] = idx;

                    sampleCoords_.push_back(static_cast<double>(samplePos.x));
                    sampleCoords_.push_back(static_cast<double>(samplePos.y));
                    sampleCoords_.push_back(static_cast<double>(samplePos.z));

                    sampleToBranches_.push_back({branch.branchId});
                    sampleToTerminals_.push_back({branch.terminalSegmentId});
                }
            }
        }
    }

    numSamples_ = sampleToBranches_.size();
}

//=============================================================================
// クエリ関数
//=============================================================================

std::vector<int> VoronoiPathSegmenter::getBranchesAtPosition(const glm::vec3& position) const {
#if !GEOGRAM_AVAILABLE
    (void)position;
    return {};
#else
    if (!isBuilt_ || !delaunay_) return {};

    double query[3] = {
        static_cast<double>(position.x),
        static_cast<double>(position.y),
        static_cast<double>(position.z)
    };

    GEO::index_t nearest = delaunay_->nearest_vertex(query);
    if (nearest >= sampleToBranches_.size()) return {};

    return sampleToBranches_[nearest];
#endif
}

std::vector<int> VoronoiPathSegmenter::getTerminalSegmentsAtPosition(const glm::vec3& position) const {
#if !GEOGRAM_AVAILABLE
    (void)position;
    return {};
#else
    if (!isBuilt_ || !delaunay_) return {};

    double query[3] = {
        static_cast<double>(position.x),
        static_cast<double>(position.y),
        static_cast<double>(position.z)
    };

    GEO::index_t nearest = delaunay_->nearest_vertex(query);
    if (nearest >= sampleToTerminals_.size()) return {};

    return sampleToTerminals_[nearest];
#endif
}

int VoronoiPathSegmenter::getBranchAtPosition(const glm::vec3& position) const {
    auto branches = getBranchesAtPosition(position);
    if (branches.empty()) return -1;
    return *std::min_element(branches.begin(), branches.end());
}

int VoronoiPathSegmenter::getTerminalSegmentAtPosition(const glm::vec3& position) const {
    auto terminals = getTerminalSegmentsAtPosition(position);
    if (terminals.empty()) return -1;
    return *std::min_element(terminals.begin(), terminals.end());
}

int VoronoiPathSegmenter::getPrimaryBranchAtPosition(const glm::vec3& position) const {
    return getBranchAtPosition(position);
}

glm::vec3 VoronoiPathSegmenter::getColorAtPosition(const glm::vec3& position) const {
#if !GEOGRAM_AVAILABLE
    (void)position;
    return glm::vec3(0.5f);
#else
    if (!isBuilt_ || !delaunay_) return glm::vec3(0.5f);

    double query[3] = {
        static_cast<double>(position.x),
        static_cast<double>(position.y),
        static_cast<double>(position.z)
    };

    GEO::index_t nearest = delaunay_->nearest_vertex(query);
    if (nearest >= sampleToBranches_.size()) return glm::vec3(0.5f);

    const auto& branchIds = sampleToBranches_[nearest];
    if (branchIds.empty()) return glm::vec3(0.5f);

    glm::vec3 mixedColor(0.0f);
    int validCount = 0;

    for (int branchId : branchIds) {
        if (branchId >= 0 && branchId < static_cast<int>(branches_.size())) {
            mixedColor += branches_[branchId].color;
            validCount++;
        }
    }

    return (validCount > 0) ? mixedColor / static_cast<float>(validCount) : glm::vec3(0.5f);
#endif
}

//=============================================================================
// バッチ処理
//=============================================================================
void VoronoiPathSegmenter::assignTrianglesToSegments(
    const std::vector<float>& vertices,
    const std::vector<int>& indices,
    std::vector<int>& outTriangleSegments) const {

    size_t numTriangles = indices.size() / 3;
    outTriangleSegments.resize(numTriangles);

#if !GEOGRAM_AVAILABLE
    (void)vertices;
    std::fill(outTriangleSegments.begin(), outTriangleSegments.end(), -1);
    return;
#else
    if (!isBuilt_) {
        std::fill(outTriangleSegments.begin(), outTriangleSegments.end(), -1);
        return;
    }

    // OpenMP使用（WASMではシングルスレッド）
#if defined(_OPENMP) && !defined(__EMSCRIPTEN__)
#pragma omp parallel for schedule(dynamic, 100)
#endif
    for (int t = 0; t < static_cast<int>(numTriangles); ++t) {
        int i0 = indices[t * 3 + 0];
        int i1 = indices[t * 3 + 1];
        int i2 = indices[t * 3 + 2];

        glm::vec3 v0(vertices[i0 * 3], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
        glm::vec3 v1(vertices[i1 * 3], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
        glm::vec3 v2(vertices[i2 * 3], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);

        glm::vec3 centroid = (v0 + v1 + v2) / 3.0f;
        outTriangleSegments[t] = getTerminalSegmentAtPosition(centroid);
    }
#endif
}

} // namespace VoxelSkeleton
