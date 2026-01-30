#ifndef MESH_CUTTING_GPU_DUO_H
#define MESH_CUTTING_GPU_DUO_H

#include <glm/glm.hpp>
#include <vector>
#include <set>
#include <map>
#include <array>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <tuple>
#include <queue>
#include <omp.h>
#include "SoftBodyGPUDuo.h"

namespace MeshCuttingGPUDuo {

// 既存のTetrahedronIntersection名前空間をここに含める
// 四面体交差判定用のヘルパー関数
namespace TetrahedronIntersection {

// 三角形と線分の交差判定
bool triangleSegmentIntersect(const glm::vec3& p1, const glm::vec3& p2, const glm::vec3& p3,
                              const glm::vec3& q1, const glm::vec3& q2) {
    glm::vec3 d = q2 - q1;
    glm::vec3 e1 = p2 - p1;
    glm::vec3 e2 = p3 - p1;
    glm::vec3 h = glm::cross(d, e2);
    float a = glm::dot(e1, h);

    if (a > -0.00001f && a < 0.00001f) return false;

    float f = 1.0f / a;
    glm::vec3 s = q1 - p1;
    float u = f * glm::dot(s, h);

    if (u < 0.0f || u > 1.0f) return false;

    glm::vec3 q = glm::cross(s, e1);
    float v = f * glm::dot(d, q);

    if (v < 0.0f || u + v > 1.0f) return false;

    float t = f * glm::dot(e2, q);
    return (t > 0.00001f && t < 0.99999f);
}

// 点が四面体内部にあるかチェック
bool pointInTetrahedron(const glm::vec3& p,
                        const glm::vec3& a, const glm::vec3& b,
                        const glm::vec3& c, const glm::vec3& d) {
    auto sign = [](const glm::vec3& p1, const glm::vec3& p2,
                   const glm::vec3& p3, const glm::vec3& p4) {
        glm::vec3 v1 = p2 - p1;
        glm::vec3 v2 = p3 - p1;
        glm::vec3 v3 = p4 - p1;
        return glm::dot(glm::cross(v1, v2), v3);
    };

    float d0 = sign(p, a, b, c);
    float d1 = sign(p, a, c, d);
    float d2 = sign(p, a, d, b);
    float d3 = sign(p, b, d, c);

    bool hasNeg = (d0 < 0) || (d1 < 0) || (d2 < 0) || (d3 < 0);
    bool hasPos = (d0 > 0) || (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(hasNeg && hasPos);
}

// AABBの交差チェック（高速な事前判定）
bool aabbIntersect(const glm::vec3& min1, const glm::vec3& max1,
                   const glm::vec3& min2, const glm::vec3& max2) {
    return (min1.x <= max2.x && max1.x >= min2.x) &&
           (min1.y <= max2.y && max1.y >= min2.y) &&
           (min1.z <= max2.z && max1.z >= min2.z);
}

// 四面体のAABBを計算
std::pair<glm::vec3, glm::vec3> computeTetrahedronAABB(
    const glm::vec3& a, const glm::vec3& b,
    const glm::vec3& c, const glm::vec3& d) {
    glm::vec3 minPoint = glm::min(glm::min(glm::min(a, b), c), d);
    glm::vec3 maxPoint = glm::max(glm::max(glm::max(a, b), c), d);
    return {minPoint, maxPoint};
}

// 2つの四面体が交差しているかチェック
bool tetrahedronsIntersect(const glm::vec3& a1, const glm::vec3& b1,
                           const glm::vec3& c1, const glm::vec3& d1,
                           const glm::vec3& a2, const glm::vec3& b2,
                           const glm::vec3& c2, const glm::vec3& d2) {
    // まずAABBで高速判定
    auto [min1, max1] = computeTetrahedronAABB(a1, b1, c1, d1);
    auto [min2, max2] = computeTetrahedronAABB(a2, b2, c2, d2);

    if (!aabbIntersect(min1, max1, min2, max2)) {
        return false;
    }

    // 各四面体の頂点が他方の内部にあるかチェック
    if (pointInTetrahedron(a1, a2, b2, c2, d2) ||
        pointInTetrahedron(b1, a2, b2, c2, d2) ||
        pointInTetrahedron(c1, a2, b2, c2, d2) ||
        pointInTetrahedron(d1, a2, b2, c2, d2)) {
        return true;
    }

    if (pointInTetrahedron(a2, a1, b1, c1, d1) ||
        pointInTetrahedron(b2, a1, b1, c1, d1) ||
        pointInTetrahedron(c2, a1, b1, c1, d1) ||
        pointInTetrahedron(d2, a1, b1, c1, d1)) {
        return true;
    }

    // エッジと面の交差をチェック
    // 四面体1のエッジと四面体2の面
    std::vector<std::pair<glm::vec3, glm::vec3>> edges1 = {
        {a1, b1}, {a1, c1}, {a1, d1}, {b1, c1}, {b1, d1}, {c1, d1}
    };

    std::vector<std::tuple<glm::vec3, glm::vec3, glm::vec3>> faces2 = {
        {a2, b2, c2}, {a2, b2, d2}, {a2, c2, d2}, {b2, c2, d2}
    };

    for (const auto& edge : edges1) {
        for (const auto& face : faces2) {
            if (triangleSegmentIntersect(std::get<0>(face), std::get<1>(face),
                                         std::get<2>(face), edge.first, edge.second)) {
                return true;
            }
        }
    }

    // 四面体2のエッジと四面体1の面
    std::vector<std::pair<glm::vec3, glm::vec3>> edges2 = {
        {a2, b2}, {a2, c2}, {a2, d2}, {b2, c2}, {b2, d2}, {c2, d2}
    };

    std::vector<std::tuple<glm::vec3, glm::vec3, glm::vec3>> faces1 = {
        {a1, b1, c1}, {a1, b1, d1}, {a1, c1, d1}, {b1, c1, d1}
    };

    for (const auto& edge : edges2) {
        for (const auto& face : faces1) {
            if (triangleSegmentIntersect(std::get<0>(face), std::get<1>(face),
                                         std::get<2>(face), edge.first, edge.second)) {
                return true;
            }
        }
    }

    return false;
}
}

std::vector<int> findIntersectingTetrahedraBySurfaceTriangles(
    const SoftBodyGPUDuo::MeshData& cutMesh,
    const SoftBodyGPUDuo::MeshData& tetMesh,
    int intersectionMode = 2,  // 0:表面のみ, 1:重心ベース, 2:全頂点内部, 3:表面+内部ダミー
    bool verbose = true) {

    auto startTime = std::chrono::high_resolution_clock::now();

    std::vector<int> intersectingTets;
    size_t numTets = tetMesh.tetIds.size() / 4;
    size_t numSurfaceTris = cutMesh.tetSurfaceTriIds.size() / 3;

    const char* modeNames[] = {
        "SURFACE-ONLY",
        "CENTROID-BASED",
        "ALL-VERTICES-INSIDE",
        "SURFACE + SAMPLED-INTERNAL"
    };

    if (verbose) {
        std::cout << "\n=== Surface Triangle-based Intersection Detection ===" << std::endl;
        std::cout << "Method: " << modeNames[intersectionMode] << std::endl;
        std::cout << "Target tetrahedra: " << numTets << std::endl;
        std::cout << "Cut mesh surface triangles: " << numSurfaceTris << std::endl;
    }

    // ========== 三角形AABBデータ構造 ==========
    struct TriangleAABB {
        glm::vec3 min, max;
        glm::vec3 v0, v1, v2;
        glm::vec3 normal;
        int triIndex;
        bool isDummy;
    };

    std::vector<TriangleAABB> triangleAABBs;
    triangleAABBs.reserve(numSurfaceTris + 50);

    // ========== 表面三角形の処理とメッシュ重心計算 ==========
    glm::vec3 meshCenter(0.0f);
    glm::vec3 meshMin(FLT_MAX), meshMax(-FLT_MAX);

    for (size_t i = 0; i < numSurfaceTris; i++) {
        int idx0 = cutMesh.tetSurfaceTriIds[i * 3];
        int idx1 = cutMesh.tetSurfaceTriIds[i * 3 + 1];
        int idx2 = cutMesh.tetSurfaceTriIds[i * 3 + 2];

        glm::vec3 v0(cutMesh.verts[idx0 * 3],
                     cutMesh.verts[idx0 * 3 + 1],
                     cutMesh.verts[idx0 * 3 + 2]);
        glm::vec3 v1(cutMesh.verts[idx1 * 3],
                     cutMesh.verts[idx1 * 3 + 1],
                     cutMesh.verts[idx1 * 3 + 2]);
        glm::vec3 v2(cutMesh.verts[idx2 * 3],
                     cutMesh.verts[idx2 * 3 + 1],
                     cutMesh.verts[idx2 * 3 + 2]);

        meshCenter += v0 + v1 + v2;
        meshMin = glm::min(meshMin, glm::min(glm::min(v0, v1), v2));
        meshMax = glm::max(meshMax, glm::max(glm::max(v0, v1), v2));

        TriangleAABB triAABB;
        triAABB.v0 = v0;
        triAABB.v1 = v1;
        triAABB.v2 = v2;
        triAABB.min = glm::min(glm::min(v0, v1), v2);
        triAABB.max = glm::max(glm::max(v0, v1), v2);
        triAABB.normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
        triAABB.triIndex = i;
        triAABB.isDummy = false;
        triangleAABBs.push_back(triAABB);
    }

    meshCenter /= (float)(numSurfaceTris * 3);

    // ========== Mode 1, 2用: レイキャスティング関数 ==========
    auto rayTriangleIntersection = [](const glm::vec3& rayOrigin,
                                      const glm::vec3& rayDir,
                                      const glm::vec3& v0,
                                      const glm::vec3& v1,
                                      const glm::vec3& v2,
                                      float& outT) -> bool {
        const float EPSILON = 0.0000001f;
        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;
        glm::vec3 h = glm::cross(rayDir, edge2);
        float a = glm::dot(edge1, h);

        if (a > -EPSILON && a < EPSILON)
            return false;

        float f = 1.0f / a;
        glm::vec3 s = rayOrigin - v0;
        float u = f * glm::dot(s, h);

        if (u < 0.0f || u > 1.0f)
            return false;

        glm::vec3 q = glm::cross(s, edge1);
        float v = f * glm::dot(rayDir, q);

        if (v < 0.0f || u + v > 1.0f)
            return false;

        float t = f * glm::dot(edge2, q);
        outT = t;
        return t > EPSILON;
    };

    // 点がメッシュ内部にあるか判定（3方向レイキャスティング）
    auto isPointInsideMesh = [&](const glm::vec3& point) -> bool {
        // 高速化: メッシュAABB外なら即false
        if (point.x < meshMin.x || point.x > meshMax.x ||
            point.y < meshMin.y || point.y > meshMax.y ||
            point.z < meshMin.z || point.z > meshMax.z) {
            return false;
        }

        int insideCount = 0;

        // X軸方向
        {
            glm::vec3 rayDir(1.0f, 0.0f, 0.0f);
            int intersectionCount = 0;

            for (const auto& tri : triangleAABBs) {
                if (!tri.isDummy &&
                    point.y >= tri.min.y && point.y <= tri.max.y &&
                    point.z >= tri.min.z && point.z <= tri.max.z) {
                    float t;
                    if (rayTriangleIntersection(point, rayDir, tri.v0, tri.v1, tri.v2, t)) {
                        intersectionCount++;
                    }
                }
            }
            if ((intersectionCount % 2) == 1) insideCount++;
        }

        // Y軸方向
        {
            glm::vec3 rayDir(0.0f, 1.0f, 0.0f);
            int intersectionCount = 0;

            for (const auto& tri : triangleAABBs) {
                if (!tri.isDummy &&
                    point.x >= tri.min.x && point.x <= tri.max.x &&
                    point.z >= tri.min.z && point.z <= tri.max.z) {
                    float t;
                    if (rayTriangleIntersection(point, rayDir, tri.v0, tri.v1, tri.v2, t)) {
                        intersectionCount++;
                    }
                }
            }
            if ((intersectionCount % 2) == 1) insideCount++;
        }

        // Z軸方向
        {
            glm::vec3 rayDir(0.0f, 0.0f, 1.0f);
            int intersectionCount = 0;

            for (const auto& tri : triangleAABBs) {
                if (!tri.isDummy &&
                    point.x >= tri.min.x && point.x <= tri.max.x &&
                    point.y >= tri.min.y && point.y <= tri.max.y) {
                    float t;
                    if (rayTriangleIntersection(point, rayDir, tri.v0, tri.v1, tri.v2, t)) {
                        intersectionCount++;
                    }
                }
            }
            if ((intersectionCount % 2) == 1) insideCount++;
        }

        // 3方向のうち2方向以上で内部と判定
        return insideCount >= 2;
    };

    // ========== Mode 3: 内部ダミー三角形の生成 ==========
    if (intersectionMode == 3) {
        if (verbose) {
            std::cout << "Generating internal dummy triangles..." << std::endl;
        }

        // サンプリング設定
        const float SAMPLING_RATE = 0.5f;  // 5%
        const int MIN_SAMPLES = 8;
        const int MAX_SAMPLES = 1000;

        int numSamples = std::max(MIN_SAMPLES,
                                  std::min(MAX_SAMPLES,
                                           (int)(numSurfaceTris * SAMPLING_RATE)));

        // シンプルな疑似乱数生成器（randomヘッダー不要）
        unsigned int seed = 12345;
        auto simpleRand = [&seed]() -> unsigned int {
            seed = seed * 1103515245 + 12345;
            return (seed / 65536) % 32768;
        };

        int dummyCount = 0;
        int step = std::max(1, (int)numSurfaceTris / numSamples);

        // 均等サンプリング
        for (int i = 0; i < numSurfaceTris && dummyCount < numSamples; i += step) {
            int offset = simpleRand() % std::max(1, step / 2);
            int triIdx = (i + offset) % numSurfaceTris;

            const auto& surfaceTri = triangleAABBs[triIdx];

            // ランダムに辺を選択
            int edgeIdx = simpleRand() % 3;
            glm::vec3 edgeV0, edgeV1;

            switch (edgeIdx) {
            case 0: edgeV0 = surfaceTri.v0; edgeV1 = surfaceTri.v1; break;
            case 1: edgeV0 = surfaceTri.v1; edgeV1 = surfaceTri.v2; break;
            case 2: edgeV0 = surfaceTri.v2; edgeV1 = surfaceTri.v0; break;
            }

            // 辺と重心を結ぶ大きな三角形
            TriangleAABB dummyTri;
            dummyTri.v0 = meshCenter;
            dummyTri.v1 = edgeV0;
            dummyTri.v2 = edgeV1;
            dummyTri.min = glm::min(glm::min(dummyTri.v0, dummyTri.v1), dummyTri.v2);
            dummyTri.max = glm::max(glm::max(dummyTri.v0, dummyTri.v1), dummyTri.v2);

            glm::vec3 cross = glm::cross(dummyTri.v1 - dummyTri.v0,
                                         dummyTri.v2 - dummyTri.v0);
            float area = glm::length(cross) * 0.5f;

            if (area > 0.001f) {  // 退化三角形を避ける
                dummyTri.normal = glm::normalize(cross);
                dummyTri.triIndex = -1;
                dummyTri.isDummy = true;
                triangleAABBs.push_back(dummyTri);
                dummyCount++;
            }
        }

        if (verbose) {
            std::cout << "  Generated " << dummyCount << " dummy triangles" << std::endl;
        }
    }

    // ========== 交差判定の実行（最適化された並列処理） ==========
    std::vector<bool> tetIntersects(numTets, false);

    // OpenMP設定
    int numThreads = 1;
#ifdef _OPENMP
    numThreads = omp_get_max_threads();
    if (verbose) {
        std::cout << "Using " << numThreads << " threads for parallel processing" << std::endl;
    }
#endif

    // チャンクサイズを動的に決定（キャッシュ効率を考慮）
    const int chunk_size = std::max(1, (int)numTets / (numThreads * 10));

#pragma omp parallel for schedule(dynamic, chunk_size)
    for (int tetIdx = 0; tetIdx < numTets; tetIdx++) {
        // 四面体の頂点を効率的にロード
        glm::vec3 tetVerts[4];
        int baseIdx = tetIdx * 4;

        for (int j = 0; j < 4; j++) {
            int vid = tetMesh.tetIds[baseIdx + j];
            int vidBase = vid * 3;
            tetVerts[j] = glm::vec3(
                tetMesh.verts[vidBase],
                tetMesh.verts[vidBase + 1],
                tetMesh.verts[vidBase + 2]
                );
        }

        bool intersects = false;

        // ========== Mode 0: 表面交差のみ ==========
        if (intersectionMode == 0 || intersectionMode == 3) {
            // 四面体のAABBを事前計算
            glm::vec3 tetMin = tetVerts[0];
            glm::vec3 tetMax = tetVerts[0];
            for (int i = 1; i < 4; i++) {
                tetMin = glm::min(tetMin, tetVerts[i]);
                tetMax = glm::max(tetMax, tetVerts[i]);
            }

            // 各三角形との交差判定
            for (const auto& triAABB : triangleAABBs) {
                // Mode 0では表面のみ、Mode 3ではダミーも含む
                if (intersectionMode == 0 && triAABB.isDummy) {
                    continue;
                }

                // AABB高速フィルタリング
                if (tetMax.x < triAABB.min.x || tetMin.x > triAABB.max.x ||
                    tetMax.y < triAABB.min.y || tetMin.y > triAABB.max.y ||
                    tetMax.z < triAABB.min.z || tetMin.z > triAABB.max.z) {
                    continue;
                }

                // 1. 三角形の頂点が四面体内部（最も頻繁）
                if (TetrahedronIntersection::pointInTetrahedron(
                        triAABB.v0, tetVerts[0], tetVerts[1], tetVerts[2], tetVerts[3]) ||
                    TetrahedronIntersection::pointInTetrahedron(
                        triAABB.v1, tetVerts[0], tetVerts[1], tetVerts[2], tetVerts[3]) ||
                    TetrahedronIntersection::pointInTetrahedron(
                        triAABB.v2, tetVerts[0], tetVerts[1], tetVerts[2], tetVerts[3])) {
                    intersects = true;
                    break;  // 早期終了
                }

                // 2. 四面体のエッジと三角形の交差
                static const int edgePairs[6][2] = {
                    {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
                };

                bool edgeIntersects = false;
                for (int e = 0; e < 6 && !edgeIntersects; e++) {
                    if (TetrahedronIntersection::triangleSegmentIntersect(
                            triAABB.v0, triAABB.v1, triAABB.v2,
                            tetVerts[edgePairs[e][0]],
                            tetVerts[edgePairs[e][1]])) {
                        intersects = true;
                        edgeIntersects = true;
                    }
                }

                if (intersects) break;

                // 3. 四面体の面と三角形のエッジの交差
                static const int TET_FACE_INDICES[4][3] = {
                    {1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}
                };

                std::pair<glm::vec3, glm::vec3> triEdges[3] = {
                    {triAABB.v0, triAABB.v1},
                    {triAABB.v1, triAABB.v2},
                    {triAABB.v2, triAABB.v0}
                };

                for (int faceIdx = 0; faceIdx < 4 && !intersects; faceIdx++) {
                    glm::vec3 face[3] = {
                        tetVerts[TET_FACE_INDICES[faceIdx][0]],
                        tetVerts[TET_FACE_INDICES[faceIdx][1]],
                        tetVerts[TET_FACE_INDICES[faceIdx][2]]
                    };

                    for (int e = 0; e < 3; e++) {
                        if (TetrahedronIntersection::triangleSegmentIntersect(
                                face[0], face[1], face[2],
                                triEdges[e].first, triEdges[e].second)) {
                            intersects = true;
                            break;
                        }
                    }
                }

                if (intersects) break;
            }
        }
        // ========== Mode 1: 重心ベース ==========
        else if (intersectionMode == 1) {
            glm::vec3 centroid = (tetVerts[0] + tetVerts[1] + tetVerts[2] + tetVerts[3]) * 0.25f;
            intersects = isPointInsideMesh(centroid);
        }
        // ========== Mode 2: 全頂点内部 ==========
        else if (intersectionMode == 2) {
            bool allInside = true;
            for (int j = 0; j < 4; j++) {
                if (!isPointInsideMesh(tetVerts[j])) {
                    allInside = false;
                    break;
                }
            }
            intersects = allInside;
        }

        if (intersects) {
            tetIntersects[tetIdx] = true;
        }
    }

    // ========== 結果の収集（効率的な方法） ==========
    intersectingTets.reserve(numTets / 10);  // 予想サイズで予約

    for (size_t i = 0; i < numTets; i++) {
        if (tetIntersects[i]) {
            intersectingTets.push_back(i);
        }
    }

    // ========== 統計出力 ==========
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endTime - startTime;

    if (verbose) {
        std::cout << "\nResults:" << std::endl;
        std::cout << "  Method: " << modeNames[intersectionMode] << std::endl;
        std::cout << "  Intersecting tetrahedra: " << intersectingTets.size()
                  << " (" << (float)intersectingTets.size() / numTets * 100.0f << "%)" << std::endl;

        if (intersectionMode == 3) {
            std::cout << "  Total triangles processed: " << triangleAABBs.size()
                      << " (surface: " << numSurfaceTris
                      << ", dummy: " << (triangleAABBs.size() - numSurfaceTris) << ")" << std::endl;
        }

        std::cout << "  Time: " << duration.count() * 1000 << " ms" << std::endl;

        double tetsPerSec = numTets / duration.count();
        std::cout << "  Performance: " << tetsPerSec << " tets/sec" << std::endl;
    }

    return intersectingTets;
}


// ★★★ main.cppの上部で定義：高速版連結成分解析関数 ★★★
std::vector<std::vector<int>> findConnectedComponents(
    const SoftBodyGPUDuo::MeshData& meshData,
    const std::set<int>& validTetIndices)
{
    std::vector<std::vector<int>> components;
    std::set<int> visited;

    // 面をキーとして、その面を持つ四面体をマッピング
    std::map<std::array<int, 3>, std::vector<int>> faceToTets;
    std::map<int, std::set<int>> tetNeighbors;

    std::cout << "  Building face map for " << validTetIndices.size() << " valid tets..." << std::endl;

    // ステップ1: 各四面体の面を登録（O(n)）
    for (int tetIdx : validTetIndices) {
        // この四面体の4つの面
        std::array<std::array<int, 3>, 4> faces = {{
            {meshData.tetIds[tetIdx*4+0], meshData.tetIds[tetIdx*4+1], meshData.tetIds[tetIdx*4+2]},
            {meshData.tetIds[tetIdx*4+0], meshData.tetIds[tetIdx*4+1], meshData.tetIds[tetIdx*4+3]},
            {meshData.tetIds[tetIdx*4+0], meshData.tetIds[tetIdx*4+2], meshData.tetIds[tetIdx*4+3]},
            {meshData.tetIds[tetIdx*4+1], meshData.tetIds[tetIdx*4+2], meshData.tetIds[tetIdx*4+3]}
        }};

        for (auto& face : faces) {
            // 面を正規化（ソート）してキーとして使用
            std::sort(face.begin(), face.end());
            faceToTets[face].push_back(tetIdx);
        }
    }

    std::cout << "  Face map built with " << faceToTets.size() << " unique faces" << std::endl;
    std::cout << "  Building adjacency map..." << std::endl;

    // ステップ2: 面を共有する四面体同士を隣接として登録（O(n)）
    for (const auto& pair : faceToTets) {
        const std::vector<int>& tetsWithThisFace = pair.second;

        // この面を共有する四面体は互いに隣接
        if (tetsWithThisFace.size() == 2) {
            int tet1 = tetsWithThisFace[0];
            int tet2 = tetsWithThisFace[1];
            tetNeighbors[tet1].insert(tet2);
            tetNeighbors[tet2].insert(tet1);
        }
    }

    std::cout << "  Adjacency map built. Running BFS..." << std::endl;

    // ステップ3: BFSで連結成分を検出（O(n)）
    for (int startTet : validTetIndices) {
        if (visited.count(startTet) > 0) continue;

        std::vector<int> component;
        std::queue<int> queue;
        queue.push(startTet);
        visited.insert(startTet);

        while (!queue.empty()) {
            int currentTet = queue.front();
            queue.pop();
            component.push_back(currentTet);

            // 隣接する四面体を探索
            if (tetNeighbors.count(currentTet) > 0) {
                for (int neighbor : tetNeighbors[currentTet]) {
                    if (visited.count(neighbor) == 0) {
                        visited.insert(neighbor);
                        queue.push(neighbor);
                    }
                }
            }
        }

        components.push_back(component);
    }

    std::cout << "  Found " << components.size() << " connected components:" << std::endl;

    // サイズ順にソートして表示
    std::vector<size_t> componentSizes;
    for (const auto& comp : components) {
        componentSizes.push_back(comp.size());
    }
    std::sort(componentSizes.rbegin(), componentSizes.rend());

    for (size_t i = 0; i < std::min(size_t(10), componentSizes.size()); i++) {
        std::cout << "    Component " << i << ": " << componentSizes[i] << " tets" << std::endl;
    }
    if (componentSizes.size() > 10) {
        std::cout << "    ... and " << (componentSizes.size() - 10) << " more components" << std::endl;
    }

    return components;
}



//==============================================================================
// MeshCuttingGPUDuo BVH拡張
//
// このコードをMeshCuttingGPUDuo.hの末尾（#endif の前）に追加してください
// 既存のfindIntersectingTetrahedraBySurfaceTriangles()は変更不要です
//
// 使用方法:
//   // Liver等の大規模メッシュ用
//   MeshCuttingGPUDuo::buildTetBVH(target->highRes_positions,
//                                   target->highResMeshData.tetIds);
//   auto result = MeshCuttingGPUDuo::findIntersectingTetrahedraBySurfaceTriangles_BVH(
//       cutterData, highResData, target->highResTetValid, 3, true);
//
//   // Portal等の小規模メッシュ用（従来版）
//   auto result = MeshCuttingGPUDuo::findIntersectingTetrahedraBySurfaceTriangles(
//       cutterData, highResData, 3, true);
//==============================================================================

//==============================================================================
// BVH (Bounding Volume Hierarchy) 実装
//==============================================================================
namespace BVH {

struct AABB {
    glm::vec3 min;
    glm::vec3 max;

    AABB() : min(FLT_MAX), max(-FLT_MAX) {}
    AABB(const glm::vec3& minV, const glm::vec3& maxV) : min(minV), max(maxV) {}

    void expand(const glm::vec3& point) {
        min = glm::min(min, point);
        max = glm::max(max, point);
    }

    void expand(const AABB& other) {
        min = glm::min(min, other.min);
        max = glm::max(max, other.max);
    }

    bool intersects(const AABB& other) const {
        return (min.x <= other.max.x && max.x >= other.min.x) &&
               (min.y <= other.max.y && max.y >= other.min.y) &&
               (min.z <= other.max.z && max.z >= other.min.z);
    }
};

struct Node {
    AABB bounds;
    int left = -1;
    int right = -1;
    int tetStart = -1;
    int tetCount = 0;

    bool isLeaf() const { return left == -1; }
};

// グローバルBVHデータ
static std::vector<Node> g_nodes;
static std::vector<int> g_tetIndices;
static std::vector<AABB> g_tetAABBs;
static bool g_isBuilt = false;

inline bool isBuilt() { return g_isBuilt; }

inline void clear() {
    g_nodes.clear();
    g_tetIndices.clear();
    g_tetAABBs.clear();
    g_isBuilt = false;
}

inline int buildRecursive(int start, int end) {
    Node node;

    node.bounds = AABB();
    for (int i = start; i < end; i++) {
        node.bounds.expand(g_tetAABBs[g_tetIndices[i]]);
    }

    int count = end - start;
    const int LEAF_THRESHOLD = 8;

    if (count <= LEAF_THRESHOLD) {
        node.tetStart = start;
        node.tetCount = count;
        node.left = -1;
        node.right = -1;
        g_nodes.push_back(node);
        return (int)g_nodes.size() - 1;
    }

    glm::vec3 extent = node.bounds.max - node.bounds.min;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent[axis]) axis = 2;

    float mid = (node.bounds.min[axis] + node.bounds.max[axis]) * 0.5f;

    auto midIt = std::partition(g_tetIndices.begin() + start, g_tetIndices.begin() + end,
                                [axis, mid](int idx) {
                                    float center = (g_tetAABBs[idx].min[axis] + g_tetAABBs[idx].max[axis]) * 0.5f;
                                    return center < mid;
                                });

    int midIdx = (int)(midIt - g_tetIndices.begin());

    if (midIdx == start || midIdx == end) {
        midIdx = (start + end) / 2;
    }

    int nodeIdx = (int)g_nodes.size();
    g_nodes.push_back(node);

    g_nodes[nodeIdx].left = buildRecursive(start, midIdx);
    g_nodes[nodeIdx].right = buildRecursive(midIdx, end);

    return nodeIdx;
}

inline void build(const std::vector<float>& positions,
                  const std::vector<int>& tetIds,
                  size_t numTets,
                  bool verbose = true) {
    auto startTime = std::chrono::high_resolution_clock::now();

    clear();

    g_tetAABBs.resize(numTets);
    g_tetIndices.resize(numTets);

#pragma omp parallel for
    for (int tetIdx = 0; tetIdx < (int)numTets; tetIdx++) {
        AABB aabb;
        int baseIdx = tetIdx * 4;

        for (int j = 0; j < 4; j++) {
            int vid = tetIds[baseIdx + j];
            glm::vec3 v(positions[vid * 3],
                        positions[vid * 3 + 1],
                        positions[vid * 3 + 2]);
            aabb.expand(v);
        }

        g_tetAABBs[tetIdx] = aabb;
        g_tetIndices[tetIdx] = tetIdx;
    }

    g_nodes.reserve(numTets * 2);
    buildRecursive(0, (int)numTets);

    g_isBuilt = true;

    auto endTime = std::chrono::high_resolution_clock::now();
    double buildTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    if (verbose) {
        std::cout << "\n[BVH] Built for " << numTets << " tetrahedra" << std::endl;
        std::cout << "  Nodes: " << g_nodes.size() << std::endl;
        std::cout << "  Build time: " << buildTime << " ms" << std::endl;
    }
}

inline void queryRecursive(int nodeIdx, const AABB& queryAABB,
                           const std::vector<bool>& tetValid,
                           std::vector<int>& candidates) {
    if (nodeIdx < 0 || nodeIdx >= (int)g_nodes.size()) return;

    const Node& node = g_nodes[nodeIdx];

    if (!node.bounds.intersects(queryAABB)) {
        return;
    }

    if (node.isLeaf()) {
        for (int i = 0; i < node.tetCount; i++) {
            int tetIdx = g_tetIndices[node.tetStart + i];
            if (tetValid.empty() || tetValid[tetIdx]) {
                candidates.push_back(tetIdx);
            }
        }
    } else {
        queryRecursive(node.left, queryAABB, tetValid, candidates);
        queryRecursive(node.right, queryAABB, tetValid, candidates);
    }
}

inline void query(const AABB& queryAABB,
                  const std::vector<bool>& tetValid,
                  std::vector<int>& candidates) {
    if (!g_isBuilt || g_nodes.empty()) return;
    queryRecursive(0, queryAABB, tetValid, candidates);
}

} // namespace BVH


//==============================================================================
// BVH構築用ヘルパー関数
//==============================================================================
inline void buildTetBVH(const std::vector<float>& positions,
                        const std::vector<int>& tetIds,
                        bool verbose = true) {
    size_t numTets = tetIds.size() / 4;
    BVH::build(positions, tetIds, numTets, verbose);
}

inline void buildTetBVH(const SoftBodyGPUDuo* softBody, bool verbose = true) {
    if (!softBody) return;
    buildTetBVH(softBody->highRes_positions,
                softBody->highResMeshData.tetIds,
                verbose);
}

inline void clearTetBVH() {
    BVH::clear();
}

inline bool isTetBVHBuilt() {
    return BVH::isBuilt();
}


//==============================================================================
// BVH版 交差判定関数（並列化）- Mode 0, 3 専用
//==============================================================================
std::vector<int> findIntersectingTetrahedraBySurfaceTriangles_BVH(
    const SoftBodyGPUDuo::MeshData& cutMesh,
    const SoftBodyGPUDuo::MeshData& tetMesh,
    const std::vector<bool>& tetValid,
    int intersectionMode = 3,  // 0 または 3 のみ対応
    bool verbose = true) {

    auto startTime = std::chrono::high_resolution_clock::now();

    std::vector<int> intersectingTets;
    size_t numTets = tetMesh.tetIds.size() / 4;
    size_t numSurfaceTris = cutMesh.tetSurfaceTriIds.size() / 3;

    // Mode 0, 3 以外はエラー
    if (intersectionMode != 0 && intersectionMode != 3) {
        std::cerr << "[BVH] Error: Only Mode 0 and 3 are supported. Use original function for Mode 1, 2." << std::endl;
        return intersectingTets;
    }

    // BVH未構築はエラー
    if (!BVH::isBuilt()) {
        std::cerr << "[BVH] Error: BVH not built. Call buildTetBVH() first." << std::endl;
        return intersectingTets;
    }

    const char* modeNames[] = {
        "SURFACE-ONLY (BVH)",
        "",
        "",
        "SURFACE + SAMPLED-INTERNAL (BVH)"
    };

    if (verbose) {
        std::cout << "\n=== BVH Intersection Detection ===" << std::endl;
        std::cout << "Method: " << modeNames[intersectionMode] << std::endl;
        std::cout << "Target tetrahedra: " << numTets << std::endl;
        std::cout << "Cut mesh surface triangles: " << numSurfaceTris << std::endl;
    }

    // ========== 三角形AABBデータ構造 ==========
    struct TriangleAABB {
        glm::vec3 min, max;
        glm::vec3 v0, v1, v2;
        int triIndex;
        bool isDummy;
    };

    std::vector<TriangleAABB> triangleAABBs;
    triangleAABBs.reserve(numSurfaceTris + 100);

    // ========== 表面三角形の処理 ==========
    glm::vec3 meshCenter(0.0f);

    for (size_t i = 0; i < numSurfaceTris; i++) {
        int idx0 = cutMesh.tetSurfaceTriIds[i * 3];
        int idx1 = cutMesh.tetSurfaceTriIds[i * 3 + 1];
        int idx2 = cutMesh.tetSurfaceTriIds[i * 3 + 2];

        glm::vec3 v0(cutMesh.verts[idx0 * 3], cutMesh.verts[idx0 * 3 + 1], cutMesh.verts[idx0 * 3 + 2]);
        glm::vec3 v1(cutMesh.verts[idx1 * 3], cutMesh.verts[idx1 * 3 + 1], cutMesh.verts[idx1 * 3 + 2]);
        glm::vec3 v2(cutMesh.verts[idx2 * 3], cutMesh.verts[idx2 * 3 + 1], cutMesh.verts[idx2 * 3 + 2]);

        meshCenter += v0 + v1 + v2;

        TriangleAABB triAABB;
        triAABB.v0 = v0;
        triAABB.v1 = v1;
        triAABB.v2 = v2;
        triAABB.min = glm::min(glm::min(v0, v1), v2);
        triAABB.max = glm::max(glm::max(v0, v1), v2);
        triAABB.triIndex = (int)i;
        triAABB.isDummy = false;
        triangleAABBs.push_back(triAABB);
    }

    meshCenter /= (float)(numSurfaceTris * 3);

    // ========== Mode 3: ダミー三角形生成 ==========
    if (intersectionMode == 3) {
        const float SAMPLING_RATE = 0.5f;
        const int MIN_SAMPLES = 8;
        const int MAX_SAMPLES = 1000;

        int numSamples = std::max(MIN_SAMPLES, std::min(MAX_SAMPLES, (int)(numSurfaceTris * SAMPLING_RATE)));

        unsigned int seed = 12345;
        auto simpleRand = [&seed]() -> unsigned int {
            seed = seed * 1103515245 + 12345;
            return (seed / 65536) % 32768;
        };

        int dummyCount = 0;
        int step = std::max(1, (int)numSurfaceTris / numSamples);

        for (size_t i = 0; i < numSurfaceTris && dummyCount < numSamples; i += step) {
            int offset = simpleRand() % std::max(1, step / 2);
            int triIdx = ((int)i + offset) % (int)numSurfaceTris;

            const auto& surfaceTri = triangleAABBs[triIdx];

            int edgeIdx = simpleRand() % 3;
            glm::vec3 edgeV0, edgeV1;

            switch (edgeIdx) {
            case 0: edgeV0 = surfaceTri.v0; edgeV1 = surfaceTri.v1; break;
            case 1: edgeV0 = surfaceTri.v1; edgeV1 = surfaceTri.v2; break;
            case 2: edgeV0 = surfaceTri.v2; edgeV1 = surfaceTri.v0; break;
            }

            TriangleAABB dummyTri;
            dummyTri.v0 = meshCenter;
            dummyTri.v1 = edgeV0;
            dummyTri.v2 = edgeV1;
            dummyTri.min = glm::min(glm::min(dummyTri.v0, dummyTri.v1), dummyTri.v2);
            dummyTri.max = glm::max(glm::max(dummyTri.v0, dummyTri.v1), dummyTri.v2);

            glm::vec3 cross = glm::cross(dummyTri.v1 - dummyTri.v0, dummyTri.v2 - dummyTri.v0);
            float area = glm::length(cross) * 0.5f;

            if (area > 0.001f) {
                dummyTri.triIndex = -1;
                dummyTri.isDummy = true;
                triangleAABBs.push_back(dummyTri);
                dummyCount++;
            }
        }

        if (verbose) {
            std::cout << "Generated " << dummyCount << " dummy triangles" << std::endl;
        }
    }

    // ========== 並列BVH交差判定 ==========
    int numThreads = 1;
#ifdef _OPENMP
    numThreads = omp_get_max_threads();
#endif

    if (verbose) {
        std::cout << "Using " << numThreads << " threads" << std::endl;
    }

    std::vector<std::set<int>> threadResults(numThreads);
    long long totalCandidates = 0;
    long long totalDetailedTests = 0;

#pragma omp parallel reduction(+:totalCandidates, totalDetailedTests)
    {
        int threadId = 0;
#ifdef _OPENMP
        threadId = omp_get_thread_num();
#endif
        std::set<int>& localSet = threadResults[threadId];
        std::vector<int> candidates;
        candidates.reserve(256);

#pragma omp for schedule(dynamic, 16)
        for (int triIdx = 0; triIdx < (int)triangleAABBs.size(); triIdx++) {
            const auto& triAABB = triangleAABBs[triIdx];

            if (intersectionMode == 0 && triAABB.isDummy) continue;

            BVH::AABB queryAABB(triAABB.min, triAABB.max);

            candidates.clear();
            BVH::query(queryAABB, tetValid, candidates);

            totalCandidates += candidates.size();

            for (int tetIdx : candidates) {
                if (localSet.count(tetIdx) > 0) continue;

                totalDetailedTests++;

                glm::vec3 tetVerts[4];
                int baseIdx = tetIdx * 4;
                for (int j = 0; j < 4; j++) {
                    int vid = tetMesh.tetIds[baseIdx + j];
                    tetVerts[j] = glm::vec3(
                        tetMesh.verts[vid * 3],
                        tetMesh.verts[vid * 3 + 1],
                        tetMesh.verts[vid * 3 + 2]
                        );
                }

                bool intersects = false;

                // 1. 三角形の頂点が四面体内部
                if (MeshCuttingGPUDuo::TetrahedronIntersection::pointInTetrahedron(
                        triAABB.v0, tetVerts[0], tetVerts[1], tetVerts[2], tetVerts[3]) ||
                    MeshCuttingGPUDuo::TetrahedronIntersection::pointInTetrahedron(
                        triAABB.v1, tetVerts[0], tetVerts[1], tetVerts[2], tetVerts[3]) ||
                    MeshCuttingGPUDuo::TetrahedronIntersection::pointInTetrahedron(
                        triAABB.v2, tetVerts[0], tetVerts[1], tetVerts[2], tetVerts[3])) {
                    intersects = true;
                }

                // 2. 四面体のエッジと三角形の交差
                if (!intersects) {
                    static const int edgePairs[6][2] = {
                        {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
                    };

                    for (int e = 0; e < 6 && !intersects; e++) {
                        if (MeshCuttingGPUDuo::TetrahedronIntersection::triangleSegmentIntersect(
                                triAABB.v0, triAABB.v1, triAABB.v2,
                                tetVerts[edgePairs[e][0]],
                                tetVerts[edgePairs[e][1]])) {
                            intersects = true;
                        }
                    }
                }

                // 3. 四面体の面と三角形のエッジの交差
                if (!intersects) {
                    static const int TET_FACE_INDICES[4][3] = {
                        {1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}
                    };

                    std::pair<glm::vec3, glm::vec3> triEdges[3] = {
                        {triAABB.v0, triAABB.v1},
                        {triAABB.v1, triAABB.v2},
                        {triAABB.v2, triAABB.v0}
                    };

                    for (int faceIdx = 0; faceIdx < 4 && !intersects; faceIdx++) {
                        glm::vec3 face[3] = {
                            tetVerts[TET_FACE_INDICES[faceIdx][0]],
                            tetVerts[TET_FACE_INDICES[faceIdx][1]],
                            tetVerts[TET_FACE_INDICES[faceIdx][2]]
                        };

                        for (int e = 0; e < 3; e++) {
                            if (MeshCuttingGPUDuo::TetrahedronIntersection::triangleSegmentIntersect(
                                    face[0], face[1], face[2],
                                    triEdges[e].first, triEdges[e].second)) {
                                intersects = true;
                                break;
                            }
                        }
                    }
                }

                if (intersects) {
                    localSet.insert(tetIdx);
                }
            }
        }
    }

    // ========== 結果マージ ==========
    std::set<int> intersectingTetSet;
    for (const auto& localSet : threadResults) {
        intersectingTetSet.insert(localSet.begin(), localSet.end());
    }

    intersectingTets.assign(intersectingTetSet.begin(), intersectingTetSet.end());

    // ========== 統計出力 ==========
    auto endTime = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    if (verbose) {
        long long bruteForceTests = (long long)numTets * (long long)triangleAABBs.size();
        float speedup = (totalDetailedTests > 0) ? (float)bruteForceTests / totalDetailedTests : 0;

        std::cout << "\n[BVH Performance]" << std::endl;
        std::cout << "  BVH candidates: " << totalCandidates << std::endl;
        std::cout << "  Detailed tests: " << totalDetailedTests << std::endl;
        std::cout << "  Brute force would be: " << bruteForceTests << std::endl;
        std::cout << "  Theoretical speedup: " << speedup << "x" << std::endl;
        std::cout << "\nResults:" << std::endl;
        std::cout << "  Intersecting tetrahedra: " << intersectingTets.size()
                  << " (" << (float)intersectingTets.size() / numTets * 100.0f << "%)" << std::endl;
        std::cout << "  Time: " << duration << " ms" << std::endl;
    }

    return intersectingTets;
}



inline std::vector<int> findIntersectingTetrahedraBySurfaceTriangles_AdjProp(
    const SoftBodyGPUDuo::MeshData& cutMesh,
    const SoftBodyGPUDuo::MeshData& tetMesh,
    const std::vector<int>& surfaceTriToTet,           // 表面三角形→四面体マッピング
    const std::vector<float>& currentPositions,        // 現在の頂点位置（変形後）
    const std::vector<bool>& tetValid,
    const std::vector<std::vector<int>>& tetAdjacency, // 外部から渡される隣接関係
    int intersectionMode = 3,
    bool verbose = true) {

    auto startTime = std::chrono::high_resolution_clock::now();

    std::vector<int> intersectingTets;
    size_t numTets = tetMesh.tetIds.size() / 4;
    size_t numSurfaceTris = cutMesh.tetSurfaceTriIds.size() / 3;

    // 隣接関係が空の場合はエラー
    if (tetAdjacency.empty()) {
        std::cerr << "[AdjProp] Error: Adjacency is empty. Call SoftBody->buildHighResTetAdjacency() first." << std::endl;
        return intersectingTets;
    }

    if (verbose) {
        std::cout << "\n=== Adjacency Propagation Intersection Detection ===" << std::endl;
        std::cout << "Target tetrahedra: " << numTets << std::endl;
        std::cout << "Cut mesh surface triangles: " << numSurfaceTris << std::endl;
    }

    // ========== 三角形AABBデータ構造 ==========
    struct TriangleAABB {
        glm::vec3 min, max;
        glm::vec3 v0, v1, v2;
        int triIndex;
        bool isDummy;
    };

    std::vector<TriangleAABB> triangleAABBs;
    triangleAABBs.reserve(numSurfaceTris + 100);

    // ========== カッターのAABB計算と表面三角形処理 ==========
    glm::vec3 cutterMin(FLT_MAX), cutterMax(-FLT_MAX);
    glm::vec3 meshCenter(0.0f);

    for (size_t i = 0; i < numSurfaceTris; i++) {
        int idx0 = cutMesh.tetSurfaceTriIds[i * 3];
        int idx1 = cutMesh.tetSurfaceTriIds[i * 3 + 1];
        int idx2 = cutMesh.tetSurfaceTriIds[i * 3 + 2];

        glm::vec3 v0(cutMesh.verts[idx0 * 3], cutMesh.verts[idx0 * 3 + 1], cutMesh.verts[idx0 * 3 + 2]);
        glm::vec3 v1(cutMesh.verts[idx1 * 3], cutMesh.verts[idx1 * 3 + 1], cutMesh.verts[idx1 * 3 + 2]);
        glm::vec3 v2(cutMesh.verts[idx2 * 3], cutMesh.verts[idx2 * 3 + 1], cutMesh.verts[idx2 * 3 + 2]);

        meshCenter += v0 + v1 + v2;

        // カッター全体のAABB更新
        cutterMin = glm::min(cutterMin, glm::min(glm::min(v0, v1), v2));
        cutterMax = glm::max(cutterMax, glm::max(glm::max(v0, v1), v2));

        TriangleAABB triAABB;
        triAABB.v0 = v0;
        triAABB.v1 = v1;
        triAABB.v2 = v2;
        triAABB.min = glm::min(glm::min(v0, v1), v2);
        triAABB.max = glm::max(glm::max(v0, v1), v2);
        triAABB.triIndex = (int)i;
        triAABB.isDummy = false;
        triangleAABBs.push_back(triAABB);
    }

    meshCenter /= (float)(numSurfaceTris * 3);

    // ========== Mode 3: ダミー三角形生成 ==========
    if (intersectionMode == 3) {
        const float SAMPLING_RATE = 0.5f;
        const int MIN_SAMPLES = 8;
        const int MAX_SAMPLES = 1000;

        int numSamples = std::max(MIN_SAMPLES, std::min(MAX_SAMPLES, (int)(numSurfaceTris * SAMPLING_RATE)));

        unsigned int seed = 12345;
        auto simpleRand = [&seed]() -> unsigned int {
            seed = seed * 1103515245 + 12345;
            return (seed / 65536) % 32768;
        };

        int dummyCount = 0;
        int step = std::max(1, (int)numSurfaceTris / numSamples);

        for (size_t i = 0; i < numSurfaceTris && dummyCount < numSamples; i += step) {
            int offset = simpleRand() % std::max(1, step / 2);
            int triIdx = ((int)i + offset) % (int)numSurfaceTris;

            const auto& surfaceTri = triangleAABBs[triIdx];

            int edgeIdx = simpleRand() % 3;
            glm::vec3 edgeV0, edgeV1;

            switch (edgeIdx) {
            case 0: edgeV0 = surfaceTri.v0; edgeV1 = surfaceTri.v1; break;
            case 1: edgeV0 = surfaceTri.v1; edgeV1 = surfaceTri.v2; break;
            case 2: edgeV0 = surfaceTri.v2; edgeV1 = surfaceTri.v0; break;
            }

            TriangleAABB dummyTri;
            dummyTri.v0 = meshCenter;
            dummyTri.v1 = edgeV0;
            dummyTri.v2 = edgeV1;
            dummyTri.min = glm::min(glm::min(dummyTri.v0, dummyTri.v1), dummyTri.v2);
            dummyTri.max = glm::max(glm::max(dummyTri.v0, dummyTri.v1), dummyTri.v2);

            glm::vec3 cross = glm::cross(dummyTri.v1 - dummyTri.v0, dummyTri.v2 - dummyTri.v0);
            float area = glm::length(cross) * 0.5f;

            if (area > 0.001f) {
                dummyTri.triIndex = -1;
                dummyTri.isDummy = true;
                triangleAABBs.push_back(dummyTri);
                dummyCount++;
            }
        }

        if (verbose) {
            std::cout << "Generated " << dummyCount << " dummy triangles" << std::endl;
        }
    }

    // ========== Phase 1: シード四面体の取得 ==========
    auto phase1Start = std::chrono::high_resolution_clock::now();

    std::set<int> seedTetSet;
    size_t numTargetSurfaceTris = tetMesh.tetSurfaceTriIds.size() / 3;

    for (size_t i = 0; i < numTargetSurfaceTris; i++) {
        // 表面三角形のAABBを計算（現在位置で）
        int idx0 = tetMesh.tetSurfaceTriIds[i * 3];
        int idx1 = tetMesh.tetSurfaceTriIds[i * 3 + 1];
        int idx2 = tetMesh.tetSurfaceTriIds[i * 3 + 2];

        glm::vec3 tv0(currentPositions[idx0 * 3], currentPositions[idx0 * 3 + 1], currentPositions[idx0 * 3 + 2]);
        glm::vec3 tv1(currentPositions[idx1 * 3], currentPositions[idx1 * 3 + 1], currentPositions[idx1 * 3 + 2]);
        glm::vec3 tv2(currentPositions[idx2 * 3], currentPositions[idx2 * 3 + 1], currentPositions[idx2 * 3 + 2]);

        glm::vec3 triMin = glm::min(glm::min(tv0, tv1), tv2);
        glm::vec3 triMax = glm::max(glm::max(tv0, tv1), tv2);

        // カッターAABBと交差チェック
        if (triMax.x < cutterMin.x || triMin.x > cutterMax.x ||
            triMax.y < cutterMin.y || triMin.y > cutterMax.y ||
            triMax.z < cutterMin.z || triMin.z > cutterMax.z) {
            continue;
        }

        // この表面三角形に対応する四面体をシードに追加
        if (i < surfaceTriToTet.size()) {
            int tetIdx = surfaceTriToTet[i];
            if (tetIdx >= 0 && (tetValid.empty() || tetValid[tetIdx])) {
                seedTetSet.insert(tetIdx);
            }
        }
    }

    std::vector<int> seedTets(seedTetSet.begin(), seedTetSet.end());

    auto phase1End = std::chrono::high_resolution_clock::now();
    double phase1Time = std::chrono::duration<double, std::milli>(phase1End - phase1Start).count();

    if (verbose) {
        std::cout << "Phase 1 (seed detection): " << seedTets.size() << " seeds, " << phase1Time << " ms" << std::endl;
    }

    if (seedTets.empty()) {
        if (verbose) {
            std::cout << "No seeds found - no surface intersection" << std::endl;
        }
        return intersectingTets;
    }

    // ========== Phase 2: BFS伝播で候補取得 ==========
    auto phase2Start = std::chrono::high_resolution_clock::now();

    // BFS伝播探索（ローカル実装）
    std::vector<bool> visited(tetAdjacency.size(), false);
    std::queue<int> queue;
    std::vector<int> candidates;
    candidates.reserve(seedTets.size() * 10);

    // シードをキューに追加
    for (int seed : seedTets) {
        if (seed >= 0 && seed < (int)tetAdjacency.size()) {
            if (tetValid.empty() || tetValid[seed]) {
                queue.push(seed);
                visited[seed] = true;
            }
        }
    }

    while (!queue.empty()) {
        int tetIdx = queue.front();
        queue.pop();

        // 四面体のAABBを計算
        glm::vec3 tetMin(FLT_MAX), tetMax(-FLT_MAX);
        int baseIdx = tetIdx * 4;
        for (int j = 0; j < 4; j++) {
            int vid = tetMesh.tetIds[baseIdx + j];
            glm::vec3 v(currentPositions[vid*3], currentPositions[vid*3+1], currentPositions[vid*3+2]);
            tetMin = glm::min(tetMin, v);
            tetMax = glm::max(tetMax, v);
        }

        // カッターAABBと交差しなければスキップ（伝播停止）
        if (tetMax.x < cutterMin.x || tetMin.x > cutterMax.x ||
            tetMax.y < cutterMin.y || tetMin.y > cutterMax.y ||
            tetMax.z < cutterMin.z || tetMin.z > cutterMax.z) {
            continue;
        }

        // 候補に追加
        candidates.push_back(tetIdx);

        // 隣接四面体を探索
        for (int neighbor : tetAdjacency[tetIdx]) {
            if (!visited[neighbor]) {
                if (tetValid.empty() || tetValid[neighbor]) {
                    visited[neighbor] = true;
                    queue.push(neighbor);
                }
            }
        }
    }

    auto phase2End = std::chrono::high_resolution_clock::now();
    double phase2Time = std::chrono::duration<double, std::milli>(phase2End - phase2Start).count();

    if (verbose) {
        std::cout << "Phase 2 (BFS propagation): " << candidates.size() << " candidates, " << phase2Time << " ms" << std::endl;
    }

    // ========== Phase 3: 詳細交差判定（並列化）==========
    auto phase3Start = std::chrono::high_resolution_clock::now();

    int numThreads = 1;
#ifdef _OPENMP
    numThreads = omp_get_max_threads();
#endif

    std::vector<std::set<int>> threadResults(numThreads);

#pragma omp parallel
    {
        int threadId = 0;
#ifdef _OPENMP
        threadId = omp_get_thread_num();
#endif
        std::set<int>& localSet = threadResults[threadId];

#pragma omp for schedule(dynamic, 32)
        for (int candIdx = 0; candIdx < (int)candidates.size(); candIdx++) {
            int tetIdx = candidates[candIdx];

            // 四面体の頂点を取得（現在位置）
            glm::vec3 tetVerts[4];
            int baseIdx = tetIdx * 4;
            for (int j = 0; j < 4; j++) {
                int vid = tetMesh.tetIds[baseIdx + j];
                tetVerts[j] = glm::vec3(
                    currentPositions[vid * 3],
                    currentPositions[vid * 3 + 1],
                    currentPositions[vid * 3 + 2]
                    );
            }

            // 四面体のAABB
            glm::vec3 tetMin = tetVerts[0], tetMax = tetVerts[0];
            for (int j = 1; j < 4; j++) {
                tetMin = glm::min(tetMin, tetVerts[j]);
                tetMax = glm::max(tetMax, tetVerts[j]);
            }

            bool intersects = false;

            // 各カッター三角形との交差判定
            for (const auto& triAABB : triangleAABBs) {
                if (intersectionMode == 0 && triAABB.isDummy) continue;

                // AABBフィルタリング
                if (tetMax.x < triAABB.min.x || tetMin.x > triAABB.max.x ||
                    tetMax.y < triAABB.min.y || tetMin.y > triAABB.max.y ||
                    tetMax.z < triAABB.min.z || tetMin.z > triAABB.max.z) {
                    continue;
                }

                // 1. 三角形の頂点が四面体内部
                if (TetrahedronIntersection::pointInTetrahedron(
                        triAABB.v0, tetVerts[0], tetVerts[1], tetVerts[2], tetVerts[3]) ||
                    TetrahedronIntersection::pointInTetrahedron(
                        triAABB.v1, tetVerts[0], tetVerts[1], tetVerts[2], tetVerts[3]) ||
                    TetrahedronIntersection::pointInTetrahedron(
                        triAABB.v2, tetVerts[0], tetVerts[1], tetVerts[2], tetVerts[3])) {
                    intersects = true;
                    break;
                }

                // 2. 四面体のエッジと三角形の交差
                static const int edgePairs[6][2] = {
                    {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
                };

                for (int e = 0; e < 6 && !intersects; e++) {
                    if (TetrahedronIntersection::triangleSegmentIntersect(
                            triAABB.v0, triAABB.v1, triAABB.v2,
                            tetVerts[edgePairs[e][0]],
                            tetVerts[edgePairs[e][1]])) {
                        intersects = true;
                    }
                }

                if (intersects) break;

                // 3. 四面体の面と三角形のエッジの交差
                static const int TET_FACE_INDICES[4][3] = {
                    {1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}
                };

                std::pair<glm::vec3, glm::vec3> triEdges[3] = {
                    {triAABB.v0, triAABB.v1},
                    {triAABB.v1, triAABB.v2},
                    {triAABB.v2, triAABB.v0}
                };

                for (int faceIdx = 0; faceIdx < 4 && !intersects; faceIdx++) {
                    glm::vec3 face[3] = {
                        tetVerts[TET_FACE_INDICES[faceIdx][0]],
                        tetVerts[TET_FACE_INDICES[faceIdx][1]],
                        tetVerts[TET_FACE_INDICES[faceIdx][2]]
                    };

                    for (int e = 0; e < 3; e++) {
                        if (TetrahedronIntersection::triangleSegmentIntersect(
                                face[0], face[1], face[2],
                                triEdges[e].first, triEdges[e].second)) {
                            intersects = true;
                            break;
                        }
                    }
                }

                if (intersects) break;
            }

            if (intersects) {
                localSet.insert(tetIdx);
            }
        }
    }

    // 結果マージ
    std::set<int> intersectingTetSet;
    for (const auto& localSet : threadResults) {
        intersectingTetSet.insert(localSet.begin(), localSet.end());
    }

    intersectingTets.assign(intersectingTetSet.begin(), intersectingTetSet.end());

    auto phase3End = std::chrono::high_resolution_clock::now();
    double phase3Time = std::chrono::duration<double, std::milli>(phase3End - phase3Start).count();

    // ========== 統計出力 ==========
    auto endTime = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    if (verbose) {
        std::cout << "Phase 3 (detailed test): " << intersectingTets.size() << " intersecting, " << phase3Time << " ms" << std::endl;
        std::cout << "\n[AdjProp Performance]" << std::endl;
        std::cout << "  Seeds: " << seedTets.size() << std::endl;
        std::cout << "  Candidates (BFS): " << candidates.size() << std::endl;
        std::cout << "  Intersecting: " << intersectingTets.size() << std::endl;
        std::cout << "  Reduction: " << numTets << " -> " << candidates.size()
                  << " (" << (float)candidates.size() / numTets * 100.0f << "%)" << std::endl;
        std::cout << "\nResults:" << std::endl;
        std::cout << "  Intersecting tetrahedra: " << intersectingTets.size()
                  << " (" << (float)intersectingTets.size() / numTets * 100.0f << "%)" << std::endl;
        std::cout << "  Time: " << totalTime << " ms" << std::endl;
        std::cout << "    Phase 1 (seeds):    " << phase1Time << " ms" << std::endl;
        std::cout << "    Phase 2 (BFS):      " << phase2Time << " ms" << std::endl;
        std::cout << "    Phase 3 (detailed): " << phase3Time << " ms" << std::endl;
    }

    return intersectingTets;
}


//==============================================================================
// MeshCuttingGPUDuo 隣接グラフ伝播版（Adjacency Propagation）
//
// このコードをMeshCuttingGPUDuo.hの末尾（#endif の前）に追加してください
//
// 特徴:
//   - 変形後も再構築不要（トポロジーは変わらないため）
//   - 表面交差点からBFS伝播で局所探索
//   - BVHより実装がシンプルで高速
//
// 使用方法:
//   // 初回のみ隣接関係を構築
//   MeshCuttingGPUDuo::buildTetAdjacency(target->highResMeshData.tetIds);
//
//   // カット実行（変形後も再構築不要）
//   auto result = MeshCuttingGPUDuo::findIntersectingTetrahedraBySurfaceTriangles_AdjProp(
//       cutterData, highResData, target->highResSurfaceTriToTet,
//       target->highRes_positions, target->highResTetValid, 3, true);
//==============================================================================

//==============================================================================
// 隣接グラフ（Adjacency Graph）実装
//==============================================================================
namespace AdjacencyGraph {

// 隣接関係データ（初回構築後、変形しても更新不要）
static std::vector<std::vector<int>> g_tetNeighbors;
static bool g_isBuilt = false;

inline bool isBuilt() { return g_isBuilt; }

inline void clear() {
    g_tetNeighbors.clear();
    g_isBuilt = false;
}

// 隣接関係の構築（初回のみ、O(N)）
inline void build(const std::vector<int>& tetIds, bool verbose = true) {
    auto startTime = std::chrono::high_resolution_clock::now();

    size_t numTets = tetIds.size() / 4;

    // 面→四面体のマップを構築
    std::map<std::array<int, 3>, std::vector<int>> faceToTets;

    for (size_t tetIdx = 0; tetIdx < numTets; tetIdx++) {
        // 四面体の4つの面
        std::array<std::array<int, 3>, 4> faces = {{
            {tetIds[tetIdx*4+0], tetIds[tetIdx*4+1], tetIds[tetIdx*4+2]},
            {tetIds[tetIdx*4+0], tetIds[tetIdx*4+1], tetIds[tetIdx*4+3]},
            {tetIds[tetIdx*4+0], tetIds[tetIdx*4+2], tetIds[tetIdx*4+3]},
            {tetIds[tetIdx*4+1], tetIds[tetIdx*4+2], tetIds[tetIdx*4+3]}
        }};

        for (auto& face : faces) {
            std::sort(face.begin(), face.end());
            faceToTets[face].push_back(tetIdx);
        }
    }

    // 隣接リスト構築
    g_tetNeighbors.clear();
    g_tetNeighbors.resize(numTets);

    size_t totalEdges = 0;
    for (const auto& pair : faceToTets) {
        if (pair.second.size() == 2) {
            int t1 = pair.second[0];
            int t2 = pair.second[1];
            g_tetNeighbors[t1].push_back(t2);
            g_tetNeighbors[t2].push_back(t1);
            totalEdges++;
        }
    }

    g_isBuilt = true;

    auto endTime = std::chrono::high_resolution_clock::now();
    double buildTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    if (verbose) {
        std::cout << "\n[AdjacencyGraph] Built for " << numTets << " tetrahedra" << std::endl;
        std::cout << "  Adjacency edges: " << totalEdges << std::endl;
        std::cout << "  Build time: " << buildTime << " ms" << std::endl;
    }
}

// BFS伝播探索
inline std::vector<int> propagateFromSeeds(
    const std::vector<int>& seedTets,
    const glm::vec3& cutterMin,
    const glm::vec3& cutterMax,
    const std::vector<float>& positions,
    const std::vector<int>& tetIds,
    const std::vector<bool>& tetValid)
{
    std::vector<bool> visited(g_tetNeighbors.size(), false);
    std::queue<int> queue;
    std::vector<int> candidates;
    candidates.reserve(seedTets.size() * 10);

    // シードをキューに追加
    for (int seed : seedTets) {
        if (seed >= 0 && seed < (int)g_tetNeighbors.size()) {
            if (tetValid.empty() || tetValid[seed]) {
                queue.push(seed);
                visited[seed] = true;
            }
        }
    }

    while (!queue.empty()) {
        int tetIdx = queue.front();
        queue.pop();

        // 四面体のAABBを計算
        glm::vec3 tetMin(FLT_MAX), tetMax(-FLT_MAX);
        int baseIdx = tetIdx * 4;
        for (int j = 0; j < 4; j++) {
            int vid = tetIds[baseIdx + j];
            glm::vec3 v(positions[vid*3], positions[vid*3+1], positions[vid*3+2]);
            tetMin = glm::min(tetMin, v);
            tetMax = glm::max(tetMax, v);
        }

        // カッターAABBと交差しなければスキップ（伝播停止）
        if (tetMax.x < cutterMin.x || tetMin.x > cutterMax.x ||
            tetMax.y < cutterMin.y || tetMin.y > cutterMax.y ||
            tetMax.z < cutterMin.z || tetMin.z > cutterMax.z) {
            continue;
        }

        // 候補に追加
        candidates.push_back(tetIdx);

        // 隣接四面体を探索
        for (int neighbor : g_tetNeighbors[tetIdx]) {
            if (!visited[neighbor]) {
                if (tetValid.empty() || tetValid[neighbor]) {
                    visited[neighbor] = true;
                    queue.push(neighbor);
                }
            }
        }
    }

    return candidates;
}

} // namespace AdjacencyGraph


//==============================================================================
// 隣接関係構築用ヘルパー関数
//==============================================================================
inline void buildTetAdjacency(const std::vector<int>& tetIds, bool verbose = true) {
    AdjacencyGraph::build(tetIds, verbose);
}

inline void buildTetAdjacency(const SoftBodyGPUDuo* softBody, bool verbose = true) {
    if (!softBody) return;
    buildTetAdjacency(softBody->highResMeshData.tetIds, verbose);
}

inline void clearTetAdjacency() {
    AdjacencyGraph::clear();
}

inline bool isTetAdjacencyBuilt() {
    return AdjacencyGraph::isBuilt();
}


//==============================================================================
// 隣接グラフ伝播版 交差判定関数
// - 変形後も再構築不要
// - 表面からBFS伝播で局所探索
//==============================================================================
std::vector<int> findIntersectingTetrahedraBySurfaceTriangles_AdjProp(
    const SoftBodyGPUDuo::MeshData& cutMesh,
    const SoftBodyGPUDuo::MeshData& tetMesh,
    const std::vector<int>& surfaceTriToTet,  // 表面三角形→四面体マッピング
    const std::vector<float>& currentPositions,  // 現在の頂点位置（変形後）
    const std::vector<bool>& tetValid,
    int intersectionMode = 3,
    bool verbose = true) {

    auto startTime = std::chrono::high_resolution_clock::now();

    std::vector<int> intersectingTets;
    size_t numTets = tetMesh.tetIds.size() / 4;
    size_t numSurfaceTris = cutMesh.tetSurfaceTriIds.size() / 3;

    // 隣接グラフ未構築はエラー
    if (!AdjacencyGraph::isBuilt()) {
        std::cerr << "[AdjProp] Error: Adjacency graph not built. Call buildTetAdjacency() first." << std::endl;
        return intersectingTets;
    }

    if (verbose) {
        std::cout << "\n=== Adjacency Propagation Intersection Detection ===" << std::endl;
        std::cout << "Target tetrahedra: " << numTets << std::endl;
        std::cout << "Cut mesh surface triangles: " << numSurfaceTris << std::endl;
    }

    // ========== 三角形AABBデータ構造 ==========
    struct TriangleAABB {
        glm::vec3 min, max;
        glm::vec3 v0, v1, v2;
        int triIndex;
        bool isDummy;
    };

    std::vector<TriangleAABB> triangleAABBs;
    triangleAABBs.reserve(numSurfaceTris + 100);

    // ========== カッターのAABB計算と表面三角形処理 ==========
    glm::vec3 cutterMin(FLT_MAX), cutterMax(-FLT_MAX);
    glm::vec3 meshCenter(0.0f);

    for (size_t i = 0; i < numSurfaceTris; i++) {
        int idx0 = cutMesh.tetSurfaceTriIds[i * 3];
        int idx1 = cutMesh.tetSurfaceTriIds[i * 3 + 1];
        int idx2 = cutMesh.tetSurfaceTriIds[i * 3 + 2];

        glm::vec3 v0(cutMesh.verts[idx0 * 3], cutMesh.verts[idx0 * 3 + 1], cutMesh.verts[idx0 * 3 + 2]);
        glm::vec3 v1(cutMesh.verts[idx1 * 3], cutMesh.verts[idx1 * 3 + 1], cutMesh.verts[idx1 * 3 + 2]);
        glm::vec3 v2(cutMesh.verts[idx2 * 3], cutMesh.verts[idx2 * 3 + 1], cutMesh.verts[idx2 * 3 + 2]);

        meshCenter += v0 + v1 + v2;

        // カッター全体のAABB更新
        cutterMin = glm::min(cutterMin, glm::min(glm::min(v0, v1), v2));
        cutterMax = glm::max(cutterMax, glm::max(glm::max(v0, v1), v2));

        TriangleAABB triAABB;
        triAABB.v0 = v0;
        triAABB.v1 = v1;
        triAABB.v2 = v2;
        triAABB.min = glm::min(glm::min(v0, v1), v2);
        triAABB.max = glm::max(glm::max(v0, v1), v2);
        triAABB.triIndex = (int)i;
        triAABB.isDummy = false;
        triangleAABBs.push_back(triAABB);
    }

    meshCenter /= (float)(numSurfaceTris * 3);

    // ========== Mode 3: ダミー三角形生成 ==========
    if (intersectionMode == 3) {
        const float SAMPLING_RATE = 0.5f;
        const int MIN_SAMPLES = 8;
        const int MAX_SAMPLES = 1000;

        int numSamples = std::max(MIN_SAMPLES, std::min(MAX_SAMPLES, (int)(numSurfaceTris * SAMPLING_RATE)));

        unsigned int seed = 12345;
        auto simpleRand = [&seed]() -> unsigned int {
            seed = seed * 1103515245 + 12345;
            return (seed / 65536) % 32768;
        };

        int dummyCount = 0;
        int step = std::max(1, (int)numSurfaceTris / numSamples);

        for (size_t i = 0; i < numSurfaceTris && dummyCount < numSamples; i += step) {
            int offset = simpleRand() % std::max(1, step / 2);
            int triIdx = ((int)i + offset) % (int)numSurfaceTris;

            const auto& surfaceTri = triangleAABBs[triIdx];

            int edgeIdx = simpleRand() % 3;
            glm::vec3 edgeV0, edgeV1;

            switch (edgeIdx) {
            case 0: edgeV0 = surfaceTri.v0; edgeV1 = surfaceTri.v1; break;
            case 1: edgeV0 = surfaceTri.v1; edgeV1 = surfaceTri.v2; break;
            case 2: edgeV0 = surfaceTri.v2; edgeV1 = surfaceTri.v0; break;
            }

            TriangleAABB dummyTri;
            dummyTri.v0 = meshCenter;
            dummyTri.v1 = edgeV0;
            dummyTri.v2 = edgeV1;
            dummyTri.min = glm::min(glm::min(dummyTri.v0, dummyTri.v1), dummyTri.v2);
            dummyTri.max = glm::max(glm::max(dummyTri.v0, dummyTri.v1), dummyTri.v2);

            glm::vec3 cross = glm::cross(dummyTri.v1 - dummyTri.v0, dummyTri.v2 - dummyTri.v0);
            float area = glm::length(cross) * 0.5f;

            if (area > 0.001f) {
                dummyTri.triIndex = -1;
                dummyTri.isDummy = true;
                triangleAABBs.push_back(dummyTri);
                dummyCount++;
            }
        }

        if (verbose) {
            std::cout << "Generated " << dummyCount << " dummy triangles" << std::endl;
        }
    }

    // ========== Phase 1: シード四面体の取得 ==========
    // 表面三角形とカッターAABBの交差からシードを取得
    auto phase1Start = std::chrono::high_resolution_clock::now();

    std::set<int> seedTetSet;
    size_t numTargetSurfaceTris = tetMesh.tetSurfaceTriIds.size() / 3;

    for (size_t i = 0; i < numTargetSurfaceTris; i++) {
        // 表面三角形のAABBを計算（現在位置で）
        int idx0 = tetMesh.tetSurfaceTriIds[i * 3];
        int idx1 = tetMesh.tetSurfaceTriIds[i * 3 + 1];
        int idx2 = tetMesh.tetSurfaceTriIds[i * 3 + 2];

        glm::vec3 tv0(currentPositions[idx0 * 3], currentPositions[idx0 * 3 + 1], currentPositions[idx0 * 3 + 2]);
        glm::vec3 tv1(currentPositions[idx1 * 3], currentPositions[idx1 * 3 + 1], currentPositions[idx1 * 3 + 2]);
        glm::vec3 tv2(currentPositions[idx2 * 3], currentPositions[idx2 * 3 + 1], currentPositions[idx2 * 3 + 2]);

        glm::vec3 triMin = glm::min(glm::min(tv0, tv1), tv2);
        glm::vec3 triMax = glm::max(glm::max(tv0, tv1), tv2);

        // カッターAABBと交差チェック
        if (triMax.x < cutterMin.x || triMin.x > cutterMax.x ||
            triMax.y < cutterMin.y || triMin.y > cutterMax.y ||
            triMax.z < cutterMin.z || triMin.z > cutterMax.z) {
            continue;
        }

        // この表面三角形に対応する四面体をシードに追加
        if (i < surfaceTriToTet.size()) {
            int tetIdx = surfaceTriToTet[i];
            if (tetIdx >= 0 && (tetValid.empty() || tetValid[tetIdx])) {
                seedTetSet.insert(tetIdx);
            }
        }
    }

    std::vector<int> seedTets(seedTetSet.begin(), seedTetSet.end());

    auto phase1End = std::chrono::high_resolution_clock::now();
    double phase1Time = std::chrono::duration<double, std::milli>(phase1End - phase1Start).count();

    if (verbose) {
        std::cout << "Phase 1 (seed detection): " << seedTets.size() << " seeds, " << phase1Time << " ms" << std::endl;
    }

    if (seedTets.empty()) {
        if (verbose) {
            std::cout << "No seeds found - no surface intersection" << std::endl;
        }
        return intersectingTets;
    }

    // ========== Phase 2: BFS伝播で候補取得 ==========
    auto phase2Start = std::chrono::high_resolution_clock::now();

    std::vector<int> candidates = AdjacencyGraph::propagateFromSeeds(
        seedTets, cutterMin, cutterMax, currentPositions, tetMesh.tetIds, tetValid);

    auto phase2End = std::chrono::high_resolution_clock::now();
    double phase2Time = std::chrono::duration<double, std::milli>(phase2End - phase2Start).count();

    if (verbose) {
        std::cout << "Phase 2 (BFS propagation): " << candidates.size() << " candidates, " << phase2Time << " ms" << std::endl;
    }

    // ========== Phase 3: 詳細交差判定（並列化）==========
    auto phase3Start = std::chrono::high_resolution_clock::now();

    int numThreads = 1;
#ifdef _OPENMP
    numThreads = omp_get_max_threads();
#endif

    std::vector<std::set<int>> threadResults(numThreads);

#pragma omp parallel
    {
        int threadId = 0;
#ifdef _OPENMP
        threadId = omp_get_thread_num();
#endif
        std::set<int>& localSet = threadResults[threadId];

#pragma omp for schedule(dynamic, 32)
        for (int candIdx = 0; candIdx < (int)candidates.size(); candIdx++) {
            int tetIdx = candidates[candIdx];

            // 四面体の頂点を取得（現在位置）
            glm::vec3 tetVerts[4];
            int baseIdx = tetIdx * 4;
            for (int j = 0; j < 4; j++) {
                int vid = tetMesh.tetIds[baseIdx + j];
                tetVerts[j] = glm::vec3(
                    currentPositions[vid * 3],
                    currentPositions[vid * 3 + 1],
                    currentPositions[vid * 3 + 2]
                    );
            }

            // 四面体のAABB
            glm::vec3 tetMin = tetVerts[0], tetMax = tetVerts[0];
            for (int j = 1; j < 4; j++) {
                tetMin = glm::min(tetMin, tetVerts[j]);
                tetMax = glm::max(tetMax, tetVerts[j]);
            }

            bool intersects = false;

            // 各カッター三角形との交差判定
            for (const auto& triAABB : triangleAABBs) {
                if (intersectionMode == 0 && triAABB.isDummy) continue;

                // AABBフィルタリング
                if (tetMax.x < triAABB.min.x || tetMin.x > triAABB.max.x ||
                    tetMax.y < triAABB.min.y || tetMin.y > triAABB.max.y ||
                    tetMax.z < triAABB.min.z || tetMin.z > triAABB.max.z) {
                    continue;
                }

                // 1. 三角形の頂点が四面体内部
                if (MeshCuttingGPUDuo::TetrahedronIntersection::pointInTetrahedron(
                        triAABB.v0, tetVerts[0], tetVerts[1], tetVerts[2], tetVerts[3]) ||
                    MeshCuttingGPUDuo::TetrahedronIntersection::pointInTetrahedron(
                        triAABB.v1, tetVerts[0], tetVerts[1], tetVerts[2], tetVerts[3]) ||
                    MeshCuttingGPUDuo::TetrahedronIntersection::pointInTetrahedron(
                        triAABB.v2, tetVerts[0], tetVerts[1], tetVerts[2], tetVerts[3])) {
                    intersects = true;
                    break;
                }

                // 2. 四面体のエッジと三角形の交差
                static const int edgePairs[6][2] = {
                    {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
                };

                for (int e = 0; e < 6 && !intersects; e++) {
                    if (MeshCuttingGPUDuo::TetrahedronIntersection::triangleSegmentIntersect(
                            triAABB.v0, triAABB.v1, triAABB.v2,
                            tetVerts[edgePairs[e][0]],
                            tetVerts[edgePairs[e][1]])) {
                        intersects = true;
                    }
                }

                if (intersects) break;

                // 3. 四面体の面と三角形のエッジの交差
                static const int TET_FACE_INDICES[4][3] = {
                    {1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}
                };

                std::pair<glm::vec3, glm::vec3> triEdges[3] = {
                    {triAABB.v0, triAABB.v1},
                    {triAABB.v1, triAABB.v2},
                    {triAABB.v2, triAABB.v0}
                };

                for (int faceIdx = 0; faceIdx < 4 && !intersects; faceIdx++) {
                    glm::vec3 face[3] = {
                        tetVerts[TET_FACE_INDICES[faceIdx][0]],
                        tetVerts[TET_FACE_INDICES[faceIdx][1]],
                        tetVerts[TET_FACE_INDICES[faceIdx][2]]
                    };

                    for (int e = 0; e < 3; e++) {
                        if (MeshCuttingGPUDuo::TetrahedronIntersection::triangleSegmentIntersect(
                                face[0], face[1], face[2],
                                triEdges[e].first, triEdges[e].second)) {
                            intersects = true;
                            break;
                        }
                    }
                }

                if (intersects) break;
            }

            if (intersects) {
                localSet.insert(tetIdx);
            }
        }
    }

    // 結果マージ
    std::set<int> intersectingTetSet;
    for (const auto& localSet : threadResults) {
        intersectingTetSet.insert(localSet.begin(), localSet.end());
    }

    intersectingTets.assign(intersectingTetSet.begin(), intersectingTetSet.end());

    auto phase3End = std::chrono::high_resolution_clock::now();
    double phase3Time = std::chrono::duration<double, std::milli>(phase3End - phase3Start).count();

    // ========== 統計出力 ==========
    auto endTime = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    if (verbose) {
        std::cout << "Phase 3 (detailed test): " << intersectingTets.size() << " intersecting, " << phase3Time << " ms" << std::endl;
        std::cout << "\n[AdjProp Performance]" << std::endl;
        std::cout << "  Seeds: " << seedTets.size() << std::endl;
        std::cout << "  Candidates (BFS): " << candidates.size() << std::endl;
        std::cout << "  Intersecting: " << intersectingTets.size() << std::endl;
        std::cout << "  Reduction: " << numTets << " -> " << candidates.size()
                  << " (" << (float)candidates.size() / numTets * 100.0f << "%)" << std::endl;
        std::cout << "\nResults:" << std::endl;
        std::cout << "  Intersecting tetrahedra: " << intersectingTets.size()
                  << " (" << (float)intersectingTets.size() / numTets * 100.0f << "%)" << std::endl;
        std::cout << "  Time: " << totalTime << " ms" << std::endl;
        std::cout << "    Phase 1 (seeds):    " << phase1Time << " ms" << std::endl;
        std::cout << "    Phase 2 (BFS):      " << phase2Time << " ms" << std::endl;
        std::cout << "    Phase 3 (detailed): " << phase3Time << " ms" << std::endl;
    }

    return intersectingTets;
}


}




#endif // MESH_CUTTING_GPU_DUO_H
