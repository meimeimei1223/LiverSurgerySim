#include "SoftBodyParallelSolver.h"
#include <omp.h>
#include <cstring>
#include "SoftBodyGPUDuo.h"
// ===========================================================================
// 初期化
// ===========================================================================

void SoftBodyParallelSolver::initialize(
    size_t numParticles,
    const std::vector<int>& edgeIds,
    const std::vector<int>& tetIds,
    const std::vector<bool>& tetValid
    ) {
    std::cout << "\n=== SoftBodyParallelSolver Initialization ===" << std::endl;

    numParticlesStored = static_cast<int>(numParticles);

    computeEdgeColoring(numParticles, edgeIds);
    computeTetColoring(numParticles, tetIds, tetValid);

    // Jacobi用バッファ確保（通常のvector）
    corrBufferX.resize(numParticles, 0.0);
    corrBufferY.resize(numParticles, 0.0);
    corrBufferZ.resize(numParticles, 0.0);
    corrCounts.resize(numParticles, 0);

    if (numThreads <= 0) {
        numThreads = omp_get_max_threads();
    }

    initialized = true;

    //printStats();
    std::cout << "  Using " << numThreads << " threads" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

// ===========================================================================
// グラフカラーリング（エッジ）
// ===========================================================================

void SoftBodyParallelSolver::computeEdgeColoring(
    size_t numParticles,
    const std::vector<int>& edgeIds
    ) {
    std::cout << "  Computing edge coloring..." << std::endl;

    int numEdges = static_cast<int>(edgeIds.size() / 2);
    edgeColorGroups.clear();
    std::vector<int> edgeColors(numEdges, -1);

    std::vector<std::set<int>> vertexEdges(numParticles);
    for (int i = 0; i < numEdges; i++) {
        int v0 = edgeIds[i * 2 + 0];
        int v1 = edgeIds[i * 2 + 1];
        vertexEdges[v0].insert(i);
        vertexEdges[v1].insert(i);
    }

    for (int i = 0; i < numEdges; i++) {
        int v0 = edgeIds[i * 2 + 0];
        int v1 = edgeIds[i * 2 + 1];

        std::set<int> usedColors;
        for (int neighborEdge : vertexEdges[v0]) {
            if (edgeColors[neighborEdge] >= 0) {
                usedColors.insert(edgeColors[neighborEdge]);
            }
        }
        for (int neighborEdge : vertexEdges[v1]) {
            if (edgeColors[neighborEdge] >= 0) {
                usedColors.insert(edgeColors[neighborEdge]);
            }
        }

        int color = 0;
        while (usedColors.count(color) > 0) color++;
        edgeColors[i] = color;
    }

    int maxColor = *std::max_element(edgeColors.begin(), edgeColors.end());
    edgeColorGroups.resize(maxColor + 1);
    for (int i = 0; i < numEdges; i++) {
        edgeColorGroups[edgeColors[i]].push_back(i);
    }

    std::cout << "    Edge colors: " << edgeColorGroups.size() << std::endl;
}

// ===========================================================================
// グラフカラーリング（四面体）
// ===========================================================================

void SoftBodyParallelSolver::computeTetColoring(
    size_t numParticles,
    const std::vector<int>& tetIds,
    const std::vector<bool>& tetValid
    ) {
    std::cout << "  Computing tetrahedra coloring..." << std::endl;

    int numTets = static_cast<int>(tetIds.size() / 4);
    tetColorGroups.clear();
    std::vector<int> tetColors(numTets, -1);

    std::vector<std::set<int>> vertexTets(numParticles);
    for (int i = 0; i < numTets; i++) {
        if (!tetValid.empty() && !tetValid[i]) continue;
        for (int j = 0; j < 4; j++) {
            int vid = tetIds[i * 4 + j];
            vertexTets[vid].insert(i);
        }
    }

    for (int i = 0; i < numTets; i++) {
        if (!tetValid.empty() && !tetValid[i]) continue;

        std::set<int> usedColors;
        for (int j = 0; j < 4; j++) {
            int vid = tetIds[i * 4 + j];
            for (int neighborTet : vertexTets[vid]) {
                if (tetColors[neighborTet] >= 0) {
                    usedColors.insert(tetColors[neighborTet]);
                }
            }
        }

        int color = 0;
        while (usedColors.count(color) > 0) color++;
        tetColors[i] = color;
    }

    int maxColor = -1;
    for (int i = 0; i < numTets; i++) {
        if (tetColors[i] > maxColor) maxColor = tetColors[i];
    }

    if (maxColor >= 0) {
        tetColorGroups.resize(maxColor + 1);
        for (int i = 0; i < numTets; i++) {
            if (tetColors[i] >= 0) {
                tetColorGroups[tetColors[i]].push_back(i);
            }
        }
    }

    std::cout << "    Tet colors: " << tetColorGroups.size() << std::endl;
}

// ===========================================================================
// エッジ制約ソルバー（メイン）
// ===========================================================================

// void SoftBodyParallelSolver::solveEdges(
//     std::vector<float>& positions,
//     const std::vector<float>& invMasses,
//     const std::vector<float>& restLengths,
//     std::vector<float>& lambdas,
//     const std::vector<int>& edgeIds,
//     const std::vector<bool>& edgeValid,
//     float compliance,
//     float dt,
//     int iterations
//     ) {
//     float alpha = compliance / (dt * dt);

//     switch (solveType) {
//     case GAUSS_SEIDEL:
//         for (int iter = 0; iter < iterations; iter++) {
//             solveEdgesGS(positions, invMasses, restLengths, lambdas, edgeIds, edgeValid, alpha);
//         }
//         break;

//     case JACOBI:
//         for (int iter = 0; iter < iterations; iter++) {
//             solveEdgesJacobi(positions, invMasses, restLengths, lambdas, edgeIds, edgeValid, alpha);
//         }
//         break;

//     case HYBRID: {
//         int actualGsColors = std::min(gsEdgeColorCount, static_cast<int>(edgeColorGroups.size()));

//         // GS部分
//         for (int c = 0; c < actualGsColors; c++) {
//             const std::vector<int>& colorGroup = edgeColorGroups[c];
//             int groupSize = static_cast<int>(colorGroup.size());

// #pragma omp parallel for
//             for (int i = 0; i < groupSize; i++) {
//                 int edgeIdx = colorGroup[i];
//                 if (!edgeValid.empty() && !edgeValid[edgeIdx]) continue;

//                 int id0 = edgeIds[edgeIdx * 2 + 0];
//                 int id1 = edgeIds[edgeIdx * 2 + 1];

//                 float w0 = invMasses[id0];
//                 float w1 = invMasses[id1];
//                 float wSum = w0 + w1;
//                 if (wSum == 0.0f) continue;

//                 float dx = positions[id1 * 3 + 0] - positions[id0 * 3 + 0];
//                 float dy = positions[id1 * 3 + 1] - positions[id0 * 3 + 1];
//                 float dz = positions[id1 * 3 + 2] - positions[id0 * 3 + 2];
//                 float len = std::sqrt(dx*dx + dy*dy + dz*dz);
//                 if (len < 1e-7f) continue;

//                 float invLen = 1.0f / len;
//                 dx *= invLen; dy *= invLen; dz *= invLen;

//                 float restLen = restLengths[edgeIdx];
//                 float C = len - restLen;
//                 float dLambda = -(C + alpha * lambdas[edgeIdx]) / (wSum + alpha);
//                 lambdas[edgeIdx] += dLambda;

//                 positions[id0 * 3 + 0] -= w0 * dx * dLambda;
//                 positions[id0 * 3 + 1] -= w0 * dy * dLambda;
//                 positions[id0 * 3 + 2] -= w0 * dz * dLambda;
//                 positions[id1 * 3 + 0] += w1 * dx * dLambda;
//                 positions[id1 * 3 + 1] += w1 * dy * dLambda;
//                 positions[id1 * 3 + 2] += w1 * dz * dLambda;
//             }
//         }

//         // 残りはJacobi
//         if (actualGsColors < static_cast<int>(edgeColorGroups.size())) {
//             for (int jiter = 0; jiter < hybridJacobiIterations; jiter++) {
//                 int numP = numParticlesStored;

// // バッファクリア
// #pragma omp parallel for
//                 for (int i = 0; i < numP; i++) {
//                     corrBufferX[i] = 0.0;
//                     corrBufferY[i] = 0.0;
//                     corrBufferZ[i] = 0.0;
//                     corrCounts[i] = 0;
//                 }

//                 // 残りの色を処理
//                 int numColors = static_cast<int>(edgeColorGroups.size());
//                 for (int c = actualGsColors; c < numColors; c++) {
//                     const std::vector<int>& colorGroup = edgeColorGroups[c];
//                     int groupSize = static_cast<int>(colorGroup.size());

// #pragma omp parallel for
//                     for (int i = 0; i < groupSize; i++) {
//                         int edgeIdx = colorGroup[i];
//                         if (!edgeValid.empty() && !edgeValid[edgeIdx]) continue;

//                         int id0 = edgeIds[edgeIdx * 2 + 0];
//                         int id1 = edgeIds[edgeIdx * 2 + 1];

//                         float w0 = invMasses[id0];
//                         float w1 = invMasses[id1];
//                         float wSum = w0 + w1;
//                         if (wSum == 0.0f) continue;

//                         float dx = positions[id1 * 3 + 0] - positions[id0 * 3 + 0];
//                         float dy = positions[id1 * 3 + 1] - positions[id0 * 3 + 1];
//                         float dz = positions[id1 * 3 + 2] - positions[id0 * 3 + 2];
//                         float len = std::sqrt(dx*dx + dy*dy + dz*dz);
//                         if (len < 1e-7f) continue;

//                         float invLen = 1.0f / len;
//                         dx *= invLen; dy *= invLen; dz *= invLen;

//                         float restLen = restLengths[edgeIdx];
//                         float C = len - restLen;
//                         float dLambda = -(C + alpha * lambdas[edgeIdx]) / (wSum + alpha);
//                         lambdas[edgeIdx] += dLambda;

//                         double c0x = -w0 * dx * dLambda;
//                         double c0y = -w0 * dy * dLambda;
//                         double c0z = -w0 * dz * dLambda;
//                         double c1x = w1 * dx * dLambda;
//                         double c1y = w1 * dy * dLambda;
//                         double c1z = w1 * dz * dLambda;

// #pragma omp atomic
//                         corrBufferX[id0] += c0x;
// #pragma omp atomic
//                         corrBufferY[id0] += c0y;
// #pragma omp atomic
//                         corrBufferZ[id0] += c0z;
// #pragma omp atomic
//                         corrCounts[id0] += 1;

// #pragma omp atomic
//                         corrBufferX[id1] += c1x;
// #pragma omp atomic
//                         corrBufferY[id1] += c1y;
// #pragma omp atomic
//                         corrBufferZ[id1] += c1z;
// #pragma omp atomic
//                         corrCounts[id1] += 1;
//                     }
//                 }

//                 // 補正適用
// // 補正適用
// #pragma omp parallel for
//                 for (int i = 0; i < numP; i++) {
//                     if (corrCounts[i] > 0) {
//                         positions[i * 3 + 0] += static_cast<float>(corrBufferX[i]) * jacobiScale;  // ★
//                         positions[i * 3 + 1] += static_cast<float>(corrBufferY[i]) * jacobiScale;  // ★
//                         positions[i * 3 + 2] += static_cast<float>(corrBufferZ[i]) * jacobiScale;  // ★
//                     }
//                 }
//             }
//         }
//         break;
//     }
//     }
// }

// ===========================================================================
// Gauss-Seidel エッジソルバー
// ===========================================================================

void SoftBodyParallelSolver::solveEdgesGS(
    std::vector<float>& positions,
    const std::vector<float>& invMasses,
    const std::vector<float>& restLengths,
    std::vector<float>& lambdas,
    const std::vector<int>& edgeIds,
    const std::vector<bool>& edgeValid,
    float alpha
    ) {
    int numColorGroups = static_cast<int>(edgeColorGroups.size());
    for (int c = 0; c < numColorGroups; c++) {
        const std::vector<int>& colorGroup = edgeColorGroups[c];
        int groupSize = static_cast<int>(colorGroup.size());

#pragma omp parallel for
        for (int i = 0; i < groupSize; i++) {
            int edgeIdx = colorGroup[i];
            if (!edgeValid.empty() && !edgeValid[edgeIdx]) continue;

            int id0 = edgeIds[edgeIdx * 2 + 0];
            int id1 = edgeIds[edgeIdx * 2 + 1];

            float w0 = invMasses[id0];
            float w1 = invMasses[id1];
            float wSum = w0 + w1;
            if (wSum == 0.0f) continue;

            float dx = positions[id1 * 3 + 0] - positions[id0 * 3 + 0];
            float dy = positions[id1 * 3 + 1] - positions[id0 * 3 + 1];
            float dz = positions[id1 * 3 + 2] - positions[id0 * 3 + 2];
            float len = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (len < 1e-7f) continue;

            float invLen = 1.0f / len;
            dx *= invLen; dy *= invLen; dz *= invLen;

            float restLen = restLengths[edgeIdx];
            float C = len - restLen;
            float dLambda = -(C + alpha * lambdas[edgeIdx]) / (wSum + alpha);
            lambdas[edgeIdx] += dLambda;

            positions[id0 * 3 + 0] -= w0 * dx * dLambda;
            positions[id0 * 3 + 1] -= w0 * dy * dLambda;
            positions[id0 * 3 + 2] -= w0 * dz * dLambda;
            positions[id1 * 3 + 0] += w1 * dx * dLambda;
            positions[id1 * 3 + 1] += w1 * dy * dLambda;
            positions[id1 * 3 + 2] += w1 * dz * dLambda;
        }
    }
}

// ===========================================================================
// Jacobi エッジソルバー
// ===========================================================================

void SoftBodyParallelSolver::solveEdgesJacobi(
    std::vector<float>& positions,
    const std::vector<float>& invMasses,
    const std::vector<float>& restLengths,
    std::vector<float>& lambdas,
    const std::vector<int>& edgeIds,
    const std::vector<bool>& edgeValid,
    float alpha
    ) {
    int numParticles = numParticlesStored;
    int numEdges = static_cast<int>(edgeIds.size() / 2);

    // バッファクリア
#pragma omp parallel for
    for (int i = 0; i < numParticles; i++) {
        corrBufferX[i] = 0.0;
        corrBufferY[i] = 0.0;
        corrBufferZ[i] = 0.0;
        corrCounts[i] = 0;
    }

    // 全エッジ並列処理
#pragma omp parallel for
    for (int edgeIdx = 0; edgeIdx < numEdges; edgeIdx++) {
        if (!edgeValid.empty() && !edgeValid[edgeIdx]) continue;

        int id0 = edgeIds[edgeIdx * 2 + 0];
        int id1 = edgeIds[edgeIdx * 2 + 1];

        float w0 = invMasses[id0];
        float w1 = invMasses[id1];
        float wSum = w0 + w1;
        if (wSum == 0.0f) continue;

        float dx = positions[id1 * 3 + 0] - positions[id0 * 3 + 0];
        float dy = positions[id1 * 3 + 1] - positions[id0 * 3 + 1];
        float dz = positions[id1 * 3 + 2] - positions[id0 * 3 + 2];
        float len = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (len < 1e-7f) continue;

        float invLen = 1.0f / len;
        dx *= invLen; dy *= invLen; dz *= invLen;

        float restLen = restLengths[edgeIdx];
        float C = len - restLen;
        float dLambda = -(C + alpha * lambdas[edgeIdx]) / (wSum + alpha);
        lambdas[edgeIdx] += dLambda;

        double c0x = -w0 * dx * dLambda;
        double c0y = -w0 * dy * dLambda;
        double c0z = -w0 * dz * dLambda;
        double c1x = w1 * dx * dLambda;
        double c1y = w1 * dy * dLambda;
        double c1z = w1 * dz * dLambda;

#pragma omp atomic
        corrBufferX[id0] += c0x;
#pragma omp atomic
        corrBufferY[id0] += c0y;
#pragma omp atomic
        corrBufferZ[id0] += c0z;
#pragma omp atomic
        corrCounts[id0] += 1;

#pragma omp atomic
        corrBufferX[id1] += c1x;
#pragma omp atomic
        corrBufferY[id1] += c1y;
#pragma omp atomic
        corrBufferZ[id1] += c1z;
#pragma omp atomic
        corrCounts[id1] += 1;
    }

// 補正適用（末尾付近）
#pragma omp parallel for
    for (int i = 0; i < numParticles; i++) {
        if (corrCounts[i] > 0) {
            positions[i * 3 + 0] += static_cast<float>(corrBufferX[i]) * jacobiScale;  // ★追加
            positions[i * 3 + 1] += static_cast<float>(corrBufferY[i]) * jacobiScale;  // ★追加
            positions[i * 3 + 2] += static_cast<float>(corrBufferZ[i]) * jacobiScale;  // ★追加
        }
    }
}

// ===========================================================================
// 体積制約ソルバー（メイン）
// ===========================================================================

//void SoftBodyParallelSolver::solveVolumes(
//     std::vector<float>& positions,
//     const std::vector<float>& invMasses,
//     const std::vector<float>& restVolumes,
//     std::vector<float>& lambdas,
//     const std::vector<int>& tetIds,
//     const std::vector<bool>& tetValid,
//     float compliance,
//     float dt,
//     int iterations
//     ) {
//     float alpha = compliance / (dt * dt);

//     switch (solveType) {
//     case GAUSS_SEIDEL:
//         for (int iter = 0; iter < iterations; iter++) {
//             solveTetsGS(positions, invMasses, restVolumes, lambdas, tetIds, tetValid, alpha);
//         }
//         break;

//     case JACOBI:
//         for (int iter = 0; iter < iterations; iter++) {
//             solveTetsJacobi(positions, invMasses, restVolumes, lambdas, tetIds, tetValid, alpha);
//         }
//         break;

//     case HYBRID: {
//         int actualGsColors = std::min(gsTetColorCount, static_cast<int>(tetColorGroups.size()));

//         // GS部分
//         for (int c = 0; c < actualGsColors; c++) {
//             const std::vector<int>& colorGroup = tetColorGroups[c];
//             int groupSize = static_cast<int>(colorGroup.size());

// #pragma omp parallel for
//             for (int i = 0; i < groupSize; i++) {
//                 int tetIdx = colorGroup[i];
//                 if (!tetValid.empty() && !tetValid[tetIdx]) continue;

//                 int id0 = tetIds[tetIdx * 4 + 0];
//                 int id1 = tetIds[tetIdx * 4 + 1];
//                 int id2 = tetIds[tetIdx * 4 + 2];
//                 int id3 = tetIds[tetIdx * 4 + 3];

//                 float w0 = invMasses[id0], w1 = invMasses[id1], w2 = invMasses[id2], w3 = invMasses[id3];

//                 float p0x = positions[id0 * 3], p0y = positions[id0 * 3 + 1], p0z = positions[id0 * 3 + 2];
//                 float p1x = positions[id1 * 3], p1y = positions[id1 * 3 + 1], p1z = positions[id1 * 3 + 2];
//                 float p2x = positions[id2 * 3], p2y = positions[id2 * 3 + 1], p2z = positions[id2 * 3 + 2];
//                 float p3x = positions[id3 * 3], p3y = positions[id3 * 3 + 1], p3z = positions[id3 * 3 + 2];

//                 float g0x, g0y, g0z, g1x, g1y, g1z, g2x, g2y, g2z, g3x, g3y, g3z;
//                 { float ax=p3x-p1x,ay=p3y-p1y,az=p3z-p1z,bx=p2x-p1x,by=p2y-p1y,bz=p2z-p1z;
//                     g0x=(ay*bz-az*by)/6.0f; g0y=(az*bx-ax*bz)/6.0f; g0z=(ax*by-ay*bx)/6.0f; }
//                 { float ax=p2x-p0x,ay=p2y-p0y,az=p2z-p0z,bx=p3x-p0x,by=p3y-p0y,bz=p3z-p0z;
//                     g1x=(ay*bz-az*by)/6.0f; g1y=(az*bx-ax*bz)/6.0f; g1z=(ax*by-ay*bx)/6.0f; }
//                 { float ax=p3x-p0x,ay=p3y-p0y,az=p3z-p0z,bx=p1x-p0x,by=p1y-p0y,bz=p1z-p0z;
//                     g2x=(ay*bz-az*by)/6.0f; g2y=(az*bx-ax*bz)/6.0f; g2z=(ax*by-ay*bx)/6.0f; }
//                 { float ax=p1x-p0x,ay=p1y-p0y,az=p1z-p0z,bx=p2x-p0x,by=p2y-p0y,bz=p2z-p0z;
//                     g3x=(ay*bz-az*by)/6.0f; g3y=(az*bx-ax*bz)/6.0f; g3z=(ax*by-ay*bx)/6.0f; }

//                 float wSum = w0*(g0x*g0x+g0y*g0y+g0z*g0z) + w1*(g1x*g1x+g1y*g1y+g1z*g1z)
//                              + w2*(g2x*g2x+g2y*g2y+g2z*g2z) + w3*(g3x*g3x+g3y*g3y+g3z*g3z);
//                 if (wSum < 1e-10f) continue;

//                 float vol = ((p1x-p0x)*((p2y-p0y)*(p3z-p0z)-(p2z-p0z)*(p3y-p0y))
//                              + (p1y-p0y)*((p2z-p0z)*(p3x-p0x)-(p2x-p0x)*(p3z-p0z))
//                              + (p1z-p0z)*((p2x-p0x)*(p3y-p0y)-(p2y-p0y)*(p3x-p0x))) / 6.0f;

//                 float restVol = restVolumes[tetIdx];
//                 float C = vol - restVol;
//                 float dLambda = -(C + alpha * lambdas[tetIdx]) / (wSum + alpha);
//                 lambdas[tetIdx] += dLambda;

//                 positions[id0*3+0]+=w0*g0x*dLambda; positions[id0*3+1]+=w0*g0y*dLambda; positions[id0*3+2]+=w0*g0z*dLambda;
//                 positions[id1*3+0]+=w1*g1x*dLambda; positions[id1*3+1]+=w1*g1y*dLambda; positions[id1*3+2]+=w1*g1z*dLambda;
//                 positions[id2*3+0]+=w2*g2x*dLambda; positions[id2*3+1]+=w2*g2y*dLambda; positions[id2*3+2]+=w2*g2z*dLambda;
//                 positions[id3*3+0]+=w3*g3x*dLambda; positions[id3*3+1]+=w3*g3y*dLambda; positions[id3*3+2]+=w3*g3z*dLambda;
//             }
//         }

//         // 残りはJacobi
//         if (actualGsColors < static_cast<int>(tetColorGroups.size())) {
//             for (int jiter = 0; jiter < hybridJacobiIterations; jiter++) {
//                 solveTetsJacobi(positions, invMasses, restVolumes, lambdas, tetIds, tetValid, alpha);
//             }
//         }
//         break;
//     }
//     }
// }

// ===========================================================================
// Gauss-Seidel 四面体ソルバー
// ===========================================================================

void SoftBodyParallelSolver::solveTetsGS(
    std::vector<float>& positions,
    const std::vector<float>& invMasses,
    const std::vector<float>& restVolumes,
    std::vector<float>& lambdas,
    const std::vector<int>& tetIds,
    const std::vector<bool>& tetValid,
    float alpha
    ) {
    int numColorGroups = static_cast<int>(tetColorGroups.size());
    for (int c = 0; c < numColorGroups; c++) {
        const std::vector<int>& colorGroup = tetColorGroups[c];
        int groupSize = static_cast<int>(colorGroup.size());

#pragma omp parallel for
        for (int i = 0; i < groupSize; i++) {
            int tetIdx = colorGroup[i];
            if (!tetValid.empty() && !tetValid[tetIdx]) continue;

            int id0 = tetIds[tetIdx * 4 + 0];
            int id1 = tetIds[tetIdx * 4 + 1];
            int id2 = tetIds[tetIdx * 4 + 2];
            int id3 = tetIds[tetIdx * 4 + 3];

            float w0 = invMasses[id0], w1 = invMasses[id1], w2 = invMasses[id2], w3 = invMasses[id3];

            float p0x = positions[id0 * 3], p0y = positions[id0 * 3 + 1], p0z = positions[id0 * 3 + 2];
            float p1x = positions[id1 * 3], p1y = positions[id1 * 3 + 1], p1z = positions[id1 * 3 + 2];
            float p2x = positions[id2 * 3], p2y = positions[id2 * 3 + 1], p2z = positions[id2 * 3 + 2];
            float p3x = positions[id3 * 3], p3y = positions[id3 * 3 + 1], p3z = positions[id3 * 3 + 2];

            float g0x, g0y, g0z, g1x, g1y, g1z, g2x, g2y, g2z, g3x, g3y, g3z;
            { float ax=p3x-p1x,ay=p3y-p1y,az=p3z-p1z,bx=p2x-p1x,by=p2y-p1y,bz=p2z-p1z;
                g0x=(ay*bz-az*by)/6.0f; g0y=(az*bx-ax*bz)/6.0f; g0z=(ax*by-ay*bx)/6.0f; }
            { float ax=p2x-p0x,ay=p2y-p0y,az=p2z-p0z,bx=p3x-p0x,by=p3y-p0y,bz=p3z-p0z;
                g1x=(ay*bz-az*by)/6.0f; g1y=(az*bx-ax*bz)/6.0f; g1z=(ax*by-ay*bx)/6.0f; }
            { float ax=p3x-p0x,ay=p3y-p0y,az=p3z-p0z,bx=p1x-p0x,by=p1y-p0y,bz=p1z-p0z;
                g2x=(ay*bz-az*by)/6.0f; g2y=(az*bx-ax*bz)/6.0f; g2z=(ax*by-ay*bx)/6.0f; }
            { float ax=p1x-p0x,ay=p1y-p0y,az=p1z-p0z,bx=p2x-p0x,by=p2y-p0y,bz=p2z-p0z;
                g3x=(ay*bz-az*by)/6.0f; g3y=(az*bx-ax*bz)/6.0f; g3z=(ax*by-ay*bx)/6.0f; }

            float wSum = w0*(g0x*g0x+g0y*g0y+g0z*g0z) + w1*(g1x*g1x+g1y*g1y+g1z*g1z)
                         + w2*(g2x*g2x+g2y*g2y+g2z*g2z) + w3*(g3x*g3x+g3y*g3y+g3z*g3z);
            if (wSum < 1e-10f) continue;

            float vol = ((p1x-p0x)*((p2y-p0y)*(p3z-p0z)-(p2z-p0z)*(p3y-p0y))
                         + (p1y-p0y)*((p2z-p0z)*(p3x-p0x)-(p2x-p0x)*(p3z-p0z))
                         + (p1z-p0z)*((p2x-p0x)*(p3y-p0y)-(p2y-p0y)*(p3x-p0x))) / 6.0f;

            float restVol = restVolumes[tetIdx];
            float C = vol - restVol;
            float dLambda = -(C + alpha * lambdas[tetIdx]) / (wSum + alpha);
            lambdas[tetIdx] += dLambda;

            positions[id0*3+0]+=w0*g0x*dLambda; positions[id0*3+1]+=w0*g0y*dLambda; positions[id0*3+2]+=w0*g0z*dLambda;
            positions[id1*3+0]+=w1*g1x*dLambda; positions[id1*3+1]+=w1*g1y*dLambda; positions[id1*3+2]+=w1*g1z*dLambda;
            positions[id2*3+0]+=w2*g2x*dLambda; positions[id2*3+1]+=w2*g2y*dLambda; positions[id2*3+2]+=w2*g2z*dLambda;
            positions[id3*3+0]+=w3*g3x*dLambda; positions[id3*3+1]+=w3*g3y*dLambda; positions[id3*3+2]+=w3*g3z*dLambda;
        }
    }
}

// ===========================================================================
// Jacobi 四面体ソルバー
// ===========================================================================

void SoftBodyParallelSolver::solveTetsJacobi(
    std::vector<float>& positions,
    const std::vector<float>& invMasses,
    const std::vector<float>& restVolumes,
    std::vector<float>& lambdas,
    const std::vector<int>& tetIds,
    const std::vector<bool>& tetValid,
    float alpha
    ) {
    int numParticles = numParticlesStored;
    int numTets = static_cast<int>(tetIds.size() / 4);

    // バッファクリア
#pragma omp parallel for
    for (int i = 0; i < numParticles; i++) {
        corrBufferX[i] = 0.0;
        corrBufferY[i] = 0.0;
        corrBufferZ[i] = 0.0;
        corrCounts[i] = 0;
    }

    // 全四面体並列処理
#pragma omp parallel for
    for (int tetIdx = 0; tetIdx < numTets; tetIdx++) {
        if (!tetValid.empty() && !tetValid[tetIdx]) continue;

        int id0 = tetIds[tetIdx * 4 + 0];
        int id1 = tetIds[tetIdx * 4 + 1];
        int id2 = tetIds[tetIdx * 4 + 2];
        int id3 = tetIds[tetIdx * 4 + 3];

        float w0 = invMasses[id0], w1 = invMasses[id1], w2 = invMasses[id2], w3 = invMasses[id3];

        float p0x = positions[id0 * 3], p0y = positions[id0 * 3 + 1], p0z = positions[id0 * 3 + 2];
        float p1x = positions[id1 * 3], p1y = positions[id1 * 3 + 1], p1z = positions[id1 * 3 + 2];
        float p2x = positions[id2 * 3], p2y = positions[id2 * 3 + 1], p2z = positions[id2 * 3 + 2];
        float p3x = positions[id3 * 3], p3y = positions[id3 * 3 + 1], p3z = positions[id3 * 3 + 2];

        float g0x, g0y, g0z, g1x, g1y, g1z, g2x, g2y, g2z, g3x, g3y, g3z;
        { float ax=p3x-p1x,ay=p3y-p1y,az=p3z-p1z,bx=p2x-p1x,by=p2y-p1y,bz=p2z-p1z;
            g0x=(ay*bz-az*by)/6.0f; g0y=(az*bx-ax*bz)/6.0f; g0z=(ax*by-ay*bx)/6.0f; }
        { float ax=p2x-p0x,ay=p2y-p0y,az=p2z-p0z,bx=p3x-p0x,by=p3y-p0y,bz=p3z-p0z;
            g1x=(ay*bz-az*by)/6.0f; g1y=(az*bx-ax*bz)/6.0f; g1z=(ax*by-ay*bx)/6.0f; }
        { float ax=p3x-p0x,ay=p3y-p0y,az=p3z-p0z,bx=p1x-p0x,by=p1y-p0y,bz=p1z-p0z;
            g2x=(ay*bz-az*by)/6.0f; g2y=(az*bx-ax*bz)/6.0f; g2z=(ax*by-ay*bx)/6.0f; }
        { float ax=p1x-p0x,ay=p1y-p0y,az=p1z-p0z,bx=p2x-p0x,by=p2y-p0y,bz=p2z-p0z;
            g3x=(ay*bz-az*by)/6.0f; g3y=(az*bx-ax*bz)/6.0f; g3z=(ax*by-ay*bx)/6.0f; }

        float wSum = w0*(g0x*g0x+g0y*g0y+g0z*g0z) + w1*(g1x*g1x+g1y*g1y+g1z*g1z)
                     + w2*(g2x*g2x+g2y*g2y+g2z*g2z) + w3*(g3x*g3x+g3y*g3y+g3z*g3z);
        if (wSum < 1e-10f) continue;

        float vol = ((p1x-p0x)*((p2y-p0y)*(p3z-p0z)-(p2z-p0z)*(p3y-p0y))
                     + (p1y-p0y)*((p2z-p0z)*(p3x-p0x)-(p2x-p0x)*(p3z-p0z))
                     + (p1z-p0z)*((p2x-p0x)*(p3y-p0y)-(p2y-p0y)*(p3x-p0x))) / 6.0f;

        float restVol = restVolumes[tetIdx];
        float C = vol - restVol;
        float dLambda = -(C + alpha * lambdas[tetIdx]) / (wSum + alpha);
        lambdas[tetIdx] += dLambda;

        double c0x=w0*g0x*dLambda, c0y=w0*g0y*dLambda, c0z=w0*g0z*dLambda;
        double c1x=w1*g1x*dLambda, c1y=w1*g1y*dLambda, c1z=w1*g1z*dLambda;
        double c2x=w2*g2x*dLambda, c2y=w2*g2y*dLambda, c2z=w2*g2z*dLambda;
        double c3x=w3*g3x*dLambda, c3y=w3*g3y*dLambda, c3z=w3*g3z*dLambda;

#pragma omp atomic
        corrBufferX[id0] += c0x;
#pragma omp atomic
        corrBufferY[id0] += c0y;
#pragma omp atomic
        corrBufferZ[id0] += c0z;
#pragma omp atomic
        corrCounts[id0] += 1;

#pragma omp atomic
        corrBufferX[id1] += c1x;
#pragma omp atomic
        corrBufferY[id1] += c1y;
#pragma omp atomic
        corrBufferZ[id1] += c1z;
#pragma omp atomic
        corrCounts[id1] += 1;

#pragma omp atomic
        corrBufferX[id2] += c2x;
#pragma omp atomic
        corrBufferY[id2] += c2y;
#pragma omp atomic
        corrBufferZ[id2] += c2z;
#pragma omp atomic
        corrCounts[id2] += 1;

#pragma omp atomic
        corrBufferX[id3] += c3x;
#pragma omp atomic
        corrBufferY[id3] += c3y;
#pragma omp atomic
        corrBufferZ[id3] += c3z;
#pragma omp atomic
        corrCounts[id3] += 1;
    }

    // 補正適用
// 補正適用（末尾付近）
#pragma omp parallel for
    for (int i = 0; i < numParticles; i++) {
        if (corrCounts[i] > 0) {
            positions[i * 3 + 0] += static_cast<float>(corrBufferX[i]) * jacobiScale;  // ★追加
            positions[i * 3 + 1] += static_cast<float>(corrBufferY[i]) * jacobiScale;  // ★追加
            positions[i * 3 + 2] += static_cast<float>(corrBufferZ[i]) * jacobiScale;  // ★追加
        }
    }
}

// ===========================================================================
// デバッグ出力
// ===========================================================================

void SoftBodyParallelSolver::printStats() const {
    std::cout << "  Edge color groups: " << edgeColorGroups.size() << std::endl;
    for (size_t i = 0; i < edgeColorGroups.size(); i++) {
        std::cout << "    Color " << i << ": " << edgeColorGroups[i].size() << " edges" << std::endl;
    }
    std::cout << "  Tet color groups: " << tetColorGroups.size() << std::endl;
    for (size_t i = 0; i < tetColorGroups.size(); i++) {
        std::cout << "    Color " << i << ": " << tetColorGroups[i].size() << " tets" << std::endl;
    }
}


// =============================================================================
// SoftBodyParallelSolver.cpp に追加する部分
// =============================================================================

// --- ファイル先頭のincludeに追加 ---
#include "SoftBodyGPUDuo.h"

// --- ファイル末尾に追加 ---

// ===========================================================================
// ★★★ GPULowResSolverと同じインターフェース（追加）★★★
// 既存のsolveEdges/solveVolumesをそのまま呼び出すラッパー
// ===========================================================================

bool SoftBodyParallelSolver::initialize(SoftBodyGPUDuo* softBody, float edgeComp, float volComp) {
    if (!softBody) {
        std::cerr << "[ParallelSolver] Error: softBody is null" << std::endl;
        return false;
    }

    targetBody = softBody;
    edgeCompliance = edgeComp;
    volCompliance = volComp;

    // 既存のinitialize()を呼び出す
    initialize(
        softBody->lowRes_positions.size() / 3,
        softBody->lowRes_edgeIds,
        softBody->lowRes_tetIds,
        softBody->lowRes_tetValid
        );

    return true;
}

void SoftBodyParallelSolver::preSolve(float dt, const glm::vec3& gravity) {
    if (!targetBody) return;
    targetBody->lowResPreSolve(dt, gravity);
}

void SoftBodyParallelSolver::solve(float dt) {
    if (!targetBody) return;

    // 既存のsolveEdges()をそのまま呼び出す
    solveEdges(
        targetBody->lowRes_positions,
        targetBody->lowRes_invMasses,
        targetBody->lowRes_edgeLengths,
        targetBody->lowRes_edgeLambdas,
        targetBody->lowRes_edgeIds,
        targetBody->edgeValid,
        edgeCompliance,
        dt,
        numEdgeIterations
        );

    // 既存のsolveVolumes()をそのまま呼び出す
    solveVolumes(
        targetBody->lowRes_positions,
        targetBody->lowRes_invMasses,
        targetBody->lowRes_restVols,
        targetBody->lowRes_volLambdas,
        targetBody->lowRes_tetIds,
        targetBody->lowRes_tetValid,
        volCompliance,
        dt,
        numVolumeIterations
        );
}

void SoftBodyParallelSolver::postSolve(float dt) {
    if (!targetBody) return;
    targetBody->lowResPostSolve(dt);
}

// 同期メソッド（CPUでは何もしない。GPU版との互換性のため）
void SoftBodyParallelSolver::syncFromCPU(SoftBodyGPUDuo* softBody) {
    if (softBody && softBody != targetBody) {
        targetBody = softBody;
    }
}

void SoftBodyParallelSolver::syncPositionsFromCPU(SoftBodyGPUDuo* softBody) {
    (void)softBody;
}

void SoftBodyParallelSolver::syncInvMassFromCPU(SoftBodyGPUDuo* softBody) {
    (void)softBody;
}

void SoftBodyParallelSolver::syncVelocitiesFromCPU(SoftBodyGPUDuo* softBody) {
    (void)softBody;
}

void SoftBodyParallelSolver::syncToCPU(SoftBodyGPUDuo* softBody) {
    (void)softBody;
}


// ===========================================================================
// SoftBodyParallelSolver.cpp
// 【置き換え】既存の solveEdges 関数（154〜313行目）を以下に置き換え
// ===========================================================================

void SoftBodyParallelSolver::solveEdges(
    std::vector<float>& positions,
    const std::vector<float>& invMasses,
    const std::vector<float>& restLengths,
    std::vector<float>& lambdas,
    const std::vector<int>& edgeIds,
    const std::vector<bool>& edgeValid,
    float compliance,
    float dt,
    int iterations
    ) {
    float alpha = compliance / (dt * dt);

    switch (solveType) {
    case GAUSS_SEIDEL:
        for (int iter = 0; iter < iterations; iter++) {
            if (enableStrainLimiting) {
                solveEdgesGS_StrainLimiting(positions, invMasses, restLengths, lambdas, edgeIds, edgeValid, alpha);
            } else {
                solveEdgesGS(positions, invMasses, restLengths, lambdas, edgeIds, edgeValid, alpha);
            }
        }
        break;

    case JACOBI:
        for (int iter = 0; iter < iterations; iter++) {
            solveEdgesJacobi(positions, invMasses, restLengths, lambdas, edgeIds, edgeValid, alpha);
        }
        break;

    case HYBRID: {
        int actualGsColors = std::min(gsEdgeColorCount, static_cast<int>(edgeColorGroups.size()));

        // GS部分
        for (int c = 0; c < actualGsColors; c++) {
            const std::vector<int>& colorGroup = edgeColorGroups[c];
            int groupSize = static_cast<int>(colorGroup.size());

#pragma omp parallel for
            for (int i = 0; i < groupSize; i++) {
                int edgeIdx = colorGroup[i];
                if (!edgeValid.empty() && !edgeValid[edgeIdx]) continue;

                int id0 = edgeIds[edgeIdx * 2 + 0];
                int id1 = edgeIds[edgeIdx * 2 + 1];

                float w0 = invMasses[id0];
                float w1 = invMasses[id1];
                float wSum = w0 + w1;
                if (wSum == 0.0f) continue;

                float dx = positions[id1 * 3 + 0] - positions[id0 * 3 + 0];
                float dy = positions[id1 * 3 + 1] - positions[id0 * 3 + 1];
                float dz = positions[id1 * 3 + 2] - positions[id0 * 3 + 2];
                float len = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (len < 1e-7f) continue;

                float invLen = 1.0f / len;
                dx *= invLen; dy *= invLen; dz *= invLen;

                float restLen = restLengths[edgeIdx];
                float C = len - restLen;
                float dLambda = -(C + alpha * lambdas[edgeIdx]) / (wSum + alpha);

                // ★ Strain Limiting
                if (enableStrainLimiting) {
                    float strain = len / restLen;
                    dLambda *= computeEdgeStrainScale(strain);
                }

                lambdas[edgeIdx] += dLambda;

                positions[id0 * 3 + 0] -= w0 * dx * dLambda;
                positions[id0 * 3 + 1] -= w0 * dy * dLambda;
                positions[id0 * 3 + 2] -= w0 * dz * dLambda;
                positions[id1 * 3 + 0] += w1 * dx * dLambda;
                positions[id1 * 3 + 1] += w1 * dy * dLambda;
                positions[id1 * 3 + 2] += w1 * dz * dLambda;
            }
        }

        // 残りはJacobi
        if (actualGsColors < static_cast<int>(edgeColorGroups.size())) {
            for (int jiter = 0; jiter < hybridJacobiIterations; jiter++) {
                int numP = numParticlesStored;

#pragma omp parallel for
                for (int i = 0; i < numP; i++) {
                    corrBufferX[i] = 0.0;
                    corrBufferY[i] = 0.0;
                    corrBufferZ[i] = 0.0;
                    corrCounts[i] = 0;
                }

                int numColors = static_cast<int>(edgeColorGroups.size());
                for (int c = actualGsColors; c < numColors; c++) {
                    const std::vector<int>& colorGroup = edgeColorGroups[c];
                    int groupSize = static_cast<int>(colorGroup.size());

#pragma omp parallel for
                    for (int i = 0; i < groupSize; i++) {
                        int edgeIdx = colorGroup[i];
                        if (!edgeValid.empty() && !edgeValid[edgeIdx]) continue;

                        int id0 = edgeIds[edgeIdx * 2 + 0];
                        int id1 = edgeIds[edgeIdx * 2 + 1];

                        float w0 = invMasses[id0];
                        float w1 = invMasses[id1];
                        float wSum = w0 + w1;
                        if (wSum == 0.0f) continue;

                        float dx = positions[id1 * 3 + 0] - positions[id0 * 3 + 0];
                        float dy = positions[id1 * 3 + 1] - positions[id0 * 3 + 1];
                        float dz = positions[id1 * 3 + 2] - positions[id0 * 3 + 2];
                        float len = std::sqrt(dx*dx + dy*dy + dz*dz);
                        if (len < 1e-7f) continue;

                        float invLen = 1.0f / len;
                        dx *= invLen; dy *= invLen; dz *= invLen;

                        float restLen = restLengths[edgeIdx];
                        float C = len - restLen;
                        float dLambda = -(C + alpha * lambdas[edgeIdx]) / (wSum + alpha);

                        // ★ Strain Limiting
                        if (enableStrainLimiting) {
                            float strain = len / restLen;
                            dLambda *= computeEdgeStrainScale(strain);
                        }

                        lambdas[edgeIdx] += dLambda;

                        double c0x = -w0 * dx * dLambda;
                        double c0y = -w0 * dy * dLambda;
                        double c0z = -w0 * dz * dLambda;
                        double c1x = w1 * dx * dLambda;
                        double c1y = w1 * dy * dLambda;
                        double c1z = w1 * dz * dLambda;

#pragma omp atomic
                        corrBufferX[id0] += c0x;
#pragma omp atomic
                        corrBufferY[id0] += c0y;
#pragma omp atomic
                        corrBufferZ[id0] += c0z;
#pragma omp atomic
                        corrCounts[id0] += 1;

#pragma omp atomic
                        corrBufferX[id1] += c1x;
#pragma omp atomic
                        corrBufferY[id1] += c1y;
#pragma omp atomic
                        corrBufferZ[id1] += c1z;
#pragma omp atomic
                        corrCounts[id1] += 1;
                    }
                }

#pragma omp parallel for
                for (int i = 0; i < numP; i++) {
                    if (corrCounts[i] > 0) {
                        positions[i * 3 + 0] += static_cast<float>(corrBufferX[i]) * jacobiScale;
                        positions[i * 3 + 1] += static_cast<float>(corrBufferY[i]) * jacobiScale;
                        positions[i * 3 + 2] += static_cast<float>(corrBufferZ[i]) * jacobiScale;
                    }
                }
            }
        }
        break;
    }
    }
}

// ===========================================================================
// SoftBodyParallelSolver.cpp
// 【置き換え】既存の solveVolumes 関数（463〜553行目）を以下に置き換え
// ===========================================================================

void SoftBodyParallelSolver::solveVolumes(
    std::vector<float>& positions,
    const std::vector<float>& invMasses,
    const std::vector<float>& restVolumes,
    std::vector<float>& lambdas,
    const std::vector<int>& tetIds,
    const std::vector<bool>& tetValid,
    float compliance,
    float dt,
    int iterations
    ) {
    float alpha = compliance / (dt * dt);

    switch (solveType) {
    case GAUSS_SEIDEL:
        for (int iter = 0; iter < iterations; iter++) {
            if (enableStrainLimiting) {
                solveTetsGS_StrainLimiting(positions, invMasses, restVolumes, lambdas, tetIds, tetValid, alpha);
            } else {
                solveTetsGS(positions, invMasses, restVolumes, lambdas, tetIds, tetValid, alpha);
            }
        }
        break;

    case JACOBI:
        for (int iter = 0; iter < iterations; iter++) {
            solveTetsJacobi(positions, invMasses, restVolumes, lambdas, tetIds, tetValid, alpha);
        }
        break;

    case HYBRID: {
        int actualGsColors = std::min(gsTetColorCount, static_cast<int>(tetColorGroups.size()));

        // GS部分
        for (int c = 0; c < actualGsColors; c++) {
            const std::vector<int>& colorGroup = tetColorGroups[c];
            int groupSize = static_cast<int>(colorGroup.size());

#pragma omp parallel for
            for (int i = 0; i < groupSize; i++) {
                int tetIdx = colorGroup[i];
                if (!tetValid.empty() && !tetValid[tetIdx]) continue;

                int id0 = tetIds[tetIdx * 4 + 0];
                int id1 = tetIds[tetIdx * 4 + 1];
                int id2 = tetIds[tetIdx * 4 + 2];
                int id3 = tetIds[tetIdx * 4 + 3];

                float w0 = invMasses[id0], w1 = invMasses[id1], w2 = invMasses[id2], w3 = invMasses[id3];

                float p0x = positions[id0 * 3], p0y = positions[id0 * 3 + 1], p0z = positions[id0 * 3 + 2];
                float p1x = positions[id1 * 3], p1y = positions[id1 * 3 + 1], p1z = positions[id1 * 3 + 2];
                float p2x = positions[id2 * 3], p2y = positions[id2 * 3 + 1], p2z = positions[id2 * 3 + 2];
                float p3x = positions[id3 * 3], p3y = positions[id3 * 3 + 1], p3z = positions[id3 * 3 + 2];

                float g0x, g0y, g0z, g1x, g1y, g1z, g2x, g2y, g2z, g3x, g3y, g3z;
                { float ax=p3x-p1x,ay=p3y-p1y,az=p3z-p1z,bx=p2x-p1x,by=p2y-p1y,bz=p2z-p1z;
                    g0x=(ay*bz-az*by)/6.0f; g0y=(az*bx-ax*bz)/6.0f; g0z=(ax*by-ay*bx)/6.0f; }
                { float ax=p2x-p0x,ay=p2y-p0y,az=p2z-p0z,bx=p3x-p0x,by=p3y-p0y,bz=p3z-p0z;
                    g1x=(ay*bz-az*by)/6.0f; g1y=(az*bx-ax*bz)/6.0f; g1z=(ax*by-ay*bx)/6.0f; }
                { float ax=p3x-p0x,ay=p3y-p0y,az=p3z-p0z,bx=p1x-p0x,by=p1y-p0y,bz=p1z-p0z;
                    g2x=(ay*bz-az*by)/6.0f; g2y=(az*bx-ax*bz)/6.0f; g2z=(ax*by-ay*bx)/6.0f; }
                { float ax=p1x-p0x,ay=p1y-p0y,az=p1z-p0z,bx=p2x-p0x,by=p2y-p0y,bz=p2z-p0z;
                    g3x=(ay*bz-az*by)/6.0f; g3y=(az*bx-ax*bz)/6.0f; g3z=(ax*by-ay*bx)/6.0f; }

                float wSum = w0*(g0x*g0x+g0y*g0y+g0z*g0z) + w1*(g1x*g1x+g1y*g1y+g1z*g1z)
                             + w2*(g2x*g2x+g2y*g2y+g2z*g2z) + w3*(g3x*g3x+g3y*g3y+g3z*g3z);
                if (wSum < 1e-10f) continue;

                float vol = ((p1x-p0x)*((p2y-p0y)*(p3z-p0z)-(p2z-p0z)*(p3y-p0y))
                             + (p1y-p0y)*((p2z-p0z)*(p3x-p0x)-(p2x-p0x)*(p3z-p0z))
                             + (p1z-p0z)*((p2x-p0x)*(p3y-p0y)-(p2y-p0y)*(p3x-p0x))) / 6.0f;

                float restVol = restVolumes[tetIdx];
                float C = vol - restVol;
                float dLambda = -(C + alpha * lambdas[tetIdx]) / (wSum + alpha);

                // ★ Strain Limiting
                if (enableStrainLimiting) {
                    float strain = (restVol != 0.0f) ? vol / restVol : 1.0f;
                    dLambda *= computeVolumeStrainScale(strain);
                }

                lambdas[tetIdx] += dLambda;

                positions[id0*3+0]+=w0*g0x*dLambda; positions[id0*3+1]+=w0*g0y*dLambda; positions[id0*3+2]+=w0*g0z*dLambda;
                positions[id1*3+0]+=w1*g1x*dLambda; positions[id1*3+1]+=w1*g1y*dLambda; positions[id1*3+2]+=w1*g1z*dLambda;
                positions[id2*3+0]+=w2*g2x*dLambda; positions[id2*3+1]+=w2*g2y*dLambda; positions[id2*3+2]+=w2*g2z*dLambda;
                positions[id3*3+0]+=w3*g3x*dLambda; positions[id3*3+1]+=w3*g3y*dLambda; positions[id3*3+2]+=w3*g3z*dLambda;
            }
        }

        // 残りはJacobi
        if (actualGsColors < static_cast<int>(tetColorGroups.size())) {
            for (int jiter = 0; jiter < hybridJacobiIterations; jiter++) {
                solveTetsJacobi(positions, invMasses, restVolumes, lambdas, tetIds, tetValid, alpha);
            }
        }
        break;
    }
    }
}


// ===========================================================================
// SoftBodyParallelSolver.cpp
// 【新規追加】ファイル末尾（printStats関数の後、754行目付近）に以下を追加
// ===========================================================================

// ===========================================================================
// Gauss-Seidel エッジソルバー（Strain Limiting 対応版）
// ===========================================================================

void SoftBodyParallelSolver::solveEdgesGS_StrainLimiting(
    std::vector<float>& positions,
    const std::vector<float>& invMasses,
    const std::vector<float>& restLengths,
    std::vector<float>& lambdas,
    const std::vector<int>& edgeIds,
    const std::vector<bool>& edgeValid,
    float alpha
    ) {
    int numColorGroups = static_cast<int>(edgeColorGroups.size());
    for (int c = 0; c < numColorGroups; c++) {
        const std::vector<int>& colorGroup = edgeColorGroups[c];
        int groupSize = static_cast<int>(colorGroup.size());

#pragma omp parallel for
        for (int i = 0; i < groupSize; i++) {
            int edgeIdx = colorGroup[i];
            if (!edgeValid.empty() && !edgeValid[edgeIdx]) continue;

            int id0 = edgeIds[edgeIdx * 2 + 0];
            int id1 = edgeIds[edgeIdx * 2 + 1];

            float w0 = invMasses[id0];
            float w1 = invMasses[id1];
            float wSum = w0 + w1;
            if (wSum == 0.0f) continue;

            float dx = positions[id1 * 3 + 0] - positions[id0 * 3 + 0];
            float dy = positions[id1 * 3 + 1] - positions[id0 * 3 + 1];
            float dz = positions[id1 * 3 + 2] - positions[id0 * 3 + 2];
            float len = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (len < 1e-7f) continue;

            float invLen = 1.0f / len;
            dx *= invLen; dy *= invLen; dz *= invLen;

            float restLen = restLengths[edgeIdx];

            // ★ インラインで strain を計算し、scale を決定
            float strain = len / restLen;
            float scale = computeEdgeStrainScale(strain);

            float C = len - restLen;
            float dLambda = -(C + alpha * lambdas[edgeIdx]) / (wSum + alpha);
            dLambda *= scale;  // ★ スケール適用

            lambdas[edgeIdx] += dLambda;

            positions[id0 * 3 + 0] -= w0 * dx * dLambda;
            positions[id0 * 3 + 1] -= w0 * dy * dLambda;
            positions[id0 * 3 + 2] -= w0 * dz * dLambda;
            positions[id1 * 3 + 0] += w1 * dx * dLambda;
            positions[id1 * 3 + 1] += w1 * dy * dLambda;
            positions[id1 * 3 + 2] += w1 * dz * dLambda;
        }
    }
}

// ===========================================================================
// Gauss-Seidel 四面体ソルバー（Strain Limiting 対応版）
// ===========================================================================

void SoftBodyParallelSolver::solveTetsGS_StrainLimiting(
    std::vector<float>& positions,
    const std::vector<float>& invMasses,
    const std::vector<float>& restVolumes,
    std::vector<float>& lambdas,
    const std::vector<int>& tetIds,
    const std::vector<bool>& tetValid,
    float alpha
    ) {
    int numColorGroups = static_cast<int>(tetColorGroups.size());
    for (int c = 0; c < numColorGroups; c++) {
        const std::vector<int>& colorGroup = tetColorGroups[c];
        int groupSize = static_cast<int>(colorGroup.size());

#pragma omp parallel for
        for (int i = 0; i < groupSize; i++) {
            int tetIdx = colorGroup[i];
            if (!tetValid.empty() && !tetValid[tetIdx]) continue;

            int id0 = tetIds[tetIdx * 4 + 0];
            int id1 = tetIds[tetIdx * 4 + 1];
            int id2 = tetIds[tetIdx * 4 + 2];
            int id3 = tetIds[tetIdx * 4 + 3];

            float w0 = invMasses[id0], w1 = invMasses[id1], w2 = invMasses[id2], w3 = invMasses[id3];

            float p0x = positions[id0 * 3], p0y = positions[id0 * 3 + 1], p0z = positions[id0 * 3 + 2];
            float p1x = positions[id1 * 3], p1y = positions[id1 * 3 + 1], p1z = positions[id1 * 3 + 2];
            float p2x = positions[id2 * 3], p2y = positions[id2 * 3 + 1], p2z = positions[id2 * 3 + 2];
            float p3x = positions[id3 * 3], p3y = positions[id3 * 3 + 1], p3z = positions[id3 * 3 + 2];

            float g0x, g0y, g0z, g1x, g1y, g1z, g2x, g2y, g2z, g3x, g3y, g3z;
            { float ax=p3x-p1x,ay=p3y-p1y,az=p3z-p1z,bx=p2x-p1x,by=p2y-p1y,bz=p2z-p1z;
                g0x=(ay*bz-az*by)/6.0f; g0y=(az*bx-ax*bz)/6.0f; g0z=(ax*by-ay*bx)/6.0f; }
            { float ax=p2x-p0x,ay=p2y-p0y,az=p2z-p0z,bx=p3x-p0x,by=p3y-p0y,bz=p3z-p0z;
                g1x=(ay*bz-az*by)/6.0f; g1y=(az*bx-ax*bz)/6.0f; g1z=(ax*by-ay*bx)/6.0f; }
            { float ax=p3x-p0x,ay=p3y-p0y,az=p3z-p0z,bx=p1x-p0x,by=p1y-p0y,bz=p1z-p0z;
                g2x=(ay*bz-az*by)/6.0f; g2y=(az*bx-ax*bz)/6.0f; g2z=(ax*by-ay*bx)/6.0f; }
            { float ax=p1x-p0x,ay=p1y-p0y,az=p1z-p0z,bx=p2x-p0x,by=p2y-p0y,bz=p2z-p0z;
                g3x=(ay*bz-az*by)/6.0f; g3y=(az*bx-ax*bz)/6.0f; g3z=(ax*by-ay*bx)/6.0f; }

            float wSum = w0*(g0x*g0x+g0y*g0y+g0z*g0z) + w1*(g1x*g1x+g1y*g1y+g1z*g1z)
                         + w2*(g2x*g2x+g2y*g2y+g2z*g2z) + w3*(g3x*g3x+g3y*g3y+g3z*g3z);
            if (wSum < 1e-10f) continue;

            float vol = ((p1x-p0x)*((p2y-p0y)*(p3z-p0z)-(p2z-p0z)*(p3y-p0y))
                         + (p1y-p0y)*((p2z-p0z)*(p3x-p0x)-(p2x-p0x)*(p3z-p0z))
                         + (p1z-p0z)*((p2x-p0x)*(p3y-p0y)-(p2y-p0y)*(p3x-p0x))) / 6.0f;

            float restVol = restVolumes[tetIdx];

            // ★ インラインで strain を計算し、scale を決定
            float strain = (restVol != 0.0f) ? vol / restVol : 1.0f;
            float scale = computeVolumeStrainScale(strain);

            float C = vol - restVol;
            float dLambda = -(C + alpha * lambdas[tetIdx]) / (wSum + alpha);
            dLambda *= scale;  // ★ スケール適用

            lambdas[tetIdx] += dLambda;

            positions[id0*3+0]+=w0*g0x*dLambda; positions[id0*3+1]+=w0*g0y*dLambda; positions[id0*3+2]+=w0*g0z*dLambda;
            positions[id1*3+0]+=w1*g1x*dLambda; positions[id1*3+1]+=w1*g1y*dLambda; positions[id1*3+2]+=w1*g1z*dLambda;
            positions[id2*3+0]+=w2*g2x*dLambda; positions[id2*3+1]+=w2*g2y*dLambda; positions[id2*3+2]+=w2*g2z*dLambda;
            positions[id3*3+0]+=w3*g3x*dLambda; positions[id3*3+1]+=w3*g3y*dLambda; positions[id3*3+2]+=w3*g3z*dLambda;
        }
    }
}
