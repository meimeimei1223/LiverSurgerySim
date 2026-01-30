#ifndef SOFTBODY_PARALLEL_SOLVER_H
#define SOFTBODY_PARALLEL_SOLVER_H

#include <vector>
#include <set>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <glm/glm.hpp>  // ★追加

// ===========================================================================
// CPU並列ソルバー for SoftBody
// GPULowResSolver.cppと同等のグラフカラーリング並列化を実装
// ===========================================================================

class SoftBodyGPUDuo;  // ★追加

class SoftBodyParallelSolver {
public:
    // ソルバータイプ（GPULowResSolverと同じ）
    enum SolveType {
        GAUSS_SEIDEL,   // グラフカラーリング + 同色並列GS
        JACOBI,         // 全エッジ/四面体を並列Jacobi
        HYBRID          // 最初の数色はGS、残りはJacobi
    };

    SoftBodyParallelSolver() = default;

    // 既存のinitialize（そのまま）
    void initialize(
        size_t numParticles,
        const std::vector<int>& edgeIds,
        const std::vector<int>& tetIds,
        const std::vector<bool>& tetValid
        );

    // ★★★ GPULowResSolverと同じインターフェース（追加）★★★
    bool initialize(SoftBodyGPUDuo* softBody, float edgeComp = 0.0f, float volComp = 0.0f);

    void preSolve(float dt, const glm::vec3& gravity);
    void solve(float dt);
    void postSolve(float dt);

    void syncFromCPU(SoftBodyGPUDuo* softBody);
    void syncPositionsFromCPU(SoftBodyGPUDuo* softBody);
    void syncInvMassFromCPU(SoftBodyGPUDuo* softBody);
    void syncVelocitiesFromCPU(SoftBodyGPUDuo* softBody);
    void syncToCPU(SoftBodyGPUDuo* softBody);

    void setNumIterations(int edgeIter, int volIter) {
        numEdgeIterations = edgeIter;
        numVolumeIterations = volIter;
    }

    void setCompliance(float edge, float vol) {
        edgeCompliance = edge;
        volCompliance = vol;
    }
    // ★★★ ここまで追加 ★★★

    void setSolveType(SolveType type) {
        if (solveType != type) {
            solveType = type;
            const char* name;
            switch (type) {
            case GAUSS_SEIDEL: name = "Gauss-Seidel"; break;
            case JACOBI: name = "Jacobi"; break;
            case HYBRID: name = "Hybrid"; break;
            default: name = "Unknown";
            }
            std::cout << "[ParallelSolver] Solve type: " << name << std::endl;
        }
    }

    SolveType getSolveType() const { return solveType; }

    void setHybridParams(int gsEdgeColors, int gsTetColors, int jacobiIters) {
        gsEdgeColorCount = gsEdgeColors;
        gsTetColorCount = gsTetColors;
        hybridJacobiIterations = jacobiIters;
    }

    // public セクションに追加
    float getEdgeCompliance() const { return edgeCompliance; }
    float getVolCompliance() const { return volCompliance; }

    // 既存のsolveEdges（そのまま）
    void solveEdges(
        std::vector<float>& positions,
        const std::vector<float>& invMasses,
        const std::vector<float>& restLengths,
        std::vector<float>& lambdas,
        const std::vector<int>& edgeIds,
        const std::vector<bool>& edgeValid,
        float compliance,
        float dt,
        int iterations = 1
        );

    // 既存のsolveVolumes（そのまま）
    void solveVolumes(
        std::vector<float>& positions,
        const std::vector<float>& invMasses,
        const std::vector<float>& restVolumes,
        std::vector<float>& lambdas,
        const std::vector<int>& tetIds,
        const std::vector<bool>& tetValid,
        float compliance,
        float dt,
        int iterations = 1
        );

    int getNumEdgeColors() const { return static_cast<int>(edgeColorGroups.size()); }
    int getNumTetColors() const { return static_cast<int>(tetColorGroups.size()); }
    bool isInitialized() const { return initialized; }

    void printStats() const;

    // GPULowResSolver.h / SoftBodyParallelSolver.h に追加
    void solveStep(float dt, const glm::vec3& gravity) {
        preSolve(dt, gravity);
        solve(dt);
        postSolve(dt);
    }

private:
    void computeEdgeColoring(size_t numParticles, const std::vector<int>& edgeIds);
    void computeTetColoring(size_t numParticles, const std::vector<int>& tetIds, const std::vector<bool>& tetValid);

    void solveEdgesGS(std::vector<float>& positions, const std::vector<float>& invMasses,
                      const std::vector<float>& restLengths, std::vector<float>& lambdas,
                      const std::vector<int>& edgeIds, const std::vector<bool>& edgeValid, float alpha);

    void solveEdgesJacobi(std::vector<float>& positions, const std::vector<float>& invMasses,
                          const std::vector<float>& restLengths, std::vector<float>& lambdas,
                          const std::vector<int>& edgeIds, const std::vector<bool>& edgeValid, float alpha);

    void solveTetsGS(std::vector<float>& positions, const std::vector<float>& invMasses,
                     const std::vector<float>& restVolumes, std::vector<float>& lambdas,
                     const std::vector<int>& tetIds, const std::vector<bool>& tetValid, float alpha);

    void solveTetsJacobi(std::vector<float>& positions, const std::vector<float>& invMasses,
                         const std::vector<float>& restVolumes, std::vector<float>& lambdas,
                         const std::vector<int>& tetIds, const std::vector<bool>& tetValid, float alpha);

    bool initialized = false;
    SolveType solveType = GAUSS_SEIDEL;
    int numParticlesStored = 0;

    std::vector<std::vector<int>> edgeColorGroups;
    std::vector<std::vector<int>> tetColorGroups;

    int gsEdgeColorCount = 4;
    int gsTetColorCount = 4;
    int hybridJacobiIterations = 2;

    // Jacobi用バッファ（通常のvector、#pragma omp atomicで保護）
    std::vector<double> corrBufferX;
    std::vector<double> corrBufferY;
    std::vector<double> corrBufferZ;
    std::vector<int> corrCounts;

    int numThreads = 0;

    float jacobiScale = 0.25f;

    // ★★★ 新インターフェース用（追加）★★★
    SoftBodyGPUDuo* targetBody = nullptr;
    float edgeCompliance = 0.0f;
    float volCompliance = 0.0f;
    int numEdgeIterations = 3;
    int numVolumeIterations = 2;

public:
    void setJacobiScale(float scale) {
        jacobiScale = scale;
    }




    // ===========================================================================
    // SoftBodyParallelSolver.h への追加
    //
    // 【追加場所1】53行目（setHybridParams関数の後）に以下を追加：
    // ===========================================================================

    // ★★★ Strain Limiting 設定 ★★★
    void setStrainLimitingEnabled(bool enabled) {
        enableStrainLimiting = enabled;
        std::cout << "[ParallelSolver] Strain limiting: " << (enabled ? "ON" : "OFF") << std::endl;
    }

    bool isStrainLimitingEnabled() const { return enableStrainLimiting; }

    void setEdgeStrainLimits(float soft, float hard, float max) {
        edgeStrainSoftLimit = soft;
        edgeStrainHardLimit = hard;
        edgeStrainMaxLimit = max;
    }

    void setVolumeStrainLimits(float soft, float hard, float max) {
        volStrainSoftLimit = soft;
        volStrainHardLimit = hard;
        volStrainMaxLimit = max;
    }


    // ===========================================================================
    // 【追加場所2】103行目（solveTetsJacobi宣言の後）に以下を追加：
    // ===========================================================================

    // ★ Strain Limiting 対応版（インライン計算）
    void solveEdgesGS_StrainLimiting(
        std::vector<float>& positions,
        const std::vector<float>& invMasses,
        const std::vector<float>& restLengths,
        std::vector<float>& lambdas,
        const std::vector<int>& edgeIds,
        const std::vector<bool>& edgeValid,
        float alpha);

    void solveTetsGS_StrainLimiting(
        std::vector<float>& positions,
        const std::vector<float>& invMasses,
        const std::vector<float>& restVolumes,
        std::vector<float>& lambdas,
        const std::vector<int>& tetIds,
        const std::vector<bool>& tetValid,
        float alpha);


    // ===========================================================================
    // 【追加場所3】122行目（numThreads変数の後）に以下を追加：
    // ===========================================================================

    // ★★★ Strain Limiting パラメータ ★★★
    bool enableStrainLimiting = false;

    float edgeStrainSoftLimit = 1.5f;
    float edgeStrainHardLimit = 2.5f;
    float edgeStrainMaxLimit = 3.0f;

    float volStrainSoftLimit = 1.5f;
    float volStrainHardLimit = 2.5f;
    float volStrainMaxLimit = 3.0f;

    inline float computeEdgeStrainScale(float strain) const {
        float strainRatio = std::abs(strain);
        if (strainRatio > edgeStrainMaxLimit) {
            return 0.05f;
        } else if (strainRatio > edgeStrainHardLimit) {
            float t = (strainRatio - edgeStrainHardLimit) / (edgeStrainMaxLimit - edgeStrainHardLimit);
            return 0.3f - t * 0.25f;
        } else if (strainRatio > edgeStrainSoftLimit) {
            float t = (strainRatio - edgeStrainSoftLimit) / (edgeStrainHardLimit - edgeStrainSoftLimit);
            return 1.0f - t * 0.7f;
        }
        return 1.0f;
    }

    inline float computeVolumeStrainScale(float strain) const {
        float strainRatio = (strain < 1.0f) ? 1.0f / strain : strain;
        if (strainRatio > volStrainMaxLimit) {
            return 0.05f;
        } else if (strainRatio > volStrainHardLimit) {
            float t = (strainRatio - volStrainHardLimit) / (volStrainMaxLimit - volStrainHardLimit);
            return 0.3f - t * 0.25f;
        } else if (strainRatio > volStrainSoftLimit) {
            float t = (strainRatio - volStrainSoftLimit) / (volStrainHardLimit - volStrainSoftLimit);
            return 1.0f - t * 0.7f;
        }
        return 1.0f;
    }

};

#endif // SOFTBODY_PARALLEL_SOLVER_H
