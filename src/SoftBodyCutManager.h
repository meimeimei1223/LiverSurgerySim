#ifndef SOFTBODY_CUT_MANAGER_H
#define SOFTBODY_CUT_MANAGER_H

#include "SoftBodyGPUDuo.h"
#include "MeshCuttingGPUDuo.h"
#include "SoftBodyParallelSolver.h"
#include "mCutMesh.h"
#include <iostream>
#include <algorithm>
#include <set>
#include <map>
#include <vector>
#include <string>
#include <GL/glew.h>
#include <iomanip>
//------------------------------------------------------------------------------
// カットUndo用データ構造
//------------------------------------------------------------------------------
struct CutUndoData {
    // ターゲット情報
    SoftBodyGPUDuo* target = nullptr;
    SoftBodyParallelSolver* cpuSolver = nullptr;
    std::string targetName;

    // 高解像度メッシュの状態
    std::vector<bool> highResTetValid_backup;
    std::vector<int> invalidatedHighResTets;

    // 低解像度メッシュの状態
    std::vector<bool> lowRes_tetValid_backup;
    std::vector<float> lowRes_invMasses_backup;
    std::vector<int> invalidatedLowResTets;

    // エッジ有効性
    std::vector<bool> edgeValid_backup;

    // スムージング設定
    bool wasSmooth = false;
    int smoothingIterations = 0;
    float smoothingFactor = 0.0f;
    bool enableSizeAdjust = false;
    int scalingMethod = 0;

    // カットタイプ
    enum CutType { MESH_CUT, SEGMENT_CUT } cutType = MESH_CUT;
    int cutSegmentId = -1;

    // ★★★ 子オブジェクトのアンカー状態バックアップ ★★★
    struct ChildAnchorBackup {
        SoftBodyGPUDuo* child = nullptr;
        std::vector<bool> isAnchoredToParent_backup;
        std::vector<float> lowRes_invMasses_backup;
        std::vector<float> skinningToParent_backup;
        int numAnchoredVertices_backup = 0;
    };

    std::vector<ChildAnchorBackup> childAnchors;

    // ★★★ 子自身がカットされた場合のアンカー解除情報 ★★★
    struct SelfAnchorRelease {
        std::vector<int> releasedVertices;           // 解除された頂点
        std::vector<bool> isAnchoredToParent_backup; // 元のアンカー状態
        std::vector<float> lowRes_invMasses_backup;  // 元の質量
        int numAnchoredVertices_backup = 0;
    };
    SelfAnchorRelease selfAnchorRelease;

    void clear() {
        target = nullptr;
        cpuSolver = nullptr;
        targetName.clear();
        highResTetValid_backup.clear();
        lowRes_tetValid_backup.clear();
        lowRes_invMasses_backup.clear();
        invalidatedHighResTets.clear();
        invalidatedLowResTets.clear();
        edgeValid_backup.clear();
        wasSmooth = false;
        smoothingIterations = 0;
        smoothingFactor = 0.0f;
        enableSizeAdjust = false;
        scalingMethod = 0;
        cutType = MESH_CUT;
        cutSegmentId = -1;
        childAnchors.clear();
        // ★追加
        selfAnchorRelease.releasedVertices.clear();
        selfAnchorRelease.isAnchoredToParent_backup.clear();
        selfAnchorRelease.lowRes_invMasses_backup.clear();
        selfAnchorRelease.numAnchoredVertices_backup = 0;
    }
};

//------------------------------------------------------------------------------
// カット操作マネージャー
//------------------------------------------------------------------------------
class SoftBodyCutManager {
public:
    // 履歴
    std::vector<CutUndoData> undoHistory;
    int maxHistorySize = 10;

private:
    bool needsBVHRebuild_ = true;  // BVH再構築フラグ

public:
    // カットモードに入る時に呼ぶ（main.cppのKeyXハンドラから）
    void enterCutMode() {
        needsBVHRebuild_ = true;
    }


    //==============================================================================
    // performMeshCut 完全版
    //
    // 機能:
    // - ConnectedComponentsでhighResTetAdjacencyキャッシュを使用
    // - fragmentThreshold = -1: 最大クラスタ以外全て削除
    // - fragmentThreshold = 0: フラグメント削除なし
    // - fragmentThreshold > 0: threshold以下のフラグメントを削除
    //
    // 【置き換え場所】SoftBodyCutManager.h の performMeshCut 関数全体
    //==============================================================================

    bool performMeshCut(
        SoftBodyGPUDuo* target,
        SoftBodyParallelSolver* cpuSolver,
        const mCutMesh* cutterMesh,
        const std::string& targetName,
        int isolatedRemovalMode,
        int fragmentThreshold,
        int& dampingFrameOut,
        int postCutDampingFrames,
        const std::vector<SoftBodyGPUDuo*>& childObjects = {})
    {
        auto totalStart = std::chrono::high_resolution_clock::now();

        if (!target || !cutterMesh) {
            std::cout << "Invalid target or cutter mesh" << std::endl;
            return false;
        }

        std::cout << "\n=== STARTING CUT OPERATION (" << targetName << ") ===" << std::endl;

        // ====== タイミング変数 ======
        double timeBackup = 0, timeIntersection = 0, timeBoundary = 0;
        double timeInvalidateHigh = 0, timeConnectedComponents = 0;
        double timeCutBoundaryVerts = 0, timeUpdateHighMesh = 0;
        double timeLowResPropagation = 0, timeInvalidateLow = 0;
        double timeHandleGroups = 0, timeEdgeValidity = 0;
        double timeSmoothing = 0, timeFinalize = 0;

        // ====== Step 0: Undoデータのバックアップ ======
        auto t0 = std::chrono::high_resolution_clock::now();

        CutUndoData undoData;
        undoData.target = target;
        undoData.cpuSolver = cpuSolver;
        undoData.targetName = targetName;
        undoData.cutType = CutUndoData::MESH_CUT;
        undoData.highResTetValid_backup = target->highResTetValid;
        undoData.lowRes_tetValid_backup = target->lowRes_tetValid;
        undoData.lowRes_invMasses_backup = target->lowRes_invMasses;
        undoData.edgeValid_backup = target->edgeValid;
        undoData.wasSmooth = target->smoothDisplayMode;
        undoData.smoothingIterations = target->smoothingIterations;
        undoData.smoothingFactor = target->smoothingFactor;
        undoData.enableSizeAdjust = target->enableSizeAdjustment;
        undoData.scalingMethod = target->scalingMethod;

        // 子オブジェクトのアンカー状態をバックアップ
        for (SoftBodyGPUDuo* child : childObjects) {
            if (child && child->parentSoftBody == target) {
                CutUndoData::ChildAnchorBackup childBackup;
                childBackup.child = child;
                childBackup.isAnchoredToParent_backup = child->isAnchoredToParent;
                childBackup.lowRes_invMasses_backup = child->lowRes_invMasses;
                childBackup.skinningToParent_backup = child->skinningToParent;
                childBackup.numAnchoredVertices_backup = child->numAnchoredVertices;
                undoData.childAnchors.push_back(childBackup);
            }
        }

        // ターゲット自身が子メッシュの場合のアンカーバックアップ
        if (target->hasParentSoftBody()) {
            undoData.selfAnchorRelease.isAnchoredToParent_backup = target->isAnchoredToParent;
            undoData.selfAnchorRelease.lowRes_invMasses_backup = target->lowRes_invMasses;
            undoData.selfAnchorRelease.numAnchoredVertices_backup = target->numAnchoredVertices;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        timeBackup = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // スムージング状態を保存
        bool wasSmooth = target->smoothDisplayMode;
        int savedIterations = target->smoothingIterations;
        float savedFactor = target->smoothingFactor;
        bool savedSizeAdjust = target->enableSizeAdjustment;
        int savedScalingMethod = target->scalingMethod;

        // OpenGLエラーをクリア
        while (glGetError() != GL_NO_ERROR) {}

        // ====== Step 1: データ準備 ======
        SoftBodyGPUDuo::MeshData cutterData;
        cutterData.verts = cutterMesh->mVertices;
        cutterData.tetSurfaceTriIds = std::vector<int>(cutterMesh->mIndices.begin(), cutterMesh->mIndices.end());

        SoftBodyGPUDuo::MeshData highResData;
        highResData.verts = target->highRes_positions;
        highResData.tetIds = target->highResMeshData.tetIds;
        highResData.tetEdgeIds = target->highResMeshData.tetEdgeIds;
        highResData.tetSurfaceTriIds = target->highResMeshData.tetSurfaceTriIds;

        if (!target->tetMappingComputed) {
            target->computeTetToTetMappingLowToHigh();
        }

        size_t numHighResTets = highResData.tetIds.size() / 4;

        // ====== Step 2: 交差判定（AdjProp版）======
        auto t2 = std::chrono::high_resolution_clock::now();

        std::vector<int> intersectingHighResTets;
        const auto& adjacency = target->getHighResTetAdjacency();

        // まずAdjPropで高速検索
        intersectingHighResTets =
            MeshCuttingGPUDuo::findIntersectingTetrahedraBySurfaceTriangles_AdjProp(
                cutterData,
                highResData,
                target->highResSurfaceTriToTet,
                target->highRes_positions,
                target->highResTetValid,
                adjacency,
                3,
                false);  // verbose=false

        // AdjPropで検出できなかった場合、全検索でフォールバック
        if (intersectingHighResTets.empty()) {
            std::cout << "  [Fallback] AdjProp found nothing, trying full search..." << std::endl;

            // デバッグ: 有効な四面体数を確認
            int validTetCount = 0;
            for (size_t i = 0; i < target->highResTetValid.size(); i++) {
                if (target->highResTetValid[i]) validTetCount++;
            }
            std::cout << "  [DEBUG] Valid tets: " << validTetCount << " / " << target->highResTetValid.size() << std::endl;

            intersectingHighResTets =
                MeshCuttingGPUDuo::findIntersectingTetrahedraBySurfaceTriangles(
                    cutterData,
                    highResData,
                    3,      // intersectionMode = 3 (SURFACE + SAMPLED-INTERNAL)
                    false);

            if (!intersectingHighResTets.empty()) {
                std::cout << "  [Fallback] Full search found " << intersectingHighResTets.size() << " tets" << std::endl;
            }
        }

        auto t3 = std::chrono::high_resolution_clock::now();
        timeIntersection = std::chrono::duration<double, std::milli>(t3 - t2).count();

        if (intersectingHighResTets.empty()) {
            std::cout << "No intersection detected" << std::endl;
            return false;
        }

        std::cout << "  Intersection: " << intersectingHighResTets.size() << " tets, " << timeIntersection << " ms" << std::endl;

        // ====== Step 3: 境界付近の四面体も追加 ======
        auto t4 = std::chrono::high_resolution_clock::now();

        std::vector<int> allHighResTetsToInvalidate = intersectingHighResTets;

        if (isolatedRemovalMode > 0) {
            std::set<int> boundaryVertices;
            for (int tetIdx : intersectingHighResTets) {
                for (int j = 0; j < 4; j++) {
                    boundaryVertices.insert(highResData.tetIds[tetIdx * 4 + j]);
                }
            }

            for (size_t i = 0; i < numHighResTets; i++) {
                if (std::find(allHighResTetsToInvalidate.begin(),
                              allHighResTetsToInvalidate.end(), i) != allHighResTetsToInvalidate.end()) {
                    continue;
                }
                for (int j = 0; j < 4; j++) {
                    if (boundaryVertices.count(highResData.tetIds[i * 4 + j]) > 0) {
                        allHighResTetsToInvalidate.push_back(i);
                        break;
                    }
                }
            }
        }

        auto t5 = std::chrono::high_resolution_clock::now();
        timeBoundary = std::chrono::duration<double, std::milli>(t5 - t4).count();
        std::cout << "  Boundary expansion: " << timeBoundary << " ms" << std::endl;

        // ====== Step 4: HighRes四面体を無効化 ======
        auto t6 = std::chrono::high_resolution_clock::now();

        target->invalidateHighResTetrahedra(allHighResTetsToInvalidate);

        auto t7 = std::chrono::high_resolution_clock::now();
        timeInvalidateHigh = std::chrono::duration<double, std::milli>(t7 - t6).count();
        std::cout << "  InvalidateHighRes: " << allHighResTetsToInvalidate.size() << " tets, " << timeInvalidateHigh << " ms" << std::endl;

        // ====== Step 5: 連結成分を検出してフラグメント削除（最適化版）======
        auto t8 = std::chrono::high_resolution_clock::now();

        // adjacencyキャッシュを使用してBFS
        std::vector<std::vector<int>> components;
        std::vector<bool> visited(numHighResTets, false);

        int validTetCountForCC = 0;
        for (size_t i = 0; i < numHighResTets; i++) {
            if (target->highResTetValid[i]) validTetCountForCC++;
        }

        std::cout << "  Building connected components from " << validTetCountForCC << " valid tets..." << std::endl;

        // 有効な四面体のみでBFS
        for (size_t startTet = 0; startTet < numHighResTets; startTet++) {
            if (!target->highResTetValid[startTet] || visited[startTet]) continue;

            std::vector<int> component;
            std::queue<int> bfsQueue;
            bfsQueue.push(startTet);
            visited[startTet] = true;

            while (!bfsQueue.empty()) {
                int currentTet = bfsQueue.front();
                bfsQueue.pop();
                component.push_back(currentTet);

                // 隣接する四面体を探索（キャッシュから）
                for (int neighbor : adjacency[currentTet]) {
                    // 有効な四面体のみ探索（INVALIDを除外）
                    if (target->highResTetValid[neighbor] && !visited[neighbor]) {
                        visited[neighbor] = true;
                        bfsQueue.push(neighbor);
                    }
                }
            }

            components.push_back(std::move(component));
        }

        std::cout << "  Found " << components.size() << " connected components:" << std::endl;

        // サイズ順に表示
        std::vector<std::pair<size_t, size_t>> sizeIndex;
        for (size_t i = 0; i < components.size(); i++) {
            sizeIndex.push_back({components[i].size(), i});
        }
        std::sort(sizeIndex.rbegin(), sizeIndex.rend());

        for (size_t i = 0; i < std::min(size_t(10), sizeIndex.size()); i++) {
            std::cout << "    Component " << i << ": " << sizeIndex[i].first << " tets" << std::endl;
        }
        if (sizeIndex.size() > 10) {
            std::cout << "    ... and " << (sizeIndex.size() - 10) << " more components" << std::endl;
        }

        // 最大コンポーネント以外のフラグメントを削除
        if (!components.empty()) {
            size_t largestComponentIdx = sizeIndex[0].second;
            size_t largestSize = sizeIndex[0].first;

            std::vector<int> fragmentTetsToRemove;
            for (size_t i = 0; i < components.size(); i++) {
                if (i == largestComponentIdx) continue;

                // fragmentThreshold == -1: 最大クラスタ以外全て削除
                // fragmentThreshold == 0: フラグメント削除なし
                // fragmentThreshold > 0: threshold以下のフラグメントを削除
                if (fragmentThreshold == -1 ||
                    (fragmentThreshold > 0 && components[i].size() <= static_cast<size_t>(fragmentThreshold))) {
                    fragmentTetsToRemove.insert(fragmentTetsToRemove.end(),
                                                components[i].begin(),
                                                components[i].end());
                }
            }

            if (!fragmentTetsToRemove.empty()) {
                target->invalidateHighResTetrahedra(fragmentTetsToRemove);
                allHighResTetsToInvalidate.insert(allHighResTetsToInvalidate.end(),
                                                  fragmentTetsToRemove.begin(),
                                                  fragmentTetsToRemove.end());

                if (fragmentThreshold == -1) {
                    std::cout << "  Removed " << fragmentTetsToRemove.size()
                              << " tets (keeping only largest component)" << std::endl;
                } else {
                    std::cout << "  Removed " << fragmentTetsToRemove.size()
                              << " fragment tets (threshold: " << fragmentThreshold << ")" << std::endl;
                }
            }
        }

        auto t9 = std::chrono::high_resolution_clock::now();
        timeConnectedComponents = std::chrono::duration<double, std::milli>(t9 - t8).count();
        std::cout << "  ConnectedComponents: " << timeConnectedComponents << " ms" << std::endl;

        undoData.invalidatedHighResTets = allHighResTetsToInvalidate;

        // ====== Step 6: カット境界頂点を検出 ======
        auto t10 = std::chrono::high_resolution_clock::now();

        std::set<int> cutBoundaryVertices;
        std::map<int, std::pair<int, int>> vertexTetCounts;
        for (size_t i = 0; i < numHighResTets; i++) {
            bool isValid = target->highResTetValid[i];
            for (int j = 0; j < 4; j++) {
                int vid = highResData.tetIds[i * 4 + j];
                if (vertexTetCounts.find(vid) == vertexTetCounts.end()) {
                    vertexTetCounts[vid] = {0, 0};
                }
                if (isValid) {
                    vertexTetCounts[vid].first++;
                } else {
                    vertexTetCounts[vid].second++;
                }
            }
        }
        for (const auto& entry : vertexTetCounts) {
            if (entry.second.first > 0 && entry.second.second > 0) {
                cutBoundaryVertices.insert(entry.first);
            }
        }
        target->setCutBoundaryVertices(cutBoundaryVertices);

        auto t11 = std::chrono::high_resolution_clock::now();
        timeCutBoundaryVerts = std::chrono::duration<double, std::milli>(t11 - t10).count();
        std::cout << "  CutBoundaryVerts: " << cutBoundaryVertices.size() << " verts, " << timeCutBoundaryVerts << " ms" << std::endl;

        // ====== Step 7: HighResメッシュバッファを更新 ======
        auto t12 = std::chrono::high_resolution_clock::now();

        target->updateHighResMesh();
        target->updateHighResTetMesh();
        target->computeHighResNormals();

        auto t13 = std::chrono::high_resolution_clock::now();
        timeUpdateHighMesh = std::chrono::duration<double, std::milli>(t13 - t12).count();
        std::cout << "  UpdateHighResMesh: " << timeUpdateHighMesh << " ms" << std::endl;

        // ====== Step 8: LowRes四面体への伝播 ======
        auto t14 = std::chrono::high_resolution_clock::now();

        std::set<int> lowResTetsToInvalidate;

        for (size_t lowTetIdx = 0; lowTetIdx < target->numLowTets; lowTetIdx++) {
            if (!target->lowRes_tetValid[lowTetIdx]) continue;

            std::set<int> correspondingHighTets = target->getHighResTetsFromLowResTet(lowTetIdx);
            if (correspondingHighTets.empty()) continue;

            int totalCorrespondingHighTets = correspondingHighTets.size();
            int invalidHighTets = 0;

            for (int highTetIdx : correspondingHighTets) {
                if (highTetIdx >= 0 && highTetIdx < static_cast<int>(target->highResTetValid.size())) {
                    if (!target->highResTetValid[highTetIdx]) {
                        invalidHighTets++;
                    }
                }
            }

            if (invalidHighTets > 0 && invalidHighTets == totalCorrespondingHighTets) {
                lowResTetsToInvalidate.insert(lowTetIdx);
            }
        }

        std::vector<int> lowTetsVector(lowResTetsToInvalidate.begin(), lowResTetsToInvalidate.end());
        undoData.invalidatedLowResTets = lowTetsVector;

        auto t15 = std::chrono::high_resolution_clock::now();
        timeLowResPropagation = std::chrono::duration<double, std::milli>(t15 - t14).count();
        std::cout << "  LowResPropagation: " << lowTetsVector.size() << " tets, " << timeLowResPropagation << " ms" << std::endl;

        // ====== Step 9: LowRes無効化とハンドルグループ ======
        if (!lowTetsVector.empty()) {
            auto t16 = std::chrono::high_resolution_clock::now();

            target->invalidateLowResTetrahedra(lowTetsVector);

            auto t16_1 = std::chrono::high_resolution_clock::now();
            double timeInvalidateOnly = std::chrono::duration<double, std::milli>(t16_1 - t16).count();

            target->updateLowResMesh();

            auto t16_2 = std::chrono::high_resolution_clock::now();
            double timeUpdateLowMesh = std::chrono::duration<double, std::milli>(t16_2 - t16_1).count();

            target->updateLowResTetMeshes();

            auto t16_3 = std::chrono::high_resolution_clock::now();
            double timeUpdateLowTetMesh = std::chrono::duration<double, std::milli>(t16_3 - t16_2).count();

            std::cout << "  InvalidateLowRes breakdown:" << std::endl;
            std::cout << "    invalidateLowResTetrahedra: " << timeInvalidateOnly << " ms" << std::endl;
            std::cout << "    updateLowResMesh: " << timeUpdateLowMesh << " ms" << std::endl;
            std::cout << "    updateLowResTetMeshes: " << timeUpdateLowTetMesh << " ms" << std::endl;

            // ターゲットが子メッシュの場合、孤立した頂点のアンカーを解除
            if (target->hasParentSoftBody()) {
                std::vector<int> releasedVerts = target->releaseAnchorsForOrphanedVertices();
                undoData.selfAnchorRelease.releasedVertices = releasedVerts;
            }

            auto t17 = std::chrono::high_resolution_clock::now();
            timeInvalidateLow = std::chrono::duration<double, std::milli>(t17 - t16).count();
            std::cout << "  InvalidateLowRes: " << timeInvalidateLow << " ms" << std::endl;

            // ハンドルグループのチェック
            auto t18 = std::chrono::high_resolution_clock::now();

            if (!target->handleGroups.empty()) {
                std::vector<int> groupsToRemove;

                for (size_t g = 0; g < target->handleGroups.size(); g++) {
                    const auto& group = target->handleGroups[g];
                    int centerVert = group.centerVertex;

                    bool centerHasValidTet = false;
                    for (size_t t = 0; t < target->numLowTets; t++) {
                        if (!target->lowRes_tetValid[t]) continue;

                        for (int j = 0; j < 4; j++) {
                            if (target->lowRes_tetIds[t * 4 + j] == centerVert) {
                                centerHasValidTet = true;
                                break;
                            }
                        }
                        if (centerHasValidTet) break;
                    }

                    if (!centerHasValidTet) {
                        groupsToRemove.push_back(g);
                    }
                }

                std::sort(groupsToRemove.begin(), groupsToRemove.end(), std::greater<int>());
                for (int idx : groupsToRemove) {
                    target->removeHandleGroup(idx);
                }
            }

            auto t19 = std::chrono::high_resolution_clock::now();
            timeHandleGroups = std::chrono::duration<double, std::milli>(t19 - t18).count();
            std::cout << "  HandleGroups: " << timeHandleGroups << " ms" << std::endl;

            // エッジの有効性を更新（最適化版）
            auto t20 = std::chrono::high_resolution_clock::now();

            size_t numEdges = target->lowRes_edgeIds.size() / 2;
            target->edgeValid.resize(numEdges);
            std::fill(target->edgeValid.begin(), target->edgeValid.end(), false);

            // 有効な四面体から頂点ペアのセットを構築 O(numValidTets × 6)
            std::set<std::pair<int, int>> validEdgePairs;

            for (size_t t = 0; t < target->numLowTets; t++) {
                if (!target->lowRes_tetValid[t]) continue;

                int v0 = target->lowRes_tetIds[t * 4 + 0];
                int v1 = target->lowRes_tetIds[t * 4 + 1];
                int v2 = target->lowRes_tetIds[t * 4 + 2];
                int v3 = target->lowRes_tetIds[t * 4 + 3];

                validEdgePairs.insert({std::min(v0, v1), std::max(v0, v1)});
                validEdgePairs.insert({std::min(v0, v2), std::max(v0, v2)});
                validEdgePairs.insert({std::min(v0, v3), std::max(v0, v3)});
                validEdgePairs.insert({std::min(v1, v2), std::max(v1, v2)});
                validEdgePairs.insert({std::min(v1, v3), std::max(v1, v3)});
                validEdgePairs.insert({std::min(v2, v3), std::max(v2, v3)});
            }

            // 各エッジの有効性をO(log N)でチェック
            for (size_t i = 0; i < numEdges; i++) {
                int id0 = target->lowRes_edgeIds[2 * i];
                int id1 = target->lowRes_edgeIds[2 * i + 1];
                target->edgeValid[i] = (validEdgePairs.count({std::min(id0, id1), std::max(id0, id1)}) > 0);
            }

            auto t21 = std::chrono::high_resolution_clock::now();
            timeEdgeValidity = std::chrono::duration<double, std::milli>(t21 - t20).count();
            std::cout << "  EdgeValidity: " << timeEdgeValidity << " ms" << std::endl;
        }

        // ====== Step 10: スムージングをリセット ======
        auto t22 = std::chrono::high_resolution_clock::now();

        target->invalidateSmoothingCache();

        if (wasSmooth) {
            target->enableSmoothDisplay(false);
            while (glGetError() != GL_NO_ERROR) {}
            target->enableSmoothDisplay(true);

            if (target->smoothDisplayMode) {
                target->setSmoothingParameters(savedIterations, savedFactor,
                                               savedSizeAdjust, savedScalingMethod);
                target->updateSmoothBuffers();
                GLenum err = glGetError();
                if (err != GL_NO_ERROR) {
                    target->enableSmoothDisplay(false);
                    target->enableSmoothDisplay(true);
                }
            }
        }

        auto t23 = std::chrono::high_resolution_clock::now();
        timeSmoothing = std::chrono::duration<double, std::milli>(t23 - t22).count();
        std::cout << "  Smoothing: " << timeSmoothing << " ms" << std::endl;

        // ====== Step 11: 履歴に追加 ======
        auto t24 = std::chrono::high_resolution_clock::now();

        if (static_cast<int>(undoHistory.size()) >= maxHistorySize) {
            undoHistory.erase(undoHistory.begin());
        }
        undoHistory.push_back(undoData);

        dampingFrameOut = postCutDampingFrames;

        auto t25 = std::chrono::high_resolution_clock::now();
        timeFinalize = std::chrono::duration<double, std::milli>(t25 - t24).count();

        reinitializeSolver(target, cpuSolver);  // ← 自動で呼ぶ

        // ====== 合計時間と内訳 ======
        auto totalEnd = std::chrono::high_resolution_clock::now();
        double totalTime = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

        std::cout << "\n========== " << targetName << " CUT TIMING BREAKDOWN ==========" << std::endl;
        std::cout << "  Backup:              " << std::fixed << std::setprecision(2) << timeBackup << " ms" << std::endl;
        std::cout << "  Intersection:        " << timeIntersection << " ms" << std::endl;
        std::cout << "  Boundary:            " << timeBoundary << " ms" << std::endl;
        std::cout << "  InvalidateHighRes:   " << timeInvalidateHigh << " ms" << std::endl;
        std::cout << "  ConnectedComponents: " << timeConnectedComponents << " ms" << std::endl;
        std::cout << "  CutBoundaryVerts:    " << timeCutBoundaryVerts << " ms" << std::endl;
        std::cout << "  UpdateHighResMesh:   " << timeUpdateHighMesh << " ms" << std::endl;
        std::cout << "  LowResPropagation:   " << timeLowResPropagation << " ms" << std::endl;
        std::cout << "  InvalidateLowRes:    " << timeInvalidateLow << " ms" << std::endl;
        std::cout << "  HandleGroups:        " << timeHandleGroups << " ms" << std::endl;
        std::cout << "  EdgeValidity:        " << timeEdgeValidity << " ms" << std::endl;
        std::cout << "  Smoothing:           " << timeSmoothing << " ms" << std::endl;
        std::cout << "  Finalize:            " << timeFinalize << " ms" << std::endl;
        std::cout << "  ----------------------------------------" << std::endl;
        std::cout << "  TOTAL:               " << totalTime << " ms" << std::endl;
        std::cout << "================================================\n" << std::endl;

        return true;
    }



private:
    void reinitializeSolver(SoftBodyGPUDuo* body, SoftBodyParallelSolver* solver) {
        if (solver && body) {
            solver->initialize(body, solver->getEdgeCompliance(), solver->getVolCompliance());
            std::cout << "CPU Parallel Solver re-initialized after cut" << std::endl;
        }
    }

    // bool performMeshCut(
    //     SoftBodyGPUDuo* target,
    //     SoftBodyParallelSolver* cpuSolver,
    //     const mCutMesh* cutterMesh,
    //     const std::string& targetName,
    //     int isolatedRemovalMode,
    //     int fragmentThreshold,
    //     int& dampingFrameOut,
    //     int postCutDampingFrames,
    //     const std::vector<SoftBodyGPUDuo*>& childObjects = {})
    // {
    //     auto totalStart = std::chrono::high_resolution_clock::now();

    //     if (!target || !cutterMesh) {
    //         std::cout << "Invalid target or cutter mesh" << std::endl;
    //         return false;
    //     }

    //     std::cout << "\n=== STARTING CUT OPERATION (" << targetName << ") ===" << std::endl;

    //     // ====== タイミング変数 ======
    //     double timeBackup = 0, timeIntersection = 0, timeBoundary = 0;
    //     double timeInvalidateHigh = 0, timeConnectedComponents = 0;
    //     double timeCutBoundaryVerts = 0, timeUpdateHighMesh = 0;
    //     double timeLowResPropagation = 0, timeInvalidateLow = 0;
    //     double timeHandleGroups = 0, timeEdgeValidity = 0;
    //     double timeSmoothing = 0, timeFinalize = 0;

    //     // ====== Step 0: Undoデータのバックアップ ======
    //     auto t0 = std::chrono::high_resolution_clock::now();

    //     CutUndoData undoData;
    //     undoData.target = target;
    //     undoData.cpuSolver = cpuSolver;
    //     undoData.targetName = targetName;
    //     undoData.cutType = CutUndoData::MESH_CUT;
    //     undoData.highResTetValid_backup = target->highResTetValid;
    //     undoData.lowRes_tetValid_backup = target->lowRes_tetValid;
    //     undoData.lowRes_invMasses_backup = target->lowRes_invMasses;
    //     undoData.edgeValid_backup = target->edgeValid;
    //     undoData.wasSmooth = target->smoothDisplayMode;
    //     undoData.smoothingIterations = target->smoothingIterations;
    //     undoData.smoothingFactor = target->smoothingFactor;
    //     undoData.enableSizeAdjust = target->enableSizeAdjustment;
    //     undoData.scalingMethod = target->scalingMethod;

    //     // 子オブジェクトのアンカー状態をバックアップ
    //     for (SoftBodyGPUDuo* child : childObjects) {
    //         if (child && child->parentSoftBody == target) {
    //             CutUndoData::ChildAnchorBackup childBackup;
    //             childBackup.child = child;
    //             childBackup.isAnchoredToParent_backup = child->isAnchoredToParent;
    //             childBackup.lowRes_invMasses_backup = child->lowRes_invMasses;
    //             childBackup.skinningToParent_backup = child->skinningToParent;
    //             childBackup.numAnchoredVertices_backup = child->numAnchoredVertices;
    //             undoData.childAnchors.push_back(childBackup);
    //         }
    //     }

    //     // ターゲット自身が子メッシュの場合のアンカーバックアップ
    //     if (target->hasParentSoftBody()) {
    //         undoData.selfAnchorRelease.isAnchoredToParent_backup = target->isAnchoredToParent;
    //         undoData.selfAnchorRelease.lowRes_invMasses_backup = target->lowRes_invMasses;
    //         undoData.selfAnchorRelease.numAnchoredVertices_backup = target->numAnchoredVertices;
    //     }

    //     auto t1 = std::chrono::high_resolution_clock::now();
    //     timeBackup = std::chrono::duration<double, std::milli>(t1 - t0).count();

    //     // スムージング状態を保存
    //     bool wasSmooth = target->smoothDisplayMode;
    //     int savedIterations = target->smoothingIterations;
    //     float savedFactor = target->smoothingFactor;
    //     bool savedSizeAdjust = target->enableSizeAdjustment;
    //     int savedScalingMethod = target->scalingMethod;

    //     // OpenGLエラーをクリア
    //     while (glGetError() != GL_NO_ERROR) {}

    //     // ====== Step 1: データ準備 ======
    //     SoftBodyGPUDuo::MeshData cutterData;
    //     cutterData.verts = cutterMesh->mVertices;
    //     cutterData.tetSurfaceTriIds = std::vector<int>(cutterMesh->mIndices.begin(), cutterMesh->mIndices.end());

    //     SoftBodyGPUDuo::MeshData highResData;
    //     highResData.verts = target->highRes_positions;
    //     highResData.tetIds = target->highResMeshData.tetIds;
    //     highResData.tetEdgeIds = target->highResMeshData.tetEdgeIds;
    //     highResData.tetSurfaceTriIds = target->highResMeshData.tetSurfaceTriIds;

    //     if (!target->tetMappingComputed) {
    //         target->computeTetToTetMappingLowToHigh();
    //     }

    //     size_t numHighResTets = highResData.tetIds.size() / 4;

    //     // ====== Step 2: 交差判定（AdjProp版）======
    //     auto t2 = std::chrono::high_resolution_clock::now();

    //     std::vector<int> intersectingHighResTets;
    //     const auto& adjacency = target->getHighResTetAdjacency();

    //     // まずAdjPropで高速検索
    //     intersectingHighResTets =
    //         MeshCuttingGPUDuo::findIntersectingTetrahedraBySurfaceTriangles_AdjProp(
    //             cutterData,
    //             highResData,
    //             target->highResSurfaceTriToTet,
    //             target->highRes_positions,
    //             target->highResTetValid,
    //             adjacency,
    //             3,
    //             false);  // verbose=false

    //     // AdjPropで検出できなかった場合、全検索でフォールバック
    //     if (intersectingHighResTets.empty()) {
    //         std::cout << "  [Fallback] AdjProp found nothing, trying full search..." << std::endl;

    //         // デバッグ: 有効な四面体数を確認
    //         int validTetCount = 0;
    //         for (size_t i = 0; i < target->highResTetValid.size(); i++) {
    //             if (target->highResTetValid[i]) validTetCount++;
    //         }
    //         std::cout << "  [DEBUG] Valid tets: " << validTetCount << " / " << target->highResTetValid.size() << std::endl;

    //         intersectingHighResTets =
    //             MeshCuttingGPUDuo::findIntersectingTetrahedraBySurfaceTriangles(
    //                 cutterData,
    //                 highResData,
    //                 3,      // intersectionMode = 3 (SURFACE + SAMPLED-INTERNAL)
    //                 false);

    //         if (!intersectingHighResTets.empty()) {
    //             std::cout << "  [Fallback] Full search found " << intersectingHighResTets.size() << " tets" << std::endl;
    //         }
    //     }

    //     auto t3 = std::chrono::high_resolution_clock::now();
    //     timeIntersection = std::chrono::duration<double, std::milli>(t3 - t2).count();

    //     if (intersectingHighResTets.empty()) {
    //         std::cout << "No intersection detected" << std::endl;
    //         return false;
    //     }

    //     std::cout << "  Intersection: " << intersectingHighResTets.size() << " tets, " << timeIntersection << " ms" << std::endl;

    //     // ====== Step 3: 境界付近の四面体も追加 ======
    //     auto t4 = std::chrono::high_resolution_clock::now();

    //     std::vector<int> allHighResTetsToInvalidate = intersectingHighResTets;

    //     if (isolatedRemovalMode > 0) {
    //         std::set<int> boundaryVertices;
    //         for (int tetIdx : intersectingHighResTets) {
    //             for (int j = 0; j < 4; j++) {
    //                 boundaryVertices.insert(highResData.tetIds[tetIdx * 4 + j]);
    //             }
    //         }

    //         for (size_t i = 0; i < numHighResTets; i++) {
    //             if (std::find(allHighResTetsToInvalidate.begin(),
    //                           allHighResTetsToInvalidate.end(), i) != allHighResTetsToInvalidate.end()) {
    //                 continue;
    //             }
    //             for (int j = 0; j < 4; j++) {
    //                 if (boundaryVertices.count(highResData.tetIds[i * 4 + j]) > 0) {
    //                     allHighResTetsToInvalidate.push_back(i);
    //                     break;
    //                 }
    //             }
    //         }
    //     }

    //     auto t5 = std::chrono::high_resolution_clock::now();
    //     timeBoundary = std::chrono::duration<double, std::milli>(t5 - t4).count();
    //     std::cout << "  Boundary expansion: " << timeBoundary << " ms" << std::endl;

    //     // ====== Step 4: HighRes四面体を無効化 ======
    //     auto t6 = std::chrono::high_resolution_clock::now();

    //     target->invalidateHighResTetrahedra(allHighResTetsToInvalidate);

    //     auto t7 = std::chrono::high_resolution_clock::now();
    //     timeInvalidateHigh = std::chrono::duration<double, std::milli>(t7 - t6).count();
    //     std::cout << "  InvalidateHighRes: " << allHighResTetsToInvalidate.size() << " tets, " << timeInvalidateHigh << " ms" << std::endl;

    //     // ====== Step 5: 連結成分を検出してフラグメント削除（最適化版）======
    //     auto t8 = std::chrono::high_resolution_clock::now();

    //     // ★★★ 最適化: adjacencyキャッシュを使用してBFS ★★★
    //     std::vector<std::vector<int>> components;
    //     std::vector<bool> visited(numHighResTets, false);

    //     int validTetCountForCC = 0;
    //     for (size_t i = 0; i < numHighResTets; i++) {
    //         if (target->highResTetValid[i]) validTetCountForCC++;
    //     }

    //     std::cout << "  Building connected components from " << validTetCountForCC << " valid tets..." << std::endl;

    //     // 有効な四面体のみでBFS
    //     for (size_t startTet = 0; startTet < numHighResTets; startTet++) {
    //         if (!target->highResTetValid[startTet] || visited[startTet]) continue;

    //         std::vector<int> component;
    //         std::queue<int> bfsQueue;
    //         bfsQueue.push(startTet);
    //         visited[startTet] = true;

    //         while (!bfsQueue.empty()) {
    //             int currentTet = bfsQueue.front();
    //             bfsQueue.pop();
    //             component.push_back(currentTet);

    //             // 隣接する四面体を探索（キャッシュから）
    //             for (int neighbor : adjacency[currentTet]) {
    //                 // ★ 有効な四面体のみ探索（INVALIDを除外）
    //                 if (target->highResTetValid[neighbor] && !visited[neighbor]) {
    //                     visited[neighbor] = true;
    //                     bfsQueue.push(neighbor);
    //                 }
    //             }
    //         }

    //         components.push_back(std::move(component));
    //     }

    //     std::cout << "  Found " << components.size() << " connected components:" << std::endl;

    //     // サイズ順に表示
    //     std::vector<std::pair<size_t, size_t>> sizeIndex;
    //     for (size_t i = 0; i < components.size(); i++) {
    //         sizeIndex.push_back({components[i].size(), i});
    //     }
    //     std::sort(sizeIndex.rbegin(), sizeIndex.rend());

    //     for (size_t i = 0; i < std::min(size_t(10), sizeIndex.size()); i++) {
    //         std::cout << "    Component " << i << ": " << sizeIndex[i].first << " tets" << std::endl;
    //     }
    //     if (sizeIndex.size() > 10) {
    //         std::cout << "    ... and " << (sizeIndex.size() - 10) << " more components" << std::endl;
    //     }

    //     // 最大コンポーネント以外のフラグメントを削除
    //     if (!components.empty()) {
    //         size_t largestComponentIdx = sizeIndex[0].second;
    //         size_t largestSize = sizeIndex[0].first;

    //         std::vector<int> fragmentTetsToRemove;
    //         for (size_t i = 0; i < components.size(); i++) {
    //             if (i == largestComponentIdx) continue;
    //             if (components[i].size() <= static_cast<size_t>(fragmentThreshold)) {
    //                 fragmentTetsToRemove.insert(fragmentTetsToRemove.end(),
    //                                             components[i].begin(),
    //                                             components[i].end());
    //             }
    //         }

    //         if (!fragmentTetsToRemove.empty()) {
    //             target->invalidateHighResTetrahedra(fragmentTetsToRemove);
    //             allHighResTetsToInvalidate.insert(allHighResTetsToInvalidate.end(),
    //                                               fragmentTetsToRemove.begin(),
    //                                               fragmentTetsToRemove.end());
    //             std::cout << "  Removed " << fragmentTetsToRemove.size() << " fragment tets (threshold: " << fragmentThreshold << ")" << std::endl;
    //         }
    //     }

    //     auto t9 = std::chrono::high_resolution_clock::now();
    //     timeConnectedComponents = std::chrono::duration<double, std::milli>(t9 - t8).count();
    //     std::cout << "  ConnectedComponents: " << timeConnectedComponents << " ms" << std::endl;

    //     undoData.invalidatedHighResTets = allHighResTetsToInvalidate;

    //     // ====== Step 6: カット境界頂点を検出 ======
    //     auto t10 = std::chrono::high_resolution_clock::now();

    //     std::set<int> cutBoundaryVertices;
    //     std::map<int, std::pair<int, int>> vertexTetCounts;
    //     for (size_t i = 0; i < numHighResTets; i++) {
    //         bool isValid = target->highResTetValid[i];
    //         for (int j = 0; j < 4; j++) {
    //             int vid = highResData.tetIds[i * 4 + j];
    //             if (vertexTetCounts.find(vid) == vertexTetCounts.end()) {
    //                 vertexTetCounts[vid] = {0, 0};
    //             }
    //             if (isValid) {
    //                 vertexTetCounts[vid].first++;
    //             } else {
    //                 vertexTetCounts[vid].second++;
    //             }
    //         }
    //     }
    //     for (const auto& entry : vertexTetCounts) {
    //         if (entry.second.first > 0 && entry.second.second > 0) {
    //             cutBoundaryVertices.insert(entry.first);
    //         }
    //     }
    //     target->setCutBoundaryVertices(cutBoundaryVertices);

    //     auto t11 = std::chrono::high_resolution_clock::now();
    //     timeCutBoundaryVerts = std::chrono::duration<double, std::milli>(t11 - t10).count();
    //     std::cout << "  CutBoundaryVerts: " << cutBoundaryVertices.size() << " verts, " << timeCutBoundaryVerts << " ms" << std::endl;

    //     // ====== Step 7: HighResメッシュバッファを更新 ======
    //     auto t12 = std::chrono::high_resolution_clock::now();

    //     target->updateHighResMesh();
    //     target->updateHighResTetMesh();
    //     target->computeHighResNormals();

    //     auto t13 = std::chrono::high_resolution_clock::now();
    //     timeUpdateHighMesh = std::chrono::duration<double, std::milli>(t13 - t12).count();
    //     std::cout << "  UpdateHighResMesh: " << timeUpdateHighMesh << " ms" << std::endl;

    //     // ====== Step 8: LowRes四面体への伝播 ======
    //     auto t14 = std::chrono::high_resolution_clock::now();

    //     std::set<int> lowResTetsToInvalidate;

    //     for (size_t lowTetIdx = 0; lowTetIdx < target->numLowTets; lowTetIdx++) {
    //         if (!target->lowRes_tetValid[lowTetIdx]) continue;

    //         std::set<int> correspondingHighTets = target->getHighResTetsFromLowResTet(lowTetIdx);
    //         if (correspondingHighTets.empty()) continue;

    //         int totalCorrespondingHighTets = correspondingHighTets.size();
    //         int invalidHighTets = 0;

    //         for (int highTetIdx : correspondingHighTets) {
    //             if (highTetIdx >= 0 && highTetIdx < static_cast<int>(target->highResTetValid.size())) {
    //                 if (!target->highResTetValid[highTetIdx]) {
    //                     invalidHighTets++;
    //                 }
    //             }
    //         }

    //         if (invalidHighTets > 0 && invalidHighTets == totalCorrespondingHighTets) {
    //             lowResTetsToInvalidate.insert(lowTetIdx);
    //         }
    //     }

    //     std::vector<int> lowTetsVector(lowResTetsToInvalidate.begin(), lowResTetsToInvalidate.end());
    //     undoData.invalidatedLowResTets = lowTetsVector;

    //     auto t15 = std::chrono::high_resolution_clock::now();
    //     timeLowResPropagation = std::chrono::duration<double, std::milli>(t15 - t14).count();
    //     std::cout << "  LowResPropagation: " << lowTetsVector.size() << " tets, " << timeLowResPropagation << " ms" << std::endl;

    //     // ====== Step 9: LowRes無効化とハンドルグループ ======
    //     if (!lowTetsVector.empty()) {
    //         auto t16 = std::chrono::high_resolution_clock::now();

    //         target->invalidateLowResTetrahedra(lowTetsVector);

    //         auto t16_1 = std::chrono::high_resolution_clock::now();
    //         double timeInvalidateOnly = std::chrono::duration<double, std::milli>(t16_1 - t16).count();

    //         target->updateLowResMesh();

    //         auto t16_2 = std::chrono::high_resolution_clock::now();
    //         double timeUpdateLowMesh = std::chrono::duration<double, std::milli>(t16_2 - t16_1).count();

    //         target->updateLowResTetMeshes();

    //         auto t16_3 = std::chrono::high_resolution_clock::now();
    //         double timeUpdateLowTetMesh = std::chrono::duration<double, std::milli>(t16_3 - t16_2).count();

    //         std::cout << "  InvalidateLowRes breakdown:" << std::endl;
    //         std::cout << "    invalidateLowResTetrahedra: " << timeInvalidateOnly << " ms" << std::endl;
    //         std::cout << "    updateLowResMesh: " << timeUpdateLowMesh << " ms" << std::endl;
    //         std::cout << "    updateLowResTetMeshes: " << timeUpdateLowTetMesh << " ms" << std::endl;

    //         // ターゲットが子メッシュの場合、孤立した頂点のアンカーを解除
    //         if (target->hasParentSoftBody()) {
    //             std::vector<int> releasedVerts = target->releaseAnchorsForOrphanedVertices();
    //             undoData.selfAnchorRelease.releasedVertices = releasedVerts;
    //         }

    //         auto t17 = std::chrono::high_resolution_clock::now();
    //         timeInvalidateLow = std::chrono::duration<double, std::milli>(t17 - t16).count();
    //         std::cout << "  InvalidateLowRes: " << timeInvalidateLow << " ms" << std::endl;

    //         // ハンドルグループのチェック
    //         auto t18 = std::chrono::high_resolution_clock::now();

    //         if (!target->handleGroups.empty()) {
    //             std::vector<int> groupsToRemove;

    //             for (size_t g = 0; g < target->handleGroups.size(); g++) {
    //                 const auto& group = target->handleGroups[g];
    //                 int centerVert = group.centerVertex;

    //                 bool centerHasValidTet = false;
    //                 for (size_t t = 0; t < target->numLowTets; t++) {
    //                     if (!target->lowRes_tetValid[t]) continue;

    //                     for (int j = 0; j < 4; j++) {
    //                         if (target->lowRes_tetIds[t * 4 + j] == centerVert) {
    //                             centerHasValidTet = true;
    //                             break;
    //                         }
    //                     }
    //                     if (centerHasValidTet) break;
    //                 }

    //                 if (!centerHasValidTet) {
    //                     groupsToRemove.push_back(g);
    //                 }
    //             }

    //             std::sort(groupsToRemove.begin(), groupsToRemove.end(), std::greater<int>());
    //             for (int idx : groupsToRemove) {
    //                 target->removeHandleGroup(idx);
    //             }
    //         }

    //         auto t19 = std::chrono::high_resolution_clock::now();
    //         timeHandleGroups = std::chrono::duration<double, std::milli>(t19 - t18).count();
    //         std::cout << "  HandleGroups: " << timeHandleGroups << " ms" << std::endl;

    //         // エッジの有効性を更新（最適化版）
    //         auto t20 = std::chrono::high_resolution_clock::now();

    //         size_t numEdges = target->lowRes_edgeIds.size() / 2;
    //         target->edgeValid.resize(numEdges);
    //         std::fill(target->edgeValid.begin(), target->edgeValid.end(), false);

    //         // Step 9-1: 有効な四面体から頂点ペアのセットを構築 O(numValidTets × 6)
    //         std::set<std::pair<int, int>> validEdgePairs;

    //         for (size_t t = 0; t < target->numLowTets; t++) {
    //             if (!target->lowRes_tetValid[t]) continue;

    //             int v0 = target->lowRes_tetIds[t * 4 + 0];
    //             int v1 = target->lowRes_tetIds[t * 4 + 1];
    //             int v2 = target->lowRes_tetIds[t * 4 + 2];
    //             int v3 = target->lowRes_tetIds[t * 4 + 3];

    //             validEdgePairs.insert({std::min(v0, v1), std::max(v0, v1)});
    //             validEdgePairs.insert({std::min(v0, v2), std::max(v0, v2)});
    //             validEdgePairs.insert({std::min(v0, v3), std::max(v0, v3)});
    //             validEdgePairs.insert({std::min(v1, v2), std::max(v1, v2)});
    //             validEdgePairs.insert({std::min(v1, v3), std::max(v1, v3)});
    //             validEdgePairs.insert({std::min(v2, v3), std::max(v2, v3)});
    //         }

    //         // Step 9-2: 各エッジの有効性をO(log N)でチェック
    //         for (size_t i = 0; i < numEdges; i++) {
    //             int id0 = target->lowRes_edgeIds[2 * i];
    //             int id1 = target->lowRes_edgeIds[2 * i + 1];
    //             target->edgeValid[i] = (validEdgePairs.count({std::min(id0, id1), std::max(id0, id1)}) > 0);
    //         }

    //         auto t21 = std::chrono::high_resolution_clock::now();
    //         timeEdgeValidity = std::chrono::duration<double, std::milli>(t21 - t20).count();
    //         std::cout << "  EdgeValidity: " << timeEdgeValidity << " ms" << std::endl;
    //     }

    //     // ====== Step 10: スムージングをリセット ======
    //     auto t22 = std::chrono::high_resolution_clock::now();

    //     target->invalidateSmoothingCache();

    //     if (wasSmooth) {
    //         target->enableSmoothDisplay(false);
    //         while (glGetError() != GL_NO_ERROR) {}
    //         target->enableSmoothDisplay(true);

    //         if (target->smoothDisplayMode) {
    //             target->setSmoothingParameters(savedIterations, savedFactor,
    //                                            savedSizeAdjust, savedScalingMethod);
    //             target->updateSmoothBuffers();
    //             GLenum err = glGetError();
    //             if (err != GL_NO_ERROR) {
    //                 target->enableSmoothDisplay(false);
    //                 target->enableSmoothDisplay(true);
    //             }
    //         }
    //     }

    //     auto t23 = std::chrono::high_resolution_clock::now();
    //     timeSmoothing = std::chrono::duration<double, std::milli>(t23 - t22).count();
    //     std::cout << "  Smoothing: " << timeSmoothing << " ms" << std::endl;

    //     // ====== Step 11: 履歴に追加 ======
    //     auto t24 = std::chrono::high_resolution_clock::now();

    //     if (static_cast<int>(undoHistory.size()) >= maxHistorySize) {
    //         undoHistory.erase(undoHistory.begin());
    //     }
    //     undoHistory.push_back(undoData);

    //     dampingFrameOut = postCutDampingFrames;

    //     auto t25 = std::chrono::high_resolution_clock::now();
    //     timeFinalize = std::chrono::duration<double, std::milli>(t25 - t24).count();

    //     // ====== 合計時間と内訳 ======
    //     auto totalEnd = std::chrono::high_resolution_clock::now();
    //     double totalTime = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    //     std::cout << "\n========== " << targetName << " CUT TIMING BREAKDOWN ==========" << std::endl;
    //     std::cout << "  Backup:              " << std::fixed << std::setprecision(2) << timeBackup << " ms" << std::endl;
    //     std::cout << "  Intersection:        " << timeIntersection << " ms" << std::endl;
    //     std::cout << "  Boundary:            " << timeBoundary << " ms" << std::endl;
    //     std::cout << "  InvalidateHighRes:   " << timeInvalidateHigh << " ms" << std::endl;
    //     std::cout << "  ConnectedComponents: " << timeConnectedComponents << " ms" << std::endl;
    //     std::cout << "  CutBoundaryVerts:    " << timeCutBoundaryVerts << " ms" << std::endl;
    //     std::cout << "  UpdateHighResMesh:   " << timeUpdateHighMesh << " ms" << std::endl;
    //     std::cout << "  LowResPropagation:   " << timeLowResPropagation << " ms" << std::endl;
    //     std::cout << "  InvalidateLowRes:    " << timeInvalidateLow << " ms" << std::endl;
    //     std::cout << "  HandleGroups:        " << timeHandleGroups << " ms" << std::endl;
    //     std::cout << "  EdgeValidity:        " << timeEdgeValidity << " ms" << std::endl;
    //     std::cout << "  Smoothing:           " << timeSmoothing << " ms" << std::endl;
    //     std::cout << "  Finalize:            " << timeFinalize << " ms" << std::endl;
    //     std::cout << "  ----------------------------------------" << std::endl;
    //     std::cout << "  TOTAL:               " << totalTime << " ms" << std::endl;
    //     std::cout << "================================================\n" << std::endl;

    //     return true;
    // }

/*
    //--------------------------------------------------------------------------
    // メッシュカット実行
    //--------------------------------------------------------------------------
    bool performMeshCut(
        SoftBodyGPUDuo* target,
        SoftBodyParallelSolver* cpuSolver,
        const mCutMesh* cutterMesh,
        const std::string& targetName,
        int isolatedRemovalMode,
        int fragmentThreshold,
        int& dampingFrameOut,
        int postCutDampingFrames,
        const std::vector<SoftBodyGPUDuo*>& childObjects = {})  // ★追加
    {
        if (!target || !cutterMesh) {
            std::cout << "Invalid target or cutter mesh" << std::endl;
            return false;
        }

        std::cout << "\n=== STARTING CUT OPERATION (" << targetName << ") ===" << std::endl;

        // ====== Undoデータのバックアップ ======
        CutUndoData undoData;
        undoData.target = target;
        undoData.cpuSolver = cpuSolver;
        undoData.targetName = targetName;
        undoData.cutType = CutUndoData::MESH_CUT;
        undoData.highResTetValid_backup = target->highResTetValid;
        undoData.lowRes_tetValid_backup = target->lowRes_tetValid;
        undoData.lowRes_invMasses_backup = target->lowRes_invMasses;
        undoData.edgeValid_backup = target->edgeValid;
        undoData.wasSmooth = target->smoothDisplayMode;
        undoData.smoothingIterations = target->smoothingIterations;
        undoData.smoothingFactor = target->smoothingFactor;
        undoData.enableSizeAdjust = target->enableSizeAdjustment;
        undoData.scalingMethod = target->scalingMethod;

        // ★★★ 子オブジェクトのアンカー状態をバックアップ ★★★
        for (SoftBodyGPUDuo* child : childObjects) {
            if (child && child->parentSoftBody == target) {
                CutUndoData::ChildAnchorBackup childBackup;
                childBackup.child = child;
                childBackup.isAnchoredToParent_backup = child->isAnchoredToParent;
                childBackup.lowRes_invMasses_backup = child->lowRes_invMasses;
                childBackup.skinningToParent_backup = child->skinningToParent;
                childBackup.numAnchoredVertices_backup = child->numAnchoredVertices;
                undoData.childAnchors.push_back(childBackup);
                std::cout << "  Backed up child anchor state: " << childBackup.numAnchoredVertices_backup << " anchored vertices" << std::endl;
            }
        }

        // ★★★ ターゲット自身が子メッシュの場合（親を持つ場合）のアンカーバックアップ ★★★
        if (target->hasParentSoftBody()) {
            undoData.selfAnchorRelease.isAnchoredToParent_backup = target->isAnchoredToParent;
            undoData.selfAnchorRelease.lowRes_invMasses_backup = target->lowRes_invMasses;
            undoData.selfAnchorRelease.numAnchoredVertices_backup = target->numAnchoredVertices;
            std::cout << "  Backed up self anchor state (child mesh): "
                      << target->numAnchoredVertices << " anchored vertices" << std::endl;
        }

        // ====== デバッグ: カット前の状態 ======
        std::cout << "\n[DEBUG] === PRE-CUT STATE ===" << std::endl;
        std::cout << "  lowRes_pinnedVertices.size(): " << target->lowRes_pinnedVertices.size() << std::endl;

        int zeroInvMassBefore = 0;
        for (size_t i = 0; i < target->numLowResParticles; i++) {
            if (target->lowRes_invMasses[i] == 0.0f) {
                zeroInvMassBefore++;
            }
        }
        std::cout << "  invMass==0 (fixed): " << zeroInvMassBefore << " / " << target->numLowResParticles << std::endl;

        // スムージング状態を保存
        bool wasSmooth = target->smoothDisplayMode;
        int savedIterations = target->smoothingIterations;
        float savedFactor = target->smoothingFactor;
        bool savedSizeAdjust = target->enableSizeAdjustment;
        int savedScalingMethod = target->scalingMethod;

        // OpenGLエラーをクリア
        while (glGetError() != GL_NO_ERROR) {}

        // ====== カッターメッシュをMeshData形式に変換 ======
        SoftBodyGPUDuo::MeshData cutterData;
        cutterData.verts = cutterMesh->mVertices;
        cutterData.tetSurfaceTriIds = std::vector<int>(cutterMesh->mIndices.begin(), cutterMesh->mIndices.end());

        // ====== HighResMeshをMeshData形式に変換 ======
        SoftBodyGPUDuo::MeshData highResData;
        highResData.verts = target->highRes_positions;
        highResData.tetIds = target->highResMeshData.tetIds;
        highResData.tetEdgeIds = target->highResMeshData.tetEdgeIds;
        highResData.tetSurfaceTriIds = target->highResMeshData.tetSurfaceTriIds;

        // ====== TetMapping計算（まだなら）======
        if (!target->tetMappingComputed) {
            target->computeTetToTetMappingLowToHigh();
        }
        // ====== 交差判定（AdjProp版 - 全ターゲット共通）======
        std::vector<int> intersectingHighResTets;

        const auto& adjacency = target->getHighResTetAdjacency();

        // ========== AdjProp版 ==========
        auto adjStart = std::chrono::high_resolution_clock::now();

        intersectingHighResTets =
            MeshCuttingGPUDuo::findIntersectingTetrahedraBySurfaceTriangles_AdjProp(
                cutterData,
                highResData,
                target->highResSurfaceTriToTet,
                target->highRes_positions,
                target->highResTetValid,
                adjacency,
                3,
                true);  // verbose=false で詳細ログ抑制

        auto adjEnd = std::chrono::high_resolution_clock::now();
        double adjTime = std::chrono::duration<double, std::milli>(adjEnd - adjStart).count();

        std::cout << "\n========== " << targetName << " 時間比較 ==========" << std::endl;
        std::cout << "AdjProp版:    " << adjTime << " ms (" << intersectingHighResTets.size() << " tets)" << std::endl;
        if (intersectingHighResTets.empty()) {
            std::cout << "No intersection detected" << std::endl;
            return false;
        }

        std::cout << "Found " << intersectingHighResTets.size() << " intersecting high-res tets" << std::endl;

        // ====== ステップ1: 境界付近の四面体も追加（オプション）======
        std::vector<int> allHighResTetsToInvalidate = intersectingHighResTets;

        if (isolatedRemovalMode > 0) {
            std::set<int> boundaryVertices;
            for (int tetIdx : intersectingHighResTets) {
                for (int j = 0; j < 4; j++) {
                    boundaryVertices.insert(highResData.tetIds[tetIdx * 4 + j]);
                }
            }

            size_t numTets = highResData.tetIds.size() / 4;
            for (size_t i = 0; i < numTets; i++) {
                if (std::find(allHighResTetsToInvalidate.begin(),
                              allHighResTetsToInvalidate.end(), i) != allHighResTetsToInvalidate.end()) {
                    continue;
                }
                for (int j = 0; j < 4; j++) {
                    if (boundaryVertices.count(highResData.tetIds[i * 4 + j]) > 0) {
                        allHighResTetsToInvalidate.push_back(i);
                        break;
                    }
                }
            }
        }

        // ====== ステップ2: HighRes四面体を無効化 ======
        target->invalidateHighResTetrahedra(allHighResTetsToInvalidate);
        std::cout << "Invalidated " << allHighResTetsToInvalidate.size() << " high-res tets" << std::endl;

        // ====== ステップ3: 連結成分を検出してフラグメント削除 ======
        std::set<int> allValidHighTets;
        size_t totalHighTets = highResData.tetIds.size() / 4;
        for (size_t i = 0; i < totalHighTets; i++) {
            if (target->highResTetValid[i]) {
                allValidHighTets.insert(i);
            }
        }

        if (!allValidHighTets.empty()) {
            std::vector<std::vector<int>> components =
                MeshCuttingGPUDuo::findConnectedComponents(highResData, allValidHighTets);

            size_t largestComponentIdx = 0;
            size_t largestSize = 0;
            for (size_t i = 0; i < components.size(); i++) {
                if (components[i].size() > largestSize) {
                    largestSize = components[i].size();
                    largestComponentIdx = i;
                }
            }

            std::vector<int> fragmentTetsToRemove;
            for (size_t i = 0; i < components.size(); i++) {
                if (i == largestComponentIdx) continue;
                if (components[i].size() <= static_cast<size_t>(fragmentThreshold)) {
                    fragmentTetsToRemove.insert(fragmentTetsToRemove.end(),
                                                components[i].begin(),
                                                components[i].end());
                }
            }

            if (!fragmentTetsToRemove.empty()) {
                std::cout << "Removing small fragments: " << fragmentTetsToRemove.size() << " high-res tets" << std::endl;
                target->invalidateHighResTetrahedra(fragmentTetsToRemove);
                allHighResTetsToInvalidate.insert(allHighResTetsToInvalidate.end(),
                                                  fragmentTetsToRemove.begin(),
                                                  fragmentTetsToRemove.end());
            }

            std::cout << "Connected components found: " << components.size() << std::endl;
            std::cout << "Largest component size: " << largestSize << " tets" << std::endl;
        }

        // Undo用に無効化されたHighRes四面体を記録
        undoData.invalidatedHighResTets = allHighResTetsToInvalidate;

        // ====== ステップ4: カット境界頂点を検出 ======
        std::set<int> cutBoundaryVertices;
        std::map<int, std::pair<int, int>> vertexTetCounts;
        for (size_t i = 0; i < totalHighTets; i++) {
            bool isValid = target->highResTetValid[i];
            for (int j = 0; j < 4; j++) {
                int vid = highResData.tetIds[i * 4 + j];
                if (vertexTetCounts.find(vid) == vertexTetCounts.end()) {
                    vertexTetCounts[vid] = {0, 0};
                }
                if (isValid) {
                    vertexTetCounts[vid].first++;
                } else {
                    vertexTetCounts[vid].second++;
                }
            }
        }
        for (const auto& entry : vertexTetCounts) {
            if (entry.second.first > 0 && entry.second.second > 0) {
                cutBoundaryVertices.insert(entry.first);
            }
        }
        std::cout << "Detected " << cutBoundaryVertices.size() << " cut boundary vertices" << std::endl;
        // ★★★ 境界頂点をSoftBodyに設定（追加） ★★★
        target->setCutBoundaryVertices(cutBoundaryVertices);
        // ====== ステップ5: HighResメッシュバッファを更新 ======
        target->updateHighResMesh();
        target->updateHighResTetMesh();
        target->computeHighResNormals();

        // ====== ステップ6: LowRes四面体への伝播 ======
        std::set<int> lowResTetsToInvalidate;

        for (size_t lowTetIdx = 0; lowTetIdx < target->numLowTets; lowTetIdx++) {
            if (!target->lowRes_tetValid[lowTetIdx]) continue;

            std::set<int> correspondingHighTets = target->getHighResTetsFromLowResTet(lowTetIdx);
            if (correspondingHighTets.empty()) continue;

            int totalCorrespondingHighTets = correspondingHighTets.size();
            int invalidHighTets = 0;

            for (int highTetIdx : correspondingHighTets) {
                if (highTetIdx >= 0 && highTetIdx < static_cast<int>(target->highResTetValid.size())) {
                    if (!target->highResTetValid[highTetIdx]) {
                        invalidHighTets++;
                    }
                }
            }

            if (invalidHighTets > 0 && invalidHighTets == totalCorrespondingHighTets) {
                lowResTetsToInvalidate.insert(lowTetIdx);
            }
        }

        std::vector<int> lowTetsVector(lowResTetsToInvalidate.begin(), lowResTetsToInvalidate.end());

        // Undo用に無効化されたLowRes四面体を記録
        undoData.invalidatedLowResTets = lowTetsVector;

        if (!lowTetsVector.empty()) {
            std::cout << "\n[DEBUG] === INVALIDATING LOW-RES TETS ===" << std::endl;
            std::cout << "  Tets to invalidate: " << lowTetsVector.size() << std::endl;

            target->invalidateLowResTetrahedra(lowTetsVector);
            target->updateLowResMesh();
            target->updateLowResTetMeshes();

            // ★★★ ターゲットが子メッシュの場合、孤立した頂点のアンカーを解除 ★★★
            if (target->hasParentSoftBody()) {
                std::vector<int> releasedVerts = target->releaseAnchorsForOrphanedVertices();
                undoData.selfAnchorRelease.releasedVertices = releasedVerts;
                if (!releasedVerts.empty()) {
                    std::cout << "  Released " << releasedVerts.size()
                              << " orphaned anchored vertices" << std::endl;
                }
            }

            // ====== ステップ7: ハンドルグループのチェック ======
            if (!target->handleGroups.empty()) {
                std::cout << "\n[DEBUG] === CHECKING HANDLE GROUPS AFTER CUT ===" << std::endl;

                std::vector<int> groupsToRemove;

                for (size_t g = 0; g < target->handleGroups.size(); g++) {
                    const auto& group = target->handleGroups[g];
                    int centerVert = group.centerVertex;

                    bool centerHasValidTet = false;
                    for (size_t t = 0; t < target->numLowTets; t++) {
                        if (!target->lowRes_tetValid[t]) continue;

                        for (int j = 0; j < 4; j++) {
                            if (target->lowRes_tetIds[t * 4 + j] == centerVert) {
                                centerHasValidTet = true;
                                break;
                            }
                        }
                        if (centerHasValidTet) break;
                    }

                    if (!centerHasValidTet) {
                        std::cout << "  Handle group " << g << " center vertex " << centerVert << " is isolated" << std::endl;
                        groupsToRemove.push_back(g);
                    }
                }

                std::sort(groupsToRemove.begin(), groupsToRemove.end(), std::greater<int>());
                for (int idx : groupsToRemove) {
                    std::cout << "  Removing handle group " << idx << std::endl;
                    target->removeHandleGroup(idx);
                }

                if (!groupsToRemove.empty()) {
                    std::cout << "  " << groupsToRemove.size() << " handle group(s) removed" << std::endl;
                    std::cout << "  Remaining handle groups: " << target->handleGroups.size() << std::endl;
                }
            }

            // ====== ステップ8: エッジの有効性を更新 ======
            size_t numEdges = target->lowRes_edgeIds.size() / 2;
            target->edgeValid.resize(numEdges);
            std::fill(target->edgeValid.begin(), target->edgeValid.end(), false);

            for (size_t i = 0; i < numEdges; i++) {
                int id0 = target->lowRes_edgeIds[2 * i];
                int id1 = target->lowRes_edgeIds[2 * i + 1];

                for (size_t t = 0; t < target->numLowTets; t++) {
                    if (!target->lowRes_tetValid[t]) continue;

                    bool hasId0 = false, hasId1 = false;
                    for (int j = 0; j < 4; j++) {
                        int vid = target->lowRes_tetIds[t * 4 + j];
                        if (vid == id0) hasId0 = true;
                        if (vid == id1) hasId1 = true;
                    }

                    if (hasId0 && hasId1) {
                        target->edgeValid[i] = true;
                        break;
                    }
                }
            }

        }

        // ====== ステップ10: スムージングをリセット ======
        target->invalidateSmoothingCache();

        if (wasSmooth) {
            target->enableSmoothDisplay(false);
            while (glGetError() != GL_NO_ERROR) {}
            target->enableSmoothDisplay(true);

            if (target->smoothDisplayMode) {
                target->setSmoothingParameters(savedIterations, savedFactor,
                                               savedSizeAdjust, savedScalingMethod);
                target->updateSmoothBuffers();
                GLenum err = glGetError();
                if (err != GL_NO_ERROR) {
                    std::cout << "Warning: OpenGL error after smooth reconstruction: " << err << std::endl;
                    target->enableSmoothDisplay(false);
                    target->enableSmoothDisplay(true);
                }
            }
        }

        // ====== 結果出力 ======
        std::cout << "\n=== CUT OPERATION (" << targetName << ") COMPLETED ===" << std::endl;
        std::cout << "  High-res tets invalidated: " << allHighResTetsToInvalidate.size() << std::endl;
        std::cout << "  Low-res tets invalidated: " << lowResTetsToInvalidate.size() << std::endl;

        int validHighTets = 0;
        for (size_t i = 0; i < target->highResTetValid.size(); i++) {
            if (target->highResTetValid[i]) validHighTets++;
        }
        std::cout << "  HighRes after cut: " << validHighTets << " / "
                  << target->highResTetValid.size() << " tets valid" << std::endl;

        int finalZeroInvMass = 0;
        for (size_t i = 0; i < target->numLowResParticles; i++) {
            if (target->lowRes_invMasses[i] == 0.0f) finalZeroInvMass++;
        }
        std::cout << "  Fixed vertices change: " << zeroInvMassBefore << " -> " << finalZeroInvMass << std::endl;
        std::cout << "  Child objects backed up: " << undoData.childAnchors.size() << std::endl;

        // ====== 履歴に追加 ======
        if (static_cast<int>(undoHistory.size()) >= maxHistorySize) {
            undoHistory.erase(undoHistory.begin());
        }
        undoHistory.push_back(undoData);
        std::cout << "  Undo history size: " << undoHistory.size() << std::endl;

        // ====== カット後ダンピング ======
        dampingFrameOut = postCutDampingFrames;
        std::cout << "  Post-cut damping: " << dampingFrameOut << " frames" << std::endl;

        return true;
    }
*/



public:
    //--------------------------------------------------------------------------
    // セグメントカット実行
    //--------------------------------------------------------------------------
    //--------------------------------------------------------------------------
    // セグメントカット実行
    //--------------------------------------------------------------------------
    bool performSegmentCut(
        SoftBodyGPUDuo* target,
        SoftBodyParallelSolver* cpuSolver,
        const std::vector<int>& segmentTets,
        const std::string& targetName,
        int isolatedRemovalMode,
        const std::vector<SoftBodyGPUDuo*>& childObjects = {})
    {
        if (!target) {
            std::cout << "Invalid target" << std::endl;
            return false;
        }

        if (segmentTets.empty()) {
            std::cout << "No segments selected or no tetrahedra found" << std::endl;
            return false;
        }

        std::cout << "\n=== STARTING SEGMENT CUT OPERATION ===" << std::endl;
        std::cout << "Found " << segmentTets.size() << " tetrahedra to cut" << std::endl;

        // ====== Undoデータのバックアップ ======
        CutUndoData undoData;
        undoData.target = target;
        undoData.cpuSolver = cpuSolver;
        undoData.targetName = targetName;
        undoData.cutType = CutUndoData::SEGMENT_CUT;
        undoData.highResTetValid_backup = target->highResTetValid;
        undoData.lowRes_tetValid_backup = target->lowRes_tetValid;
        undoData.lowRes_invMasses_backup = target->lowRes_invMasses;
        undoData.edgeValid_backup = target->edgeValid;
        undoData.wasSmooth = target->smoothDisplayMode;
        undoData.smoothingIterations = target->smoothingIterations;
        undoData.smoothingFactor = target->smoothingFactor;
        undoData.enableSizeAdjust = target->enableSizeAdjustment;
        undoData.scalingMethod = target->scalingMethod;

        // ★★★ 子オブジェクトのアンカー状態をバックアップ ★★★
        for (SoftBodyGPUDuo* child : childObjects) {
            if (child && child->parentSoftBody == target) {
                CutUndoData::ChildAnchorBackup childBackup;
                childBackup.child = child;
                childBackup.isAnchoredToParent_backup = child->isAnchoredToParent;
                childBackup.lowRes_invMasses_backup = child->lowRes_invMasses;
                childBackup.skinningToParent_backup = child->skinningToParent;
                childBackup.numAnchoredVertices_backup = child->numAnchoredVertices;
                undoData.childAnchors.push_back(childBackup);
                std::cout << "  Backed up child anchor state: " << childBackup.numAnchoredVertices_backup << " anchored vertices" << std::endl;
            }
        }

        // ★★★ ターゲット自身が子メッシュの場合（親を持つ場合）のアンカーバックアップ ★★★
        if (target->hasParentSoftBody()) {
            undoData.selfAnchorRelease.isAnchoredToParent_backup = target->isAnchoredToParent;
            undoData.selfAnchorRelease.lowRes_invMasses_backup = target->lowRes_invMasses;
            undoData.selfAnchorRelease.numAnchoredVertices_backup = target->numAnchoredVertices;
            std::cout << "  Backed up self anchor state (child mesh): "
                      << target->numAnchoredVertices << " anchored vertices" << std::endl;
        }

        // ====== ステップ1: 境界付近の四面体も追加（オプション）======
        std::vector<int> allHighResTetsToInvalidate = segmentTets;

        if (isolatedRemovalMode > 0) {
            std::set<int> boundaryVertices;
            for (int tetIdx : segmentTets) {
                for (int j = 0; j < 4; j++) {
                    boundaryVertices.insert(target->highResMeshData.tetIds[tetIdx * 4 + j]);
                }
            }

            size_t numTets = target->highResMeshData.tetIds.size() / 4;
            for (size_t i = 0; i < numTets; i++) {
                if (!target->highResTetValid[i]) continue;
                if (std::find(allHighResTetsToInvalidate.begin(),
                              allHighResTetsToInvalidate.end(), i) != allHighResTetsToInvalidate.end()) {
                    continue;
                }
                for (int j = 0; j < 4; j++) {
                    if (boundaryVertices.count(target->highResMeshData.tetIds[i * 4 + j]) > 0) {
                        allHighResTetsToInvalidate.push_back(i);
                        break;
                    }
                }
            }
            std::cout << "  After boundary expansion: " << allHighResTetsToInvalidate.size() << " tets" << std::endl;
        }

        // Undo用に記録
        undoData.invalidatedHighResTets = allHighResTetsToInvalidate;

        // ====== ステップ2: HighRes四面体を無効化 ======
        target->invalidateHighResTetrahedra(allHighResTetsToInvalidate);
        target->updateHighResMesh();
        target->updateHighResTetMesh();
        target->computeHighResNormals();

        // ★★★ ステップ2.5: カット境界頂点を検出して設定（境界スムージング用）★★★
        {
            std::set<int> cutBoundaryVertices;
            std::map<int, std::pair<int, int>> vertexTetCounts;
            size_t totalHighTets = target->highResMeshData.tetIds.size() / 4;

            // 各頂点について、有効/無効な四面体の数をカウント
            for (size_t i = 0; i < totalHighTets; i++) {
                bool isValid = target->highResTetValid[i];
                for (int j = 0; j < 4; j++) {
                    int vid = target->highResMeshData.tetIds[i * 4 + j];
                    if (vertexTetCounts.find(vid) == vertexTetCounts.end()) {
                        vertexTetCounts[vid] = {0, 0};
                    }
                    if (isValid) {
                        vertexTetCounts[vid].first++;   // 有効な四面体数
                    } else {
                        vertexTetCounts[vid].second++;  // 無効な四面体数
                    }
                }
            }

            // 有効と無効の両方に属する頂点 = カット境界頂点
            for (const auto& entry : vertexTetCounts) {
                if (entry.second.first > 0 && entry.second.second > 0) {
                    cutBoundaryVertices.insert(entry.first);
                }
            }

            std::cout << "Detected " << cutBoundaryVertices.size() << " cut boundary vertices" << std::endl;

            // SoftBodyに境界頂点を設定（スムージング強度調整のため）
            target->setCutBoundaryVertices(cutBoundaryVertices);
        }

        // ====== ステップ3: LowRes四面体への伝播 ======
        std::set<int> lowResTetsToInvalidate;

        for (size_t lowTetIdx = 0; lowTetIdx < target->numLowTets; lowTetIdx++) {
            if (!target->lowRes_tetValid[lowTetIdx]) continue;

            std::set<int> correspondingHighTets = target->getHighResTetsFromLowResTet(lowTetIdx);
            if (correspondingHighTets.empty()) continue;

            int totalHighTets = correspondingHighTets.size();
            int invalidHighTets = 0;

            for (int highTetIdx : correspondingHighTets) {
                if (highTetIdx >= 0 && highTetIdx < static_cast<int>(target->highResTetValid.size())) {
                    if (!target->highResTetValid[highTetIdx]) {
                        invalidHighTets++;
                    }
                }
            }

            if (invalidHighTets > 0 && invalidHighTets == totalHighTets) {
                lowResTetsToInvalidate.insert(lowTetIdx);
            }
        }

        std::vector<int> lowTetsVector(lowResTetsToInvalidate.begin(), lowResTetsToInvalidate.end());

        // Undo用に記録
        undoData.invalidatedLowResTets = lowTetsVector;

        if (!lowTetsVector.empty()) {
            target->invalidateLowResTetrahedra(lowTetsVector);
            target->updateLowResMesh();
            target->updateLowResTetMeshes();

            // ★★★ ターゲットが子メッシュの場合、孤立した頂点のアンカーを解除 ★★★
            if (target->hasParentSoftBody()) {
                std::vector<int> releasedVerts = target->releaseAnchorsForOrphanedVertices();
                undoData.selfAnchorRelease.releasedVertices = releasedVerts;
                if (!releasedVerts.empty()) {
                    std::cout << "  Released " << releasedVerts.size()
                              << " orphaned anchored vertices" << std::endl;
                }
            }

            // ====== ステップ4: エッジの有効性を更新 ======
            size_t numEdges = target->lowRes_edgeIds.size() / 2;
            target->edgeValid.resize(numEdges);
            std::fill(target->edgeValid.begin(), target->edgeValid.end(), false);

            for (size_t i = 0; i < numEdges; i++) {
                int id0 = target->lowRes_edgeIds[2 * i];
                int id1 = target->lowRes_edgeIds[2 * i + 1];

                for (size_t t = 0; t < target->numLowTets; t++) {
                    if (!target->lowRes_tetValid[t]) continue;

                    bool hasId0 = false, hasId1 = false;
                    for (int j = 0; j < 4; j++) {
                        int vid = target->lowRes_tetIds[t * 4 + j];
                        if (vid == id0) hasId0 = true;
                        if (vid == id1) hasId1 = true;
                    }

                    if (hasId0 && hasId1) {
                        target->edgeValid[i] = true;
                        break;
                    }
                }
            }

            // ====== ステップ5: GPUソルバーとの同期 ======
            // Note: performSegmentCutにはgpuSolverがないので、cpuSolverのみ
            // 必要に応じてGPU同期を追加
        }

        // ====== ステップ6: スムージングの復元 ======
        if (undoData.wasSmooth) {
            target->invalidateSmoothingCache();
            target->enableSmoothDisplay(false);
            while (glGetError() != GL_NO_ERROR) {}
            target->enableSmoothDisplay(true);
            if (target->smoothDisplayMode) {
                target->setSmoothingParameters(
                    undoData.smoothingIterations,
                    undoData.smoothingFactor,
                    undoData.enableSizeAdjust,
                    undoData.scalingMethod);
                target->updateSmoothBuffers();
            }
        }

        // ====== 履歴に追加 ======
        if (static_cast<int>(undoHistory.size()) >= maxHistorySize) {
            undoHistory.erase(undoHistory.begin());
        }
        undoHistory.push_back(undoData);

        reinitializeSolver(target, cpuSolver);  // ← 自動で呼ぶ
        // ====== 結果出力 ======
        std::cout << "Segment cut operation completed" << std::endl;
        std::cout << "  High-res tets invalidated: " << allHighResTetsToInvalidate.size() << std::endl;
        std::cout << "  Low-res tets invalidated: " << lowTetsVector.size() << std::endl;
        std::cout << "  Boundary vertices for smoothing: " << target->getBoundaryVertexCount() << std::endl;
        std::cout << "  Child objects backed up: " << undoData.childAnchors.size() << std::endl;
        std::cout << "  Undo history size: " << undoHistory.size() << std::endl;
        std::cout << "  Press 'U' to undo" << std::endl;

        return true;
    }








    /*
    //--------------------------------------------------------------------------
    // Undo実行
    // //--------------------------------------------------------------------------
    // bool performUndo() {
    //     if (undoHistory.empty()) {
    //         std::cout << "No cut history to undo" << std::endl;
    //         return false;
    //     }

    //     CutUndoData& undo = undoHistory.back();

    //     if (!undo.target) {
    //         std::cout << "Invalid undo target" << std::endl;
    //         undoHistory.pop_back();
    //         return false;
    //     }

    //     std::cout << "\n=== UNDOING CUT OPERATION (" << undo.targetName << ") ===" << std::endl;
    //     std::cout << "  Cut type: " << (undo.cutType == CutUndoData::MESH_CUT ? "MESH_CUT" : "SEGMENT_CUT") << std::endl;

    //     // ====== 高解像度メッシュの復元 ======
    //     if (!undo.invalidatedHighResTets.empty()) {
    //         undo.target->validateHighResTetrahedra(undo.invalidatedHighResTets);
    //         undo.target->updateHighResMesh();
    //         undo.target->updateHighResTetMesh();
    //         undo.target->computeHighResNormals();
    //         std::cout << "  Restored " << undo.invalidatedHighResTets.size()
    //                   << " high-res tetrahedra" << std::endl;
    //     }

    //     // ====== 低解像度メッシュの復元（質量情報付き）======
    //     if (!undo.invalidatedLowResTets.empty()) {
    //         undo.target->validateLowResTetrahedraWithMasses(
    //             undo.invalidatedLowResTets,
    //             undo.lowRes_invMasses_backup);
    //         undo.target->updateLowResMesh();
    //         undo.target->updateLowResTetMeshes();
    //         std::cout << "  Restored " << undo.invalidatedLowResTets.size()
    //                   << " low-res tetrahedra" << std::endl;
    //     }

    //     // ====== エッジ有効性の復元 ======
    //     if (!undo.edgeValid_backup.empty()) {
    //         undo.target->edgeValid = undo.edgeValid_backup;
    //         std::cout << "  Restored edge validity" << std::endl;
    //     }

    //     // ★★★ 子オブジェクトのアンカー状態を復元 ★★★
    //     for (const auto& childBackup : undo.childAnchors) {
    //         if (childBackup.child) {
    //             childBackup.child->isAnchoredToParent = childBackup.isAnchoredToParent_backup;
    //             childBackup.child->lowRes_invMasses = childBackup.lowRes_invMasses_backup;
    //             childBackup.child->skinningToParent = childBackup.skinningToParent_backup;
    //             childBackup.child->numAnchoredVertices = childBackup.numAnchoredVertices_backup;
    //             std::cout << "  Restored child anchor state: " << childBackup.numAnchoredVertices_backup << " anchored vertices" << std::endl;
    //         }
    //     }

    //     // ★★★ ターゲット自身が子メッシュだった場合のアンカー復元 ★★★
    //     if (undo.target->hasParentSoftBody() && !undo.selfAnchorRelease.releasedVertices.empty()) {
    //         undo.target->restoreAnchorsForRestoredVertices(
    //             undo.invalidatedLowResTets,
    //             undo.selfAnchorRelease.isAnchoredToParent_backup,
    //             undo.selfAnchorRelease.lowRes_invMasses_backup);
    //         std::cout << "  Restored self anchor state (child mesh)" << std::endl;
    //     }


    //     // ====== スムージング設定の復元 ======
    //     if (undo.wasSmooth) {
    //         undo.target->invalidateSmoothingCache();
    //         undo.target->enableSmoothDisplay(false);
    //         while (glGetError() != GL_NO_ERROR) {}
    //         undo.target->enableSmoothDisplay(true);
    //         if (undo.target->smoothDisplayMode) {
    //             undo.target->setSmoothingParameters(
    //                 undo.smoothingIterations,
    //                 undo.smoothingFactor,
    //                 undo.enableSizeAdjust,
    //                 undo.scalingMethod);
    //             undo.target->updateSmoothBuffers();
    //         }
    //         std::cout << "  Smoothing settings restored" << std::endl;
    //     }

    //     // ====== 統計出力 ======
    //     int validHighTets = 0;
    //     for (size_t i = 0; i < undo.target->highResTetValid.size(); i++) {
    //         if (undo.target->highResTetValid[i]) validHighTets++;
    //     }
    //     std::cout << "  HighRes after undo: " << validHighTets << " / "
    //               << undo.target->highResTetValid.size() << " tets valid" << std::endl;

    //     int validLowTets = 0;
    //     for (size_t i = 0; i < undo.target->numLowTets; i++) {
    //         if (undo.target->lowRes_tetValid[i]) validLowTets++;
    //     }
    //     std::cout << "  LowRes after undo: " << validLowTets << " / "
    //               << undo.target->numLowTets << " tets valid" << std::endl;

    //     std::cout << "=== CUT UNDO COMPLETED (" << undo.targetName << ") ===" << std::endl;

    //     undoHistory.pop_back();
    //     std::cout << "  Remaining undo history: " << undoHistory.size() << std::endl;

    //     return true;
    // }

    */

    //--------------------------------------------------------------------------
    // ユーティリティ
    //--------------------------------------------------------------------------
    bool canUndo() const { return !undoHistory.empty(); }
    size_t getHistorySize() const { return undoHistory.size(); }
    void clearHistory() { undoHistory.clear(); }

    // Undo前にターゲット情報を取得（main.cppでcpuSolver更新用）
    SoftBodyGPUDuo* getLastUndoTarget() const {
        if (undoHistory.empty()) return nullptr;
        return undoHistory.back().target;
    }

    SoftBodyParallelSolver* getLastUndoCpuSolver() const {
        if (undoHistory.empty()) return nullptr;
        return undoHistory.back().cpuSolver;
    }


    // ★★★ デバッグ：ヒストリー一覧を表示 ★★★
    void printUndoHistory() const {
        std::cout << "\n╔════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║           UNDO HISTORY (" << undoHistory.size() << " entries)                        ║" << std::endl;
        std::cout << "╠════════════════════════════════════════════════════════════╣" << std::endl;

        if (undoHistory.empty()) {
            std::cout << "║  (empty)                                                   ║" << std::endl;
        } else {
            for (size_t i = 0; i < undoHistory.size(); i++) {
                const auto& entry = undoHistory[i];
                std::string typeStr = (entry.cutType == CutUndoData::MESH_CUT) ? "MESH" : "SEG ";
                std::string nameStr = entry.targetName.empty() ? "(unnamed)" : entry.targetName;

                // 最後のエントリ（次にUndoされる）をマーク
                std::string marker = (i == undoHistory.size() - 1) ? " <-- NEXT UNDO" : "";

                std::cout << "║  [" << i << "] " << typeStr << " | " << nameStr;

                // 子アンカーバックアップがある場合
                if (!entry.childAnchors.empty()) {
                    std::cout << " (+" << entry.childAnchors.size() << " children)";
                }

                std::cout << marker << std::endl;
            }
        }
        std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
    }

    bool performUndo() {
        // ★★★ デバッグ：Undo実行前にヒストリー一覧を表示 ★★★
        std::cout << "\n[DEBUG] performUndo() CALLED" << std::endl;
        printUndoHistory();

        if (undoHistory.empty()) {
            std::cout << "No cut history to undo" << std::endl;
            return false;
        }

        CutUndoData& undo = undoHistory.back();

        if (!undo.target) {
            std::cout << "Invalid undo target" << std::endl;
            undoHistory.pop_back();
            return false;
        }

        std::cout << "\n=== UNDOING CUT OPERATION (" << undo.targetName << ") ===" << std::endl;
        std::cout << "  Cut type: " << (undo.cutType == CutUndoData::MESH_CUT ? "MESH_CUT" : "SEGMENT_CUT") << std::endl;

        // ====== 高解像度メッシュの復元 ======
        if (!undo.invalidatedHighResTets.empty()) {
            undo.target->validateHighResTetrahedra(undo.invalidatedHighResTets);
            undo.target->updateHighResMesh();
            undo.target->updateHighResTetMesh();
            undo.target->computeHighResNormals();
            std::cout << "  Restored " << undo.invalidatedHighResTets.size()
                      << " high-res tetrahedra" << std::endl;
        }

        // ====== 低解像度メッシュの復元（質量情報付き）======
        if (!undo.invalidatedLowResTets.empty()) {
            undo.target->validateLowResTetrahedraWithMasses(
                undo.invalidatedLowResTets,
                undo.lowRes_invMasses_backup);
            undo.target->updateLowResMesh();
            undo.target->updateLowResTetMeshes();
            std::cout << "  Restored " << undo.invalidatedLowResTets.size()
                      << " low-res tetrahedra" << std::endl;
        }

        // ====== エッジ有効性の復元 ======
        if (!undo.edgeValid_backup.empty()) {
            undo.target->edgeValid = undo.edgeValid_backup;
            std::cout << "  Restored edge validity" << std::endl;
        }

        // ★★★ 子オブジェクトのアンカー状態を復元 ★★★
        for (const auto& childBackup : undo.childAnchors) {
            if (childBackup.child) {
                childBackup.child->isAnchoredToParent = childBackup.isAnchoredToParent_backup;
                childBackup.child->lowRes_invMasses = childBackup.lowRes_invMasses_backup;
                childBackup.child->skinningToParent = childBackup.skinningToParent_backup;
                childBackup.child->numAnchoredVertices = childBackup.numAnchoredVertices_backup;
                std::cout << "  Restored child anchor state: " << childBackup.numAnchoredVertices_backup << " anchored vertices" << std::endl;
            }
        }

        // ★★★ ターゲット自身が子メッシュだった場合のアンカー復元 ★★★
        if (undo.target->hasParentSoftBody() && !undo.selfAnchorRelease.releasedVertices.empty()) {
            undo.target->restoreAnchorsForRestoredVertices(
                undo.invalidatedLowResTets,
                undo.selfAnchorRelease.isAnchoredToParent_backup,
                undo.selfAnchorRelease.lowRes_invMasses_backup);
            std::cout << "  Restored self anchor state (child mesh)" << std::endl;
        }

        // ====== スムージング設定の復元 ======
        if (undo.wasSmooth) {
            undo.target->invalidateSmoothingCache();
            undo.target->enableSmoothDisplay(false);
            while (glGetError() != GL_NO_ERROR) {}
            undo.target->enableSmoothDisplay(true);
            if (undo.target->smoothDisplayMode) {
                undo.target->setSmoothingParameters(
                    undo.smoothingIterations,
                    undo.smoothingFactor,
                    undo.enableSizeAdjust,
                    undo.scalingMethod);
                undo.target->updateSmoothBuffers();
            }
            std::cout << "  Smoothing settings restored" << std::endl;
        }

        reinitializeSolver(undo.target, undo.cpuSolver);  // ← 正しい
        // ====== 統計出力 ======
        int validHighTets = 0;
        for (size_t i = 0; i < undo.target->highResTetValid.size(); i++) {
            if (undo.target->highResTetValid[i]) validHighTets++;
        }
        std::cout << "  HighRes after undo: " << validHighTets << " / "
                  << undo.target->highResTetValid.size() << " tets valid" << std::endl;

        int validLowTets = 0;
        for (size_t i = 0; i < undo.target->numLowTets; i++) {
            if (undo.target->lowRes_tetValid[i]) validLowTets++;
        }
        std::cout << "  LowRes after undo: " << validLowTets << " / "
                  << undo.target->numLowTets << " tets valid" << std::endl;

        std::cout << "=== CUT UNDO COMPLETED (" << undo.targetName << ") ===" << std::endl;

        undoHistory.pop_back();
        std::cout << "  Remaining undo history: " << undoHistory.size() << std::endl;

        return true;
    }
};

#endif // SOFTBODY_CUT_MANAGER_H
