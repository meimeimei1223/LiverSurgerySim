// CutSegmentManager.cpp
#include "CutSegmentManager.h"
#include "SoftBodyGPUDuo.h"
#include "VoxelSkeletonSegmentation.h"
#include <iostream>
#include <set>
#include <limits>
#include <cmath>

//=============================================================================
// カッターモード管理
//=============================================================================

std::string CutSegmentManager::getCutModeName() const {
    switch (cutMode_) {
    case CutMode::Skeleton:  return "Skeleton";
    case CutMode::Voronoi3D: return "Voronoi3D";
    default:                 return "None";
    }
}

//=============================================================================
// クリック選択モード管理
//=============================================================================

void CutSegmentManager::setClickMode(ClickMode mode) {
    clickMode_ = mode;
}

void CutSegmentManager::toggleLiverSelectMode() {
    if (clickMode_ == ClickMode::Liver) {
        clickMode_ = ClickMode::None;
    } else {
        clickMode_ = ClickMode::Liver;
    }
    std::cout << "Liver select mode: " << (clickMode_ == ClickMode::Liver ? "ON" : "OFF") << std::endl;
    if (clickMode_ == ClickMode::Liver) {
        std::cout << "  Click on liver to select corresponding portal region" << std::endl;
    }
}

void CutSegmentManager::togglePortalSelectMode() {
    if (clickMode_ == ClickMode::Portal) {
        clickMode_ = ClickMode::None;
    } else {
        clickMode_ = ClickMode::Portal;
    }
    std::cout << "Portal select mode: " << (clickMode_ == ClickMode::Portal ? "ON" : "OFF") << std::endl;
}

//=============================================================================
// 頂点検索関数
//=============================================================================

int CutSegmentManager::findNearestSmoothSurfaceVertex(SoftBodyGPUDuo* mesh, const glm::vec3& position) {
    if (!mesh) return -1;

    std::vector<float> vertices;
    std::vector<int> triIds;

    if (mesh->smoothDisplayMode &&
        !mesh->smoothedVertices.empty() &&
        !mesh->smoothSurfaceTriIds.empty()) {
        vertices = mesh->smoothedVertices;
        triIds = mesh->smoothSurfaceTriIds;
    } else {
        vertices = mesh->getLowResPositions();
        triIds = mesh->getLowResMeshData().tetSurfaceTriIds;
    }

    if (vertices.empty() || triIds.empty()) return -1;

    float minDist2 = std::numeric_limits<float>::max();
    int nearestVertex = -1;

    std::set<int> surfaceVerts;
    for (size_t i = 0; i < triIds.size(); i++) {
        surfaceVerts.insert(triIds[i]);
    }

    for (int vid : surfaceVerts) {
        if (vid * 3 + 2 >= static_cast<int>(vertices.size())) continue;
        glm::vec3 vpos(vertices[vid*3], vertices[vid*3+1], vertices[vid*3+2]);
        glm::vec3 diff = vpos - position;
        float dist2 = glm::dot(diff, diff);
        if (dist2 < minDist2) {
            minDist2 = dist2;
            nearestVertex = vid;
        }
    }

    return nearestVertex;
}

int CutSegmentManager::raycastToVertex(SoftBodyGPUDuo* mesh,
                                       float mouseX, float mouseY,
                                       const glm::mat4& view, const glm::mat4& projection,
                                       const glm::vec3& cameraPos,
                                       int windowWidth, int windowHeight,
                                       glm::vec3& outHitPos, int& outTriIndex) {
    if (!mesh) return -1;

    float x = (2.0f * mouseX) / windowWidth - 1.0f;
    float y = 1.0f - (2.0f * mouseY) / windowHeight;

    glm::vec4 rayClip(x, y, -1.0f, 1.0f);
    glm::vec4 rayEye = glm::inverse(projection) * rayClip;
    rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0f, 0.0f);
    glm::vec3 rayDir = glm::normalize(glm::vec3(glm::inverse(view) * rayEye));
    glm::vec3 rayOrigin = cameraPos;

    float closestT = std::numeric_limits<float>::max();
    int hitVertex = -1;
    outTriIndex = -1;

    const auto& pos = mesh->highRes_positions;
    const auto& tri = mesh->highResMeshData.tetSurfaceTriIds;

    for (size_t i = 0; i < tri.size(); i += 3) {
        glm::vec3 p0(pos[tri[i]*3], pos[tri[i]*3+1], pos[tri[i]*3+2]);
        glm::vec3 p1(pos[tri[i+1]*3], pos[tri[i+1]*3+1], pos[tri[i+1]*3+2]);
        glm::vec3 p2(pos[tri[i+2]*3], pos[tri[i+2]*3+1], pos[tri[i+2]*3+2]);

        glm::vec3 e1 = p1 - p0, e2 = p2 - p0;
        glm::vec3 h = glm::cross(rayDir, e2);
        float a = glm::dot(e1, h);
        if (std::abs(a) < 1e-7f) continue;

        float f = 1.0f / a;
        glm::vec3 s = rayOrigin - p0;
        float u = f * glm::dot(s, h);
        if (u < 0.0f || u > 1.0f) continue;

        glm::vec3 q = glm::cross(s, e1);
        float v = f * glm::dot(rayDir, q);
        if (v < 0.0f || u + v > 1.0f) continue;

        float t = f * glm::dot(e2, q);
        if (t > 0.001f && t < closestT) {
            closestT = t;
            hitVertex = tri[i];
            outTriIndex = static_cast<int>(i / 3);
        }
    }

    if (hitVertex >= 0) {
        outHitPos = rayOrigin + rayDir * closestT;
    }

    return hitVertex;
}

//=============================================================================
// 肝臓クリック処理（handleLiverClickと完全に同じ）
//=============================================================================

void CutSegmentManager::processLiverClick(int hitVertex, const glm::vec3& hitPos, int hitTriIndex, bool shiftPressed) {
    if (!skeleton_ || !liver_) return;

    VoxelSkeleton::SegmentationMode mode = skeleton_->getSegmentationMode();

    if (mode == VoxelSkeleton::SegmentationMode::OBJ) {
        int objSegId = liver_->getOBJSegmentIdAtVertex(hitVertex);

        std::cout << "\n=== Liver Click (OBJ Mode) ===" << std::endl;
        std::cout << "  Hit triangle: " << hitTriIndex << std::endl;
        std::cout << "  Hit vertex: " << hitVertex << std::endl;
        std::cout << "  Hit position: (" << hitPos.x << ", " << hitPos.y << ", " << hitPos.z << ")" << std::endl;
        std::cout << "  OBJ Segment ID: S" << objSegId << std::endl;
        std::cout << "==============================\n" << std::endl;  // 30文字

        if (objSegId > 0) {
            skeleton_->selectOBJSegment(objSegId);

            if (portal_) {
                portal_->updateOBJSegmentColorsWithSelection(*skeleton_, objSegId);
            }
            if (liver_) {
                liver_->updateOBJSegmentColorsWithSelection(*skeleton_, objSegId);
            }

            std::cout << "Selected OBJ Segment from Liver: S" << objSegId << std::endl;
        }

    } else if (mode == VoxelSkeleton::SegmentationMode::Voronoi3D) {
        glm::vec3 vertexPos = liver_->getRestVertexPosition(hitVertex);

        const auto* voronoi = skeleton_->getVoronoiSegmenter();
        if (!voronoi) return;

        std::vector<int> branchIds = voronoi->getBranchesAtPosition(vertexPos);

        std::cout << "\n=== Liver Click (Voronoi3D Mode) ===" << std::endl;
        std::cout << "  Hit triangle: " << hitTriIndex << std::endl;
        std::cout << "  Hit vertex: " << hitVertex << std::endl;
        std::cout << "  Hit position: (" << hitPos.x << ", " << hitPos.y << ", " << hitPos.z << ")" << std::endl;
        std::cout << "  Rest position: (" << vertexPos.x << ", " << vertexPos.y << ", " << vertexPos.z << ")" << std::endl;
        std::cout << "  Branches at this position: ";
        for (int bid : branchIds) {
            std::cout << bid << " ";
        }
        std::cout << "(" << branchIds.size() << " branches)" << std::endl;
        std::cout << "====================================\n" << std::endl;  // 36文字

        if (!branchIds.empty()) {
            skeleton_->selectBranches(branchIds);

            if (portal_) {
                portal_->updateVoronoi3DColorsWithSelection(*skeleton_, branchIds);
            }
            if (liver_) {
                liver_->updateVoronoi3DColorsWithSelection(*skeleton_, branchIds);
            }

            std::cout << "Selected " << branchIds.size() << " branches from Liver" << std::endl;
        }

    } else {
        // ★★★ Skeletonモード: handleLiverClickと完全に同じ ★★★
        // Vertex position出力なし、セグメント詳細なし、isTerminalなし、manualExtendModeなし
        int segId = liver_->getVertexSegmentId(hitVertex);

        std::cout << "\n=== Liver Click (Skeleton Distance Mode) ===" << std::endl;
        std::cout << "  Hit triangle: " << hitTriIndex << std::endl;
        std::cout << "  Hit vertex: " << hitVertex << std::endl;
        std::cout << "  Hit position: (" << hitPos.x << ", " << hitPos.y << ", " << hitPos.z << ")" << std::endl;
        std::cout << "  Segment ID: " << segId << std::endl;
        std::cout << "============================================\n" << std::endl;  // 44文字

        // ★★★ オリジナルと同じ: segId >= 0 && skeleton_ ★★★
        if (segId >= 0 && skeleton_) {
            // ★★★ liver と portal の両方を更新 ★★★
            if (shiftPressed) {
                liver_->toggleSegmentSelection(segId, *skeleton_);
                portal_->toggleSegmentSelection(segId, *skeleton_);
            } else {
                liver_->selectSegmentWithDownstream(segId, *skeleton_, false);
                portal_->selectSegmentWithDownstream(segId, *skeleton_, false);
            }

            std::cout << "Selected Segment from Liver: " << segId << std::endl;
        }
    }
}

//=============================================================================
// 門脈クリック処理（handlePortalClickと完全に同じ）
//=============================================================================

void CutSegmentManager::processPortalClick(int hitVertex, const glm::vec3& hitPos, int hitTriIndex, bool shiftPressed) {
    if (!skeleton_ || !portal_) return;

    VoxelSkeleton::SegmentationMode mode = skeleton_->getSegmentationMode();

    if (mode == VoxelSkeleton::SegmentationMode::OBJ) {
        int objSegId = portal_->getOBJSegmentIdAtVertex(hitVertex);

        std::cout << "\n=== Portal Click (OBJ Mode) ===" << std::endl;
        std::cout << "  Hit triangle: " << hitTriIndex << std::endl;
        std::cout << "  Hit vertex: " << hitVertex << std::endl;
        std::cout << "  Hit position: (" << hitPos.x << ", " << hitPos.y << ", " << hitPos.z << ")" << std::endl;
        std::cout << "  OBJ Segment ID: S" << objSegId << std::endl;
        std::cout << "================================\n" << std::endl;  // 32文字

        if (objSegId > 0) {
            skeleton_->selectOBJSegment(objSegId);

            if (portal_) {
                portal_->updateOBJSegmentColorsWithSelection(*skeleton_, objSegId);
            }
            if (liver_) {
                liver_->updateOBJSegmentColorsWithSelection(*skeleton_, objSegId);
            }

            std::cout << "Selected OBJ Segment: S" << objSegId << std::endl;
        }

    } else if (mode == VoxelSkeleton::SegmentationMode::Voronoi3D) {
        glm::vec3 vertexPos = portal_->getRestVertexPosition(hitVertex);

        const auto* voronoi = skeleton_->getVoronoiSegmenter();
        if (!voronoi) return;

        std::vector<int> branchIds = voronoi->getBranchesAtPosition(vertexPos);

        std::cout << "\n=== Portal Click (Voronoi3D Mode) ===" << std::endl;
        std::cout << "  Hit triangle: " << hitTriIndex << std::endl;
        std::cout << "  Hit vertex: " << hitVertex << std::endl;
        std::cout << "  Hit position: (" << hitPos.x << ", " << hitPos.y << ", " << hitPos.z << ")" << std::endl;
        std::cout << "  Rest position: (" << vertexPos.x << ", " << vertexPos.y << ", " << vertexPos.z << ")" << std::endl;
        std::cout << "  Branches at this position: ";
        for (int bid : branchIds) {
            std::cout << bid << " ";
        }
        std::cout << "(" << branchIds.size() << " branches)" << std::endl;
        std::cout << "=====================================\n" << std::endl;  // 37文字

        if (!branchIds.empty()) {
            skeleton_->selectBranches(branchIds);

            if (portal_) {
                portal_->updateVoronoi3DColorsWithSelection(*skeleton_, branchIds);
            }
            if (liver_) {
                liver_->updateVoronoi3DColorsWithSelection(*skeleton_, branchIds);
            }

            std::cout << "Selected " << branchIds.size() << " branches" << std::endl;
        }

    } else {
        // ★★★ Skeletonモード: handlePortalClickと完全に同じ ★★★
        // Vertex position出力あり、セグメント詳細あり、isTerminalあり、manualExtendModeあり
        int segId = portal_->getVertexSegmentId(hitVertex);

        // ★★★ Vertex position を出力 ★★★
        const auto& pos = portal_->highRes_positions;

        std::cout << "\n=== Portal Click (Skeleton Distance Mode) ===" << std::endl;
        std::cout << "  Hit triangle: " << hitTriIndex << std::endl;
        std::cout << "  Hit vertex: " << hitVertex << std::endl;
        std::cout << "  Hit position: (" << hitPos.x << ", " << hitPos.y << ", " << hitPos.z << ")" << std::endl;
        std::cout << "  Vertex position: ("
                  << pos[hitVertex*3] << ", "
                  << pos[hitVertex*3+1] << ", "
                  << pos[hitVertex*3+2] << ")" << std::endl;
        std::cout << "  Segment ID: " << segId << std::endl;

        // ★★★ オリジナルと同じ: segId >= 0 && skeleton_ ★★★
        if (segId >= 0 && skeleton_) {
            // ★★★ セグメント詳細情報を出力 ★★★
            const auto& segs = skeleton_->getSegments();
            if (segId < static_cast<int>(segs.size())) {
                std::cout << "  Segment " << segId << " info:" << std::endl;
                std::cout << "    childIds: " << segs[segId].childIds.size() << std::endl;
                std::cout << "    nodeIds: " << segs[segId].nodeIds.size() << std::endl;
                std::cout << "    hierarchyLevel: " << segs[segId].hierarchyLevel << std::endl;
            }

            // ★★★ isTerminalSegment をチェック ★★★
            bool isTerminal = skeleton_->isTerminalSegment(segId);
            std::cout << "  isTerminalSegment: " << (isTerminal ? "YES" : "NO") << std::endl;
            std::cout << "=============================================\n" << std::endl;  // 45文字

            // ★★★ manualExtendMode の処理 ★★★
            if (manualExtendMode_) {
                if (isTerminal) {
                    skeleton_->backupBeforeManualExtension();

                    if (skeleton_->extendTerminalBranch(segId, extensionNodeCount_)) {
                        skeleton_->saveManualExtension();

                        portal_->bindToSkeleton(*skeleton_);
                        liver_->bindToSkeleton(*skeleton_);

                        std::cout << "Extended segment " << segId << ". Press E to revert." << std::endl;
                    }
                } else {
                    std::cout << "Segment " << segId << " is not a terminal branch" << std::endl;
                }
                return;  // ★★★ manualExtendModeの場合はここで終了 ★★★
            }

            // ★★★ portal のみ色更新（liverは更新しない！） ★★★
            if (shiftPressed) {
                portal_->toggleSegmentSelection(segId, *skeleton_);
            } else {
                portal_->selectSegmentWithDownstream(segId, *skeleton_, false);
            }
        }
    }
}

//=============================================================================
// クリック処理（オリジナルと同じ呼び出し条件）
//=============================================================================

void CutSegmentManager::handleClick(float mouseX, float mouseY, bool shiftPressed,
                                    const glm::mat4& view, const glm::mat4& projection,
                                    const glm::vec3& cameraPos,
                                    int windowWidth, int windowHeight) {

    // ★★★ オリジナルと同じ: liverSelectMode なら必ず handleLiverClick を呼んで return ★★★
    if (clickMode_ == ClickMode::Liver) {
        if (liver_ && liver_->isSkeletonBound()) {
            glm::vec3 hitPos;
            int hitTriIndex;
            int hitVertex = raycastToVertex(liver_, mouseX, mouseY, view, projection,
                                            cameraPos, windowWidth, windowHeight, hitPos, hitTriIndex);

            if (hitVertex >= 0) {
                processLiverClick(hitVertex, hitPos, hitTriIndex, shiftPressed);
            } else {
                std::cout << "No liver mesh hit" << std::endl;  // ★★★ Liverのみメッセージ出力 ★★★
            }
        }
        return;  // ★★★ clickMode_ == ClickMode::Liver なら必ずreturn ★★★
    }

    // ★★★ オリジナルと同じ: portalSelectMode || manualExtendMode || shiftPressed ★★★
    if (clickMode_ == ClickMode::Portal || manualExtendMode_ || shiftPressed) {
        if (portal_ && portal_->isSkeletonBound()) {
            glm::vec3 hitPos;
            int hitTriIndex;
            int hitVertex = raycastToVertex(portal_, mouseX, mouseY, view, projection,
                                            cameraPos, windowWidth, windowHeight, hitPos, hitTriIndex);

            if (hitVertex >= 0) {
                processPortalClick(hitVertex, hitPos, hitTriIndex, shiftPressed);
            }
            // ★★★ Portalはヒットなしメッセージなし ★★★
        }
        return;  // ★★★ この条件なら必ずreturn ★★★
    }
}

//=============================================================================
// カッターヒット処理
//=============================================================================

void CutSegmentManager::updateFromCutterHit(const glm::vec3& hitPosition, bool isLiver, bool isPortal) {
    info_.valid = false;
    info_.hitPosition = hitPosition;
    info_.selectedSegments.clear();
    info_.selectedBranches.clear();
    info_.isLiverHit = isLiver;
    info_.isPortalHit = isPortal;

    if (!skeleton_) {
        std::cout << "[CutSegmentManager] No skeleton available" << std::endl;
        return;
    }

    SoftBodyGPUDuo* targetMesh = nullptr;
    std::string targetName;
    if (isLiver && liver_ && liver_->isSkeletonBound()) {
        targetMesh = liver_;
        targetName = "Liver";
    } else if (isPortal && portal_ && portal_->isSkeletonBound()) {
        targetMesh = portal_;
        targetName = "Portal";
    } else {
        std::cout << "[CutSegmentManager] No valid target mesh" << std::endl;
        return;
    }

    int hitVertex = findNearestSmoothSurfaceVertex(targetMesh, hitPosition);

    if (hitVertex < 0) {
        std::cout << "[CutSegmentManager] Could not find nearest vertex" << std::endl;
        return;
    }

    info_.nearestVertex = hitVertex;

    std::cout << "\n=== CutSegmentManager Update ===" << std::endl;
    std::cout << "  Target: " << targetName << std::endl;
    std::cout << "  Mode: " << getCutModeName() << std::endl;
    std::cout << "  Hit position: (" << hitPosition.x << ", " << hitPosition.y << ", " << hitPosition.z << ")" << std::endl;
    std::cout << "  Hit vertex: " << hitVertex << std::endl;

    if (cutMode_ == CutMode::Skeleton) {
        int segId = targetMesh->getVertexSegmentId(hitVertex);
        std::cout << "  Skeleton Segment ID: " << segId << std::endl;

        if (segId >= 0) {
            info_.selectedSegments = skeleton_->getDownstreamSegments(segId);
            info_.selectedSegments.insert(segId);
            std::cout << "  Selected segments: " << info_.selectedSegments.size() << std::endl;
            info_.valid = true;
        }

    } else if (cutMode_ == CutMode::Voronoi3D) {
        glm::vec3 restPos = targetMesh->getRestVertexPosition(hitVertex);
        std::cout << "  Rest position: (" << restPos.x << ", " << restPos.y << ", " << restPos.z << ")" << std::endl;

        const auto* voronoi = skeleton_->getVoronoiSegmenter();
        if (voronoi) {
            info_.selectedBranches = voronoi->getBranchesAtPosition(restPos);
            std::cout << "  Voronoi branches: " << info_.selectedBranches.size() << std::endl;

            if (!info_.selectedBranches.empty()) {
                info_.valid = true;
            }
        }
    }
    printSelectedLiverVolumeInfo();
    std::cout << "================================\n" << std::endl;
}

void CutSegmentManager::cycleCutMode() {
    switch (cutMode_) {
    case CutMode::None:
        cutMode_ = CutMode::Skeleton;
        applyBaseColorOverlay();  // ★追加：全体をベースカラーで塗る
        break;
    case CutMode::Skeleton:
        cutMode_ = CutMode::Voronoi3D;
        applyBaseColorOverlay();  // ★追加
        break;
    case CutMode::Voronoi3D:
        cutMode_ = CutMode::None;
        // resetCutOverlay() は main.cpp で呼ばれる
        break;
    }
}

void CutSegmentManager::applyCutOverlay() {
    if (!info_.valid) return;

    if (cutMode_ == CutMode::Skeleton && !info_.selectedSegments.empty()) {
        if (liver_ && liver_->isSkeletonBound()) {
            liver_->applyCutSegmentOverlaySkeleton(
                info_.selectedSegments, *skeleton_, liverBaseColor_, liverHighlightColor_);
        }
        if (portal_ && portal_->isSkeletonBound()) {
            portal_->applyCutSegmentOverlaySkeleton(
                info_.selectedSegments, *skeleton_, portalBaseColor_, portalHighlightColor_);
        }

    } else if (cutMode_ == CutMode::Voronoi3D && !info_.selectedBranches.empty()) {
        if (liver_ && liver_->isSkeletonBound()) {
            liver_->applyCutSegmentOverlayVoronoi(
                info_.selectedBranches, *skeleton_, liverBaseColor_, liverHighlightColor_);
        }
        if (portal_ && portal_->isSkeletonBound()) {
            portal_->applyCutSegmentOverlayVoronoi(
                info_.selectedBranches, *skeleton_, portalBaseColor_, portalHighlightColor_);
        }
    }
}

void CutSegmentManager::applyBaseColorOverlay() {
    if (!skeleton_) return;

    // 空のセグメント選択 → 全体がベースカラーになる
    std::set<int> emptySegments;
    std::vector<int> emptyBranches;

    if (cutMode_ == CutMode::Skeleton) {
        if (liver_ && liver_->isSkeletonBound()) {
            liver_->applyCutSegmentOverlaySkeleton(
                emptySegments, *skeleton_, liverBaseColor_, liverHighlightColor_);
        }
        if (portal_ && portal_->isSkeletonBound()) {
            portal_->applyCutSegmentOverlaySkeleton(
                emptySegments, *skeleton_, portalBaseColor_, portalHighlightColor_);
        }
    } else if (cutMode_ == CutMode::Voronoi3D) {
        if (liver_ && liver_->isSkeletonBound()) {
            liver_->applyCutSegmentOverlayVoronoi(
                emptyBranches, *skeleton_, liverBaseColor_, liverHighlightColor_);
        }
        if (portal_ && portal_->isSkeletonBound()) {
            portal_->applyCutSegmentOverlayVoronoi(
                emptyBranches, *skeleton_, portalBaseColor_, portalHighlightColor_);
        }
    }
}

void CutSegmentManager::resetCutOverlay() {
    info_.valid = false;
    info_.selectedSegments.clear();
    info_.selectedBranches.clear();

    // ★★★ 肝臓をリセット ★★★
    if (liver_) {
        liver_->resetToDefaultColors();
    }

    // ★★★ 門脈もリセット ★★★
    if (portal_) {
        portal_->resetToDefaultColors();
    }
}


// CutSegmentManager.cpp
void CutSegmentManager::printSelectedLiverVolumeInfo() {
    if (!liver_ || cutMode_ != CutMode::Skeleton || info_.selectedSegments.empty()) return;

    // 選択セグメントのオリジナル体積
    float originalSelected = 0.0f;
    for (int segId : info_.selectedSegments) {
        originalSelected += liver_->getOriginalSegmentVolume(segId);
    }

    // 選択セグメントの現在体積（VALID）
    float currentSelected = 0.0f;
    for (int segId : info_.selectedSegments) {
        currentSelected += liver_->calculateSegmentVolume(segId, false);
    }

    float originalTotal = liver_->getOriginalTotalVolume();
    float ratio = (originalTotal > 0) ? (originalSelected / originalTotal * 100.0f) : 0.0f;

    std::cout << "\n[Skeleton] Selected " << info_.selectedSegments.size() << " segments" << std::endl;
    std::cout << "  Original: " << (originalSelected / 1000.0f) << " cm³" << std::endl;
    std::cout << "  Current:  " << (currentSelected / 1000.0f) << " cm³" << std::endl;
    std::cout << "  Total:    " << (originalTotal / 1000.0f) << " cm³" << std::endl;
    std::cout << "  Ratio:    " << ratio << "%" << std::endl;
}
