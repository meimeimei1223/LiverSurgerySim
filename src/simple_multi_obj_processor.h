#ifndef SIMPLE_MULTI_OBJ_PROCESSOR_H
#define SIMPLE_MULTI_OBJ_PROCESSOR_H

/**
 * @file simple_multi_obj_processor.h
 * @brief 複数OBJメッシュの位置・スケール正規化処理
 * 
 * tinyobjloaderを使用してOBJファイルを読み書きし、
 * 親メッシュを基準に全メッシュを正規化します。
 * 
 * 使用方法:
 *   // main.cppなど、一箇所でのみ実装を有効化
 *   #define TINYOBJLOADER_IMPLEMENTATION
 *   #include "tiny_obj_loader.h"
 *   #include "simple_multi_obj_processor.h"
 */

#include "tiny_obj_loader.h"
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <cmath>
#include <algorithm>

namespace SimpleOBJ {

// ============================================================
// データ構造
// ============================================================

/**
 * @brief シンプルなOBJメッシュ構造体
 */
struct SimpleMesh {
    std::vector<std::array<double, 3>> vertices;
    std::vector<std::vector<int>> faces;  // 可変頂点数対応
    std::string name;
    
    void clear() {
        vertices.clear();
        faces.clear();
        name.clear();
    }
    
    bool empty() const {
        return vertices.empty();
    }
    
    size_t vertexCount() const { return vertices.size(); }
    size_t faceCount() const { return faces.size(); }
};

/**
 * @brief 変換情報（逆変換用に保持）
 */
struct TransformInfo {
    std::array<double, 3> translation = {0.0, 0.0, 0.0};  // 元の重心位置
    double scaleFactor = 1.0;                              // 適用したスケール係数
    double originalMaxDistance = 0.0;                      // 正規化前の最大距離
    
    std::string toString() const {
        std::ostringstream oss;
        oss << "translation: [" << translation[0] << ", " 
            << translation[1] << ", " << translation[2] << "], ";
        oss << "scaleFactor: " << scaleFactor << ", ";
        oss << "originalMaxDistance: " << originalMaxDistance;
        return oss.str();
    }
};

/**
 * @brief 処理結果
 */
struct ProcessingResult {
    bool success = false;
    TransformInfo transform;
    std::vector<std::string> processedFiles;
    std::string errorMessage;
};

// ============================================================
// OBJ読み込み・保存
// ============================================================

/**
 * @brief OBJファイルを読み込む
 */
inline bool loadOBJ(const std::string& path, SimpleMesh& mesh, bool triangulate = false) {
    mesh.clear();
    
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;
    
    // マテリアルディレクトリ
    std::filesystem::path filePath(path);
    std::string baseDir = filePath.parent_path().string();
    if (!baseDir.empty()) baseDir += "/";
    
    bool ret = tinyobj::LoadObj(
        &attrib, &shapes, &materials, &warn, &err,
        path.c_str(),
        baseDir.empty() ? nullptr : baseDir.c_str(),
        triangulate
    );
    
    if (!warn.empty()) {
        std::cerr << "[tinyobj warn] " << warn << std::endl;
    }
    if (!err.empty()) {
        std::cerr << "[tinyobj error] " << err << std::endl;
    }
    if (!ret) {
        return false;
    }
    
    // 頂点コピー
    mesh.vertices.reserve(attrib.vertices.size() / 3);
    for (size_t i = 0; i < attrib.vertices.size(); i += 3) {
        mesh.vertices.push_back({
            static_cast<double>(attrib.vertices[i]),
            static_cast<double>(attrib.vertices[i + 1]),
            static_cast<double>(attrib.vertices[i + 2])
        });
    }
    
    // 面コピー
    for (const auto& shape : shapes) {
        if (mesh.name.empty() && !shape.name.empty()) {
            mesh.name = shape.name;
        }
        
        size_t indexOffset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            int fv = shape.mesh.num_face_vertices[f];
            std::vector<int> face;
            face.reserve(fv);
            
            for (int v = 0; v < fv; v++) {
                face.push_back(shape.mesh.indices[indexOffset + v].vertex_index);
            }
            
            if (face.size() >= 3) {
                mesh.faces.push_back(std::move(face));
            }
            indexOffset += fv;
        }
    }
    
    return true;
}

/**
 * @brief OBJファイルを保存
 */
inline bool saveOBJ(const std::string& path, const SimpleMesh& mesh, int precision = 8) {
    // ディレクトリ作成
    std::filesystem::path outputPath(path);
    std::filesystem::path outputDir = outputPath.parent_path();
    if (!outputDir.empty() && !std::filesystem::exists(outputDir)) {
        try {
            std::filesystem::create_directories(outputDir);
        } catch (const std::exception& e) {
            std::cerr << "Error creating directory: " << e.what() << std::endl;
            return false;
        }
    }
    
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create file: " << path << std::endl;
        return false;
    }
    
    file << "# Simple OBJ Processor Output\n";
    file << "# Vertices: " << mesh.vertices.size() << "\n";
    file << "# Faces: " << mesh.faces.size() << "\n";
    
    if (!mesh.name.empty()) {
        file << "o " << mesh.name << "\n";
    }
    file << "\n";
    
    // 頂点出力
    file << std::fixed;
    file.precision(precision);
    for (const auto& v : mesh.vertices) {
        file << "v " << v[0] << " " << v[1] << " " << v[2] << "\n";
    }
    file << "\n";
    
    // 面出力（1-indexed）
    for (const auto& f : mesh.faces) {
        file << "f";
        for (int idx : f) {
            file << " " << (idx + 1);
        }
        file << "\n";
    }
    
    return true;
}

// ============================================================
// メッシュ操作ユーティリティ
// ============================================================

/**
 * @brief 重心計算
 */
inline std::array<double, 3> computeCentroid(const SimpleMesh& mesh) {
    std::array<double, 3> centroid = {0.0, 0.0, 0.0};
    if (mesh.vertices.empty()) return centroid;
    
    for (const auto& v : mesh.vertices) {
        centroid[0] += v[0];
        centroid[1] += v[1];
        centroid[2] += v[2];
    }
    
    double n = static_cast<double>(mesh.vertices.size());
    centroid[0] /= n;
    centroid[1] /= n;
    centroid[2] /= n;
    
    return centroid;
}

/**
 * @brief メッシュ移動
 */
inline void translateMesh(SimpleMesh& mesh, double dx, double dy, double dz) {
    for (auto& v : mesh.vertices) {
        v[0] += dx;
        v[1] += dy;
        v[2] += dz;
    }
}

inline void translateMesh(SimpleMesh& mesh, const std::array<double, 3>& delta) {
    translateMesh(mesh, delta[0], delta[1], delta[2]);
}

/**
 * @brief メッシュスケーリング（原点基準）
 */
inline void scaleMesh(SimpleMesh& mesh, double scale) {
    for (auto& v : mesh.vertices) {
        v[0] *= scale;
        v[1] *= scale;
        v[2] *= scale;
    }
}

/**
 * @brief 原点からの最大距離
 */
inline double computeMaxDistanceFromOrigin(const SimpleMesh& mesh) {
    double maxDist = 0.0;
    for (const auto& v : mesh.vertices) {
        double dist = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        maxDist = std::max(maxDist, dist);
    }
    return maxDist;
}

/**
 * @brief バウンディングボックス計算
 */
inline void computeBoundingBox(
    const SimpleMesh& mesh,
    std::array<double, 3>& minBound,
    std::array<double, 3>& maxBound) 
{
    if (mesh.vertices.empty()) {
        minBound = maxBound = {0.0, 0.0, 0.0};
        return;
    }
    
    minBound = maxBound = mesh.vertices[0];
    for (const auto& v : mesh.vertices) {
        for (int i = 0; i < 3; i++) {
            minBound[i] = std::min(minBound[i], v[i]);
            maxBound[i] = std::max(maxBound[i], v[i]);
        }
    }
}

// ============================================================
// メイン処理関数
// ============================================================

/**
 * @brief 複数OBJを一括処理（ファイルベース）
 * 
 * @param inputPaths  入力パス（最初が親メッシュ）
 * @param outputPaths 出力パス
 * @param normalizeRange 正規化範囲（デフォルト1.0）
 * @param verbose 詳細ログ
 * @return ProcessingResult
 */
inline ProcessingResult processMultipleOBJSimple(
    const std::vector<std::string>& inputPaths,
    const std::vector<std::string>& outputPaths,
    double normalizeRange = 1.0,
    bool verbose = true)
{
    ProcessingResult result;
    
    // バリデーション
    if (inputPaths.size() != outputPaths.size()) {
        result.errorMessage = "Input/output path count mismatch";
        std::cerr << "Error: " << result.errorMessage << std::endl;
        return result;
    }
    if (inputPaths.empty()) {
        result.errorMessage = "No input files";
        std::cerr << "Error: " << result.errorMessage << std::endl;
        return result;
    }
    if (normalizeRange <= 0.0) {
        result.errorMessage = "normalizeRange must be positive";
        std::cerr << "Error: " << result.errorMessage << std::endl;
        return result;
    }
    
    if (verbose) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Simple Multi-OBJ Processor" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Meshes: " << inputPaths.size() << std::endl;
        std::cout << "Range: [-" << normalizeRange << ", " << normalizeRange << "]" << std::endl;
        std::cout << "========================================\n" << std::endl;
    }
    
    // Step 1: 読み込み
    std::vector<SimpleMesh> meshes(inputPaths.size());
    
    for (size_t i = 0; i < inputPaths.size(); i++) {
        if (verbose) std::cout << "[Load] " << inputPaths[i] << std::endl;
        
        if (!loadOBJ(inputPaths[i], meshes[i])) {
            result.errorMessage = "Failed to load: " + inputPaths[i];
            std::cerr << "Error: " << result.errorMessage << std::endl;
            return result;
        }
        
        if (verbose) {
            std::cout << "       V:" << meshes[i].vertexCount() 
                      << " F:" << meshes[i].faceCount() << std::endl;
        }
    }
    
    // Step 2: 親メッシュの重心計算
    auto parentCentroid = computeCentroid(meshes[0]);
    result.transform.translation = parentCentroid;
    
    if (verbose) {
        std::cout << "\n[Parent Centroid] (" 
                  << parentCentroid[0] << ", "
                  << parentCentroid[1] << ", "
                  << parentCentroid[2] << ")" << std::endl;
    }
    
    // Step 3: 全メッシュを移動（親の重心→原点）
    for (auto& mesh : meshes) {
        translateMesh(mesh, -parentCentroid[0], -parentCentroid[1], -parentCentroid[2]);
    }
    if (verbose) std::cout << "[Translate] Done" << std::endl;
    
    // Step 4: 最大距離計算
    double maxDist = computeMaxDistanceFromOrigin(meshes[0]);
    result.transform.originalMaxDistance = maxDist;
    if (verbose) std::cout << "[Max Distance] " << maxDist << std::endl;
    
    // Step 5: スケーリング
    double scaleFactor = (maxDist > 1e-10) ? (normalizeRange / maxDist) : 1.0;
    result.transform.scaleFactor = scaleFactor;
    
    for (auto& mesh : meshes) {
        scaleMesh(mesh, scaleFactor);
    }
    if (verbose) std::cout << "[Scale] " << scaleFactor << std::endl;
    
    // Step 6: 確認
    if (verbose) {
        std::array<double, 3> minB, maxB;
        computeBoundingBox(meshes[0], minB, maxB);
        std::cout << "\n[Parent BBox]" << std::endl;
        std::cout << "  X: [" << minB[0] << ", " << maxB[0] << "]" << std::endl;
        std::cout << "  Y: [" << minB[1] << ", " << maxB[1] << "]" << std::endl;
        std::cout << "  Z: [" << minB[2] << ", " << maxB[2] << "]" << std::endl;
    }
    
    // Step 7: 保存
    if (verbose) std::cout << "\n[Saving]" << std::endl;
    
    for (size_t i = 0; i < meshes.size(); i++) {
        if (!saveOBJ(outputPaths[i], meshes[i])) {
            result.errorMessage = "Failed to save: " + outputPaths[i];
            std::cerr << "Error: " << result.errorMessage << std::endl;
            return result;
        }
        result.processedFiles.push_back(outputPaths[i]);
        if (verbose) std::cout << "  " << outputPaths[i] << std::endl;
    }
    
    result.success = true;
    
    if (verbose) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Complete!" << std::endl;
        std::cout << "Transform: " << result.transform.toString() << std::endl;
        std::cout << "========================================\n" << std::endl;
    }
    
    return result;
}

/**
 * @brief メモリ上で直接処理（ファイルI/Oなし）
 */
inline TransformInfo processMultipleOBJInMemory(
    std::vector<SimpleMesh>& meshes,
    double normalizeRange = 1.0)
{
    TransformInfo transform;
    if (meshes.empty()) return transform;
    
    // 親の重心→原点
    auto centroid = computeCentroid(meshes[0]);
    transform.translation = centroid;
    
    for (auto& mesh : meshes) {
        translateMesh(mesh, -centroid[0], -centroid[1], -centroid[2]);
    }
    
    // スケーリング
    double maxDist = computeMaxDistanceFromOrigin(meshes[0]);
    transform.originalMaxDistance = maxDist;
    
    if (maxDist > 1e-10) {
        transform.scaleFactor = normalizeRange / maxDist;
        for (auto& mesh : meshes) {
            scaleMesh(mesh, transform.scaleFactor);
        }
    }
    
    return transform;
}

/**
 * @brief 逆変換（元のスケール・位置に戻す）
 */
inline void applyInverseTransform(SimpleMesh& mesh, const TransformInfo& transform) {
    // スケールを戻す
    if (std::abs(transform.scaleFactor) > 1e-10) {
        scaleMesh(mesh, 1.0 / transform.scaleFactor);
    }
    // 位置を戻す
    translateMesh(mesh, transform.translation);
}

inline void applyInverseTransform(std::vector<SimpleMesh>& meshes, const TransformInfo& transform) {
    for (auto& mesh : meshes) {
        applyInverseTransform(mesh, transform);
    }
}

} // namespace SimpleOBJ

#endif // SIMPLE_MULTI_OBJ_PROCESSOR_H
