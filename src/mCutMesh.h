#ifndef MCUTMESH_H
#define MCUTMESH_H

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include "ShaderProgram.h"  // ShaderProgramクラスのヘッダー

// ShaderProgramクラスの前方宣言
class ShaderProgram;

struct mCutMesh {
    GLuint VAO, VBO, EBO,NBO;
    std::vector<GLfloat> mVertices;
    std::vector<GLfloat> mNormals;
    std::vector<GLuint> mIndices;
    int numFaces;
    glm::vec3 mColor;


    mCutMesh() : VAO(0), VBO(0), EBO(0), NBO(0), numFaces(0),
    mColor(0.7f, 0.7f, 0.7f) {}

    // ファイルパスから読み込むコンストラクタ（NEW!）
    mCutMesh(const char* filePath) : mCutMesh() {
        *this = loadMeshFromFile(filePath);
    }


    // メッシュを描画
    void draw(ShaderProgram& shader, float alpha) {
        // カラーのuniformを設定
        shader.setUniform("objectColor", glm::vec4(mColor,alpha));

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, mIndices.size(), GL_UNSIGNED_INT, 0);
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            std::cerr << "OpenGL error during drawing: " << err << std::endl;
        }
        glBindVertexArray(0);
    }

    // メッシュを描画
    // メッシュを描画

    // メッシュを描画（2つのalphaパラメータ版）
    void draw(ShaderProgram& shader, float alpha, float alpha2) {
        // ★★★ 重要：累積エラーをクリア ★★★
        while (glGetError() != GL_NO_ERROR) {}

        // useVertexColor = false にして objectColor を使う
        GLint loc = glGetUniformLocation(shader.getProgram(), "useVertexColor");
        if (loc != -1) {
            glUniform1i(loc, 0);
        }

        // objectColor (vec3) を設定
        shader.setUniform("objectColor", mColor);

        // ★★★ objectAlphaが存在する場合のみ設定 ★★★
        GLint alphaLoc = glGetUniformLocation(shader.getProgram(), "objectAlpha");
        if (alphaLoc != -1) {
            shader.setUniform("objectAlpha", alpha);
        }

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, mIndices.size(), GL_UNSIGNED_INT, 0);
        // エラーチェックを削除（または必要時のみ有効化）
        // GLenum err = glGetError();
        // if (err != GL_NO_ERROR) {
        //     std::cerr << "OpenGL error during drawing: " << err << std::endl;
        // }
        glBindVertexArray(0);
    }

    // MCUTからメッシュを読み込む

    static mCutMesh loadMeshFromFile(const char* filePath) {

        mCutMesh mesh;


        // ファイルを開く

        std::ifstream file(filePath);

        if (!file.is_open()) {

            std::cerr << "Error: Could not open file " << filePath << std::endl;

            return mesh;

        }


        // 一時的なデータ構造

        std::vector<glm::vec3> vertices;

        std::vector<std::vector<int>> faces;


        // OBJファイルを解析

        std::string line;

        while (std::getline(file, line)) {

            std::istringstream iss(line);

            std::string type;

            iss >> type;


            if (type == "v") {

                // 頂点データ

                float x, y, z;

                iss >> x >> y >> z;

                vertices.push_back(glm::vec3(x, y, z));

            }

            else if (type == "f") {

                // 面データ

                std::vector<int> face;

                std::string vertex;

                while (iss >> vertex) {

                    // 頂点インデックスを抽出（v/vt/vnの形式から最初の数字だけ）

                    size_t pos = vertex.find('/');

                    if (pos != std::string::npos) {

                        vertex = vertex.substr(0, pos);

                    }

                    // OBJの頂点インデックスは1始まりなので、0始まりに変換

                    face.push_back(std::stoi(vertex) - 1);

                }

                // 三角形の面だけを保存

                if (face.size() >= 3) {

                    // 多角形の三角形分割（ファンタイプ）

                    for (size_t i = 1; i < face.size() - 1; ++i) {

                        std::vector<int> triangle = {face[0], face[i], face[i + 1]};

                        faces.push_back(triangle);

                    }

                }

            }

        }

        file.close();


        // 頂点データをmCutMesh形式に変換

        mesh.mVertices.clear();

        mesh.mVertices.reserve(vertices.size() * 3);

        for (const auto& vertex : vertices) {

            mesh.mVertices.push_back(vertex.x);

            mesh.mVertices.push_back(vertex.y);

            mesh.mVertices.push_back(vertex.z);

        }


        // インデックスデータを変換

        mesh.mIndices.clear();

        mesh.mIndices.reserve(faces.size() * 3);

        for (const auto& face : faces) {

            for (int idx : face) {

                mesh.mIndices.push_back(static_cast<GLuint>(idx));

            }

        }


        // 面数の記録

        mesh.numFaces = faces.size();


        // デフォルトの色を設定

        mesh.mColor = glm::vec3(0.7f, 0.7f, 0.7f);


        return mesh;

    }

    // クリーンアップ
    void cleanup() {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &NBO);
        glDeleteBuffers(1, &EBO);
    }

};

// メッシュのスケールを調整する関数
inline void scaleMesh(mCutMesh& mesh, float scaleX, float scaleY, float scaleZ) {
    std::cout << "Scaling mesh by (" << scaleX << ", " << scaleY << ", " << scaleZ << ")..." << std::endl;

    // メッシュの中心を計算
    glm::vec3 center(0.0f);
    size_t vertexCount = mesh.mVertices.size() / 3;

    for (size_t i = 0; i < mesh.mVertices.size(); i += 3) {
        center.x += mesh.mVertices[i];
        center.y += mesh.mVertices[i + 1];
        center.z += mesh.mVertices[i + 2];
    }
    center /= static_cast<float>(vertexCount);

    std::cout << "Mesh center: (" << center.x << ", " << center.y << ", " << center.z << ")" << std::endl;

    // 各頂点をスケーリング（中心を基準に）
    for (size_t i = 0; i < mesh.mVertices.size(); i += 3) {
        // 中心からの相対位置を計算
        glm::vec3 relativePos(
            mesh.mVertices[i] - center.x,
            mesh.mVertices[i + 1] - center.y,
            mesh.mVertices[i + 2] - center.z
            );

        // スケーリングを適用
        relativePos.x *= scaleX;
        relativePos.y *= scaleY;
        relativePos.z *= scaleZ;

        // 中心位置に戻す
        mesh.mVertices[i] = relativePos.x + center.x;
        mesh.mVertices[i + 1] = relativePos.y + center.y;
        mesh.mVertices[i + 2] = relativePos.z + center.z;
    }

    std::cout << "Scaling complete." << std::endl;
}

// 均一スケール用のオーバーロード関数
inline void scaleMesh(mCutMesh& mesh, float uniformScale) {
    scaleMesh(mesh, uniformScale, uniformScale, uniformScale);
}

// メッシュを原点に移動する関数
inline void centerMeshToOrigin(mCutMesh& mesh) {
    std::cout << "Centering mesh to origin..." << std::endl;

    // メッシュの中心を計算
    glm::vec3 center(0.0f);
    size_t vertexCount = mesh.mVertices.size() / 3;

    for (size_t i = 0; i < mesh.mVertices.size(); i += 3) {
        center.x += mesh.mVertices[i];
        center.y += mesh.mVertices[i + 1];
        center.z += mesh.mVertices[i + 2];
    }
    center /= static_cast<float>(vertexCount);

    // 全頂点を移動して中心を原点に
    for (size_t i = 0; i < mesh.mVertices.size(); i += 3) {
        mesh.mVertices[i] -= center.x;
        mesh.mVertices[i + 1] -= center.y;
        mesh.mVertices[i + 2] -= center.z;
    }

    std::cout << "Mesh centered at origin (moved by: "
              << -center.x << ", " << -center.y << ", " << -center.z << ")" << std::endl;
}

inline void draw_AllmCutMeshes(const std::vector<mCutMesh>& meshes,
                        ShaderProgram& shader,
                        const glm::vec3& camPos,
                        const std::vector<glm::vec4>& meshColors) {
    shader.use();
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    // ソート用の構造体
    struct TriangleInfo {
        size_t meshIndex;
        size_t triangleIndex;
        float distance;
    };

    std::vector<TriangleInfo> allTriangles;

    // 距離計算
    for (size_t meshIdx = 0; meshIdx < meshes.size(); meshIdx++) {
        const auto& mesh = meshes[meshIdx];

        for (size_t triIdx = 0; triIdx < mesh.mIndices.size() / 3; triIdx++) {
            glm::vec3 center(0.0f);
            for (int v = 0; v < 3; v++) {
                size_t idx = mesh.mIndices[triIdx * 3 + v];
                center.x += mesh.mVertices[idx * 3];
                center.y += mesh.mVertices[idx * 3 + 1];
                center.z += mesh.mVertices[idx * 3 + 2];
            }
            center /= 3.0f;

            float distance = glm::length(camPos - center);
            allTriangles.push_back({meshIdx, triIdx, distance});
        }
    }

    // ソート
    std::sort(allTriangles.begin(), allTriangles.end(),
              [](const TriangleInfo& a, const TriangleInfo& b) {
                  return a.distance > b.distance;
              });

    // 描画（ここが重要な修正点！）
    GLuint lastVAO = -1;
    glm::vec4 lastColor;
    bool colorInitialized = false;

    for (const auto& tri : allTriangles) {
        const auto& mesh = meshes[tri.meshIndex];

        // VAO切り替え（必要時のみ）
        if (lastVAO != mesh.VAO) {
            glBindVertexArray(mesh.VAO);  // 既存のVAO使用
            lastVAO = mesh.VAO;
        }

        // 色設定（必要時のみ）
        glm::vec4 currentColor = meshColors[tri.meshIndex % meshColors.size()];
        if (!colorInitialized || lastColor != currentColor) {
            shader.setUniform("vertColor", currentColor);
            lastColor = currentColor;
            colorInitialized = true;
        }

        // 既存のEBOを使用してオフセット指定で描画
        // tempEBOは作らない！！
        glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT,
                       (void*)(tri.triangleIndex * 3 * sizeof(GLuint)));
    }

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glBindVertexArray(0);
}


inline void setUp(mCutMesh& srcMesh) {
    // エラーをクリア
    while (glGetError() != GL_NO_ERROR) {}

    // データの有効性チェック
    if (srcMesh.mVertices.empty() || srcMesh.mIndices.empty()) {
        std::cerr << "Error: Empty mesh data" << std::endl;
        return;
    }

    // Initialize vectors with the correct size
    std::vector<float> vertices(srcMesh.mVertices.size());
    std::vector<GLuint> indices(srcMesh.mIndices.size()); // GLuintに変更

    // Copy data from source mesh
    for(size_t i = 0; i < srcMesh.mVertices.size(); i++) {
        vertices[i] = srcMesh.mVertices[i];
    }

    for(size_t i = 0; i < srcMesh.mIndices.size(); i++) {
        indices[i] = static_cast<GLuint>(srcMesh.mIndices[i]); // 明示的キャスト
    }

    // インデックスの範囲チェック
    size_t vertexCount = vertices.size() / 3;
    for (size_t i = 0; i < indices.size(); i++) {
        if (indices[i] >= vertexCount) {
            std::cerr << "Error: Index out of range at " << i << ": " << indices[i]
                      << " (vertex count: " << vertexCount << ")" << std::endl;
            return;
        }
    }

    // Create normals vector with the correct size
    std::vector<float> normals(vertices.size(), 0.0f);

    for (size_t i = 0; i < indices.size(); i += 3) {
        // 三角形の頂点インデックス
        GLuint i0 = indices[i];
        GLuint i1 = indices[i + 1];
        GLuint i2 = indices[i + 2];

        // 三角形の頂点座標
        glm::vec3 v0(vertices[i0 * 3], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
        glm::vec3 v1(vertices[i1 * 3], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
        glm::vec3 v2(vertices[i2 * 3], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);

        // エッジベクトル
        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;

        // 法線計算（外積を正規化）
        glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));

        // 各頂点に法線を加算（後で正規化）
        for (GLuint idx : {i0, i1, i2}) {
            normals[idx * 3]     += normal.x;
            normals[idx * 3 + 1] += normal.y;
            normals[idx * 3 + 2] += normal.z;
        }
    }

    // 頂点ごとの法線を正規化
    for (size_t i = 0; i < normals.size(); i += 3) {
        glm::vec3 n(normals[i], normals[i + 1], normals[i + 2]);
        float length = glm::length(n);
        if (length > 0.0001f) { // 0除算を避けるためのチェック
            n /= length;
        } else {
            n = glm::vec3(0.0f, 1.0f, 0.0f); // デフォルト法線
        }
        normals[i]     = n.x;
        normals[i + 1] = n.y;
        normals[i + 2] = n.z;
    }

    // バッファの生成
    glGenVertexArrays(1, &srcMesh.VAO);
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glGenVertexArrays: " << err << std::endl;
        return;
    }

    glGenBuffers(1, &srcMesh.VBO);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glGenBuffers(VBO): " << err << std::endl;
        return;
    }

    glGenBuffers(1, &srcMesh.EBO);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glGenBuffers(EBO): " << err << std::endl;
        return;
    }

    glGenBuffers(1, &srcMesh.NBO);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glGenBuffers(NBO): " << err << std::endl;
        return;
    }

    glBindVertexArray(srcMesh.VAO);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glBindVertexArray: " << err << std::endl;
        return;
    }

    // 頂点バッファ
    glBindBuffer(GL_ARRAY_BUFFER, srcMesh.VBO);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glBindBuffer(VBO): " << err << std::endl;
        return;
    }

    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GLfloat), vertices.data(), GL_STATIC_DRAW);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glBufferData(VBO): " << err << std::endl;
        return;
    }

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glVertexAttribPointer(pos): " << err << std::endl;
        return;
    }

    glEnableVertexAttribArray(0);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glEnableVertexAttribArray(pos): " << err << std::endl;
        return;
    }

    // 法線バッファ
    glBindBuffer(GL_ARRAY_BUFFER, srcMesh.NBO);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glBindBuffer(NBO): " << err << std::endl;
        return;
    }

    glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(GLfloat), normals.data(), GL_STATIC_DRAW);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glBufferData(NBO): " << err << std::endl;
        return;
    }

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glVertexAttribPointer(normal): " << err << std::endl;
        return;
    }

    glEnableVertexAttribArray(1);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glEnableVertexAttribArray(normal): " << err << std::endl;
        return;
    }

    // インデックスバッファ
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, srcMesh.EBO);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glBindBuffer(EBO): " << err << std::endl;
        return;
    }

    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glBufferData(EBO): " << err << std::endl;
        return;
    }

    glBindVertexArray(0);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error after glBindVertexArray(0): " << err << std::endl;
    }
}


#endif // MCUTMESH_H
