#pragma once
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <vector>
#include <limits>

class SoftBodyGPUDuo; // 前方宣言

class RayCastGPUDuo {
public:
    struct Ray {
        glm::vec3 origin;
        glm::vec3 direction;
    };

    struct RayHit {
        bool hit;
        float distance;
        glm::vec3 position;
        SoftBodyGPUDuo* hitObject;
    };

    struct RayHitTri {
        bool hit;
        float distance;
        glm::vec3 position;
    };

    // ★ 新しく追加 ★
    struct TetMeshHitResult {
        bool hit;
        int meshIndex;
        glm::vec3 hitPosition;
        float distance;

        TetMeshHitResult()
            : hit(false), meshIndex(-1), hitPosition(0.0f), distance(0.0f) {}
    };

    static Ray screenToRay(float screenX, float screenY,
                          const glm::mat4& view,
                          const glm::mat4& projection,
                          const glm::vec4& viewport);

    static RayHit intersectMesh(const Ray& ray, SoftBodyGPUDuo& mesh);

    static RayHitTri intersectMesh(const Ray& ray, 
                                   std::vector<GLfloat> vertices, 
                                   std::vector<GLuint> indices);


    // ★ SoftBodyGPUDuo用のヒット結果構造体
    struct SoftBodyHitResult {
        bool hit;
        int meshIndex;           // ヒットしたメッシュのインデックス（配列内の位置）
        glm::vec3 hitPosition;
        float distance;
        SoftBodyGPUDuo* hitObject;  // ヒットしたオブジェクトへのポインタ

        SoftBodyHitResult()
            : hit(false), meshIndex(-1), hitPosition(0.0f), distance(0.0f), hitObject(nullptr) {}
    };

    // ★ SoftBodyGPUDuo配列に対するヒット判定（スムースメッシュ対応）
    static SoftBodyHitResult FindHitInSoftBodies(
        float screenX, float screenY,
        const std::vector<SoftBodyGPUDuo*>& meshes,
        const glm::mat4& view,
        const glm::mat4& projection,
        const glm::vec3& cameraPos,
        int windowWidth, int windowHeight,
        bool useSmooth = true);  // trueならスムースメッシュを優先

private:
    static bool rayTriangleIntersect(
        const glm::vec3& rayOrigin,
        const glm::vec3& rayDir,
        const glm::vec3& v0,
        const glm::vec3& v1,
        const glm::vec3& v2,
        float& t, float& u, float& v);
};
