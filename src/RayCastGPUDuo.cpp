#include "RayCastGPUDuo.h"
#include "SoftBodyGPUDuo.h"
#include <iostream>


RayCastGPUDuo::Ray RayCastGPUDuo::screenToRay(float screenX, float screenY,
                                   const glm::mat4& view,
                                   const glm::mat4& projection,
                                   const glm::vec4& viewport) {
    float ndcX = (2.0f * screenX) / viewport.z - 1.0f;
    float ndcY = 1.0f - (2.0f * screenY) / viewport.w;
    
    glm::vec4 nearPoint = glm::vec4(ndcX, ndcY, -1.0f, 1.0f);
    glm::vec4 farPoint = glm::vec4(ndcX, ndcY, 1.0f, 1.0f);
    
    glm::mat4 invVP = glm::inverse(projection * view);
    
    glm::vec4 worldNear = invVP * nearPoint;
    glm::vec4 worldFar = invVP * farPoint;
    
    worldNear /= worldNear.w;
    worldFar /= worldFar.w;
    
    Ray ray;
    ray.origin = glm::vec3(worldNear);
    ray.direction = glm::normalize(glm::vec3(worldFar - worldNear));
    
    return ray;
}

RayCastGPUDuo::RayHit RayCastGPUDuo::intersectMesh(const Ray& ray, SoftBodyGPUDuo& mesh) {
    RayHit result = { false, std::numeric_limits<float>::max(), glm::vec3(0), nullptr };

    auto& positions = mesh.getLowResPositions();
    auto& surfaceTriIds = mesh.getLowResMeshData().tetSurfaceTriIds;

    float t, u, v;
    for (size_t i = 0; i < surfaceTriIds.size(); i += 3) {
        int idx1 = surfaceTriIds[i];
        int idx2 = surfaceTriIds[i + 1];
        int idx3 = surfaceTriIds[i + 2];
        
        glm::vec3 v1(positions[idx1 * 3], positions[idx1 * 3 + 1], positions[idx1 * 3 + 2]);
        glm::vec3 v2(positions[idx2 * 3], positions[idx2 * 3 + 1], positions[idx2 * 3 + 2]);
        glm::vec3 v3(positions[idx3 * 3], positions[idx3 * 3 + 1], positions[idx3 * 3 + 2]);
        
        if (rayTriangleIntersect(ray.origin, ray.direction, v1, v2, v3, t, u, v)) {
            if (t < result.distance) {
                result.hit = true;
                result.distance = t;
                result.position = ray.origin + ray.direction * t;
                result.hitObject = &mesh;
            }
        }
    }
    
    return result;
}


RayCastGPUDuo::RayHitTri RayCastGPUDuo::intersectMesh(const Ray& ray, 
                                          std::vector<GLfloat> vertices, 
                                          std::vector<GLuint> indices) {
    RayHitTri result = { false, std::numeric_limits<float>::max(), glm::vec3(0) };
    
    float t, u, v;
    for (size_t i = 0; i < indices.size(); i += 3) {
        int idx1 = indices[i];
        int idx2 = indices[i + 1];
        int idx3 = indices[i + 2];
        
        glm::vec3 v1(vertices[idx1 * 3], vertices[idx1 * 3 + 1], vertices[idx1 * 3 + 2]);
        glm::vec3 v2(vertices[idx2 * 3], vertices[idx2 * 3 + 1], vertices[idx2 * 3 + 2]);
        glm::vec3 v3(vertices[idx3 * 3], vertices[idx3 * 3 + 1], vertices[idx3 * 3 + 2]);
        
        if (rayTriangleIntersect(ray.origin, ray.direction, v1, v2, v3, t, u, v)) {
            if (t < result.distance) {
                result.hit = true;
                result.distance = t;
                result.position = ray.origin + ray.direction * t;
            }
        }
    }
    
    return result;
}

bool RayCastGPUDuo::rayTriangleIntersect(
    const glm::vec3& rayOrigin,
    const glm::vec3& rayDir,
    const glm::vec3& v0,
    const glm::vec3& v1,
    const glm::vec3& v2,
    float& t, float& u, float& v)
{
    const float EPSILON = 0.0000001f;
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 h = glm::cross(rayDir, edge2);
    float a = glm::dot(edge1, h);
    
    if (a > -EPSILON && a < EPSILON) return false;
    
    float f = 1.0f / a;
    glm::vec3 s = rayOrigin - v0;
    u = f * glm::dot(s, h);
    
    if (u < 0.0f || u > 1.0f) return false;
    
    glm::vec3 q = glm::cross(s, edge1);
    v = f * glm::dot(rayDir, q);
    
    if (v < 0.0f || u + v > 1.0f) return false;
    
    t = f * glm::dot(edge2, q);
    return t > EPSILON;
}





// RayCastGPUDuo.cpp に追加

RayCastGPUDuo::SoftBodyHitResult RayCastGPUDuo::FindHitInSoftBodies(
    float screenX, float screenY,
    const std::vector<SoftBodyGPUDuo*>& meshes,
    const glm::mat4& view,
    const glm::mat4& projection,
    const glm::vec3& cameraPos,
    int windowWidth, int windowHeight,
    bool useSmooth) {

    SoftBodyHitResult result;

    Ray worldRay = screenToRay(screenX, screenY, view, projection,
                               glm::vec4(0, 0, windowWidth, windowHeight));

    struct HitInfo {
        int meshIndex;
        float cameraDistance;
        glm::vec3 hitPosition;
        SoftBodyGPUDuo* hitObject;

        bool operator<(const HitInfo& other) const {
            return cameraDistance < other.cameraDistance;
        }
    };

    std::vector<HitInfo> allHits;

    for (size_t meshIdx = 0; meshIdx < meshes.size(); ++meshIdx) {
        SoftBodyGPUDuo* mesh = meshes[meshIdx];
        if (mesh == nullptr) continue;

        // スムースメッシュまたはローレゾメッシュを使用
        std::vector<float> vertices;
        std::vector<int> triIds;

        if (useSmooth && mesh->smoothDisplayMode &&
            !mesh->smoothedVertices.empty() &&
            !mesh->smoothSurfaceTriIds.empty()) {
            // スムースメッシュを使用
            vertices = mesh->smoothedVertices;
            triIds = mesh->smoothSurfaceTriIds;
        } else {
            // ローレゾメッシュを使用
            vertices = mesh->getLowResPositions();
            triIds = mesh->getLowResMeshData().tetSurfaceTriIds;
        }

        if (vertices.empty() || triIds.empty()) continue;

        std::vector<GLfloat> glVertices(vertices.begin(), vertices.end());
        std::vector<GLuint> glIndices(triIds.begin(), triIds.end());

        RayHitTri hit = intersectMesh(worldRay, glVertices, glIndices);

        if (hit.hit) {
            HitInfo info;
            info.meshIndex = static_cast<int>(meshIdx);
            info.hitPosition = hit.position;
            info.cameraDistance = glm::distance(cameraPos, hit.position);
            info.hitObject = mesh;

            allHits.push_back(info);

            std::cout << "  Mesh[" << meshIdx << "] hit at distance "
                      << info.cameraDistance << std::endl;
        }
    }

    if (allHits.empty()) {
        std::cout << "No mesh hit detected" << std::endl;
        return result;
    }

    // カメラに最も近いヒットを選択
    std::sort(allHits.begin(), allHits.end());
    const HitInfo& closestHit = allHits[0];

    result.hit = true;
    result.meshIndex = closestHit.meshIndex;
    result.hitPosition = closestHit.hitPosition;
    result.distance = closestHit.cameraDistance;
    result.hitObject = closestHit.hitObject;

    std::cout << "Closest hit: Mesh[" << result.meshIndex << "] at distance "
              << result.distance << std::endl;

    return result;
}
