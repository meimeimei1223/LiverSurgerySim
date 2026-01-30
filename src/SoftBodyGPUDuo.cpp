#include "SoftBodyGPUDuo.h"
#include "VectorMath.h"
#include <iostream>
#include <algorithm>
#include <limits>
#include <fstream>
#include "ShaderProgram.h"
#include <sstream>  // â† ã“ã‚Œã‚’è¿½åŠ
#include "SoftBodyGPUDuo.h"
#include "VectorMath.h"
#include <iostream>
#include <algorithm>
#include <limits>
#include <fstream>
#include "ShaderProgram.h"
#include <sstream>
#include <chrono>
#include <iomanip>
#include <array>  // ← これを追加

SoftBodyGPUDuo::MeshData SoftBodyGPUDuo::ReadVertexAndFace(const std::string& objPath) {
    std::cout << "Reading OBJ file: " << objPath << std::endl;

    SoftBodyGPUDuo::MeshData meshData;
    std::vector<glm::vec3> vertices;
    std::vector<std::vector<int>> faces;

    std::ifstream file(objPath);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open file: " << objPath << std::endl;
        return meshData;
    }

    std::string line;
    int lineNumber = 0;

    while (std::getline(file, line)) {
        lineNumber++;

        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "v") {
            float x, y, z;
            if (iss >> x >> y >> z) {
                vertices.emplace_back(x, y, z);
            }
        }
        else if (type == "f") {
            std::vector<int> face;
            std::string vertex;

            while (iss >> vertex) {
                int vertexIndex = 0;

                size_t pos = vertex.find('/');
                if (pos != std::string::npos) {
                    vertex = vertex.substr(0, pos);
                }

                try {
                    vertexIndex = std::stoi(vertex);
                    if (vertexIndex > 0) {
                        face.push_back(vertexIndex - 1);
                    } else if (vertexIndex < 0) {
                        face.push_back(vertices.size() + vertexIndex);
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Failed to parse vertex index at line " << lineNumber << std::endl;
                }
            }

            if (face.size() == 3) {
                bool validFace = true;
                for (int idx : face) {
                    if (idx < 0 || idx >= static_cast<int>(vertices.size())) {
                        validFace = false;
                        break;
                    }
                }

                if (validFace) {
                    faces.push_back(face);
                }
            }
        }
    }

    file.close();

    std::cout << "OBJ file loaded successfully:" << std::endl;
    std::cout << "  Vertices: " << vertices.size() << std::endl;
    std::cout << "  Triangular faces: " << faces.size() << std::endl;

    meshData.verts.clear();
    meshData.verts.reserve(vertices.size() * 3);

    for (const auto& v : vertices) {
        meshData.verts.push_back(v.x);
        meshData.verts.push_back(v.y);
        meshData.verts.push_back(v.z);
    }

    meshData.tetSurfaceTriIds.clear();
    meshData.tetSurfaceTriIds.reserve(faces.size() * 3);

    for (const auto& face : faces) {
        for (int idx : face) {
            meshData.tetSurfaceTriIds.push_back(idx);
        }
    }

    return meshData;
}

// TET_FACE_INDICESã®ã‚¯ãƒ©ã‚¹å¤–å®šç¾©ï¼ˆå¿…é ˆï¼‰
constexpr int SoftBodyGPUDuo::TET_FACE_INDICES[4][3];

SoftBodyGPUDuo::SoftBodyGPUDuo(const MeshData& lowResTetMesh, const MeshData& highResTetMesh,
                   float edgeCompliance, float volCompliance)
    : lowResMeshData(lowResTetMesh)
    , highResMeshData(highResTetMesh)
    , lowResEdgeCompliance(edgeCompliance)
    , lowResVolCompliance(volCompliance)
    , lowRes_grabId(-1)
    , lowRes_grabInvMass(0.0f)
    , lowRes_damping(0.99f)
    , lowRes_grabOffset(0.0f, 0.0f, 0.0f)
    , useHighResMesh(true)
    , highResTetVAO(0)
    , highResTetVBO(0)
{
    std::cout << "\n=== SoftBody Constructor (Dual Tetrahedral Mesh) ===" << std::endl;
    std::cout << "LowRes TetMesh - Vertices: " << lowResTetMesh.verts.size() / 3
              << ", Tetrahedra: " << lowResTetMesh.tetIds.size() / 4 << std::endl;
    std::cout << "HighRes TetMesh - Vertices: " << highResTetMesh.verts.size() / 3
              << ", Tetrahedra: " << highResTetMesh.tetIds.size() / 4 << std::endl;

    numLowResParticles = lowResTetMesh.verts.size() / 3;
    numLowTets = lowResTetMesh.tetIds.size() / 4;
    numHighTets = highResTetMesh.tetIds.size() / 4;
    numHighResVerts = highResTetMesh.verts.size() / 3;

    lowRes_positions = lowResTetMesh.verts;
    lowRes_prevPositions = lowResTetMesh.verts;
    lowRes_velocities.resize(3 * numLowResParticles, 0.0f);

    lowRes_tetIds = lowResTetMesh.tetIds;
    lowRes_edgeIds = lowResTetMesh.tetEdgeIds;
    lowRes_restVols.resize(numLowTets, 0.0f);
    lowRes_edgeLengths.resize(lowRes_edgeIds.size() / 2, 0.0f);
    lowRes_invMasses.resize(numLowResParticles, 0.0f);

    highResTetIds = highResTetMesh.tetIds;
    highResEdgeIds = highResTetMesh.tetEdgeIds;

    // ã‚³ãƒ³ã‚¹ãƒˆãƒ©ã‚¯ã‚¿å†…ã§
    size_t numHighResTets = highResTetMesh.tetIds.size() / 4;
    highResTetValid.resize(numHighResTets, true);  // å…¨ã¦æœ‰åŠ¹ã§åˆæœŸåŒ–

    // æ–°ã—ã„ã‚³ãƒ³ã‚¹ãƒˆãƒ©ã‚¯ã‚¿å†…ã§
    highResValidTriangles = highResTetMesh.tetSurfaceTriIds;  // åˆæœŸã¯å…¨ã¦æœ‰åŠ¹

    // å…¨ã¦ã®å››é¢ä½“ã‚’æœ‰åŠ¹ã«ã™ã‚‹
    highResTetValid.resize(numHighTets, true);


    // ã‚³ãƒ³ã‚¹ãƒˆãƒ©ã‚¯ã‚¿å†…ã€ã¾ãŸã¯ initialize() é–¢æ•°å†…ã«è¿½åŠ
    // å…¨ã¦ã®å››é¢ä½“ã‚’æœ€åˆã¯æœ‰åŠ¹ã«ã™ã‚‹
    lowRes_tetValid.clear();
    lowRes_tetValid.resize(numLowTets, true);

    lowRes_tempBuffer.resize(4 * 3, 0.0f);
    lowRes_grads.resize(4 * 3, 0.0f);

    lowRes_edgeLambdas.resize(lowRes_edgeIds.size() / 2, 0.0f);
    lowRes_volLambdas.resize(numLowTets, 0.0f);


    // é«˜è§£åƒåº¦ãƒ¡ãƒƒã‚·ãƒ¥ã®åˆæœŸåŒ–
    highRes_positions = highResTetMesh.verts;
    skinningInfoLowToHigh.resize(4 * numHighResVerts, -1.0f);
    highResTetIds  = highResTetMesh.tetIds;
    highResInvalidatedCount = 0;
    // ã‚¹ã‚­ãƒ‹ãƒ³ã‚°æƒ…å ±ã®è¨ˆç®—ï¼ˆæœ€ã‚‚é‡ã„å‡¦ç†ï¼‰
    computeSkinningInfoLowToHigh(highResTetMesh.verts);

    // è¡¨é¢ãƒžãƒƒãƒ”ãƒ³ã‚°ã®åˆæœŸåŒ–
    initLowResSurfaceToTetMapping();

    // é«˜è§£åƒåº¦è¡¨é¢ãƒžãƒƒãƒ”ãƒ³ã‚°ã®åˆæœŸåŒ–ï¼ˆã‚‚ã—å®Ÿè£…ã•ã‚Œã¦ã„ã‚‹å ´åˆï¼‰
    if (!highResSurfaceTriToTet.empty() || highResMeshData.tetSurfaceTriIds.size() > 0) {
        initHighResSurfaceToTetMapping();
    }

    // saveInitialShape();
    saveLowResInitialShape();

    // ç‰©ç†åˆæœŸåŒ–
    initPhysicsLowRes();

    // OpenGLãƒãƒƒãƒ•ã‚¡ã®ã‚»ãƒƒãƒˆã‚¢ãƒƒãƒ— - ä½Žè§£åƒåº¦
    setupLowResMesh(lowResTetMesh.tetSurfaceTriIds);

    setupLowResTetMesh();

    // OpenGLãƒãƒƒãƒ•ã‚¡ã®ã‚»ãƒƒãƒˆã‚¢ãƒƒãƒ— - é«˜è§£åƒåº¦
    setupHighResMesh();

    setupHighResTetMesh();

    // æ³•ç·šãƒãƒƒãƒ•ã‚¡ã®ã‚»ãƒƒãƒˆã‚¢ãƒƒãƒ—
    setupNormalBuffer();

    showLowHighTetMesh = false;
    showHighResMesh = true;
    lowRes_modelMatrix = glm::mat4(1.0f);

    // ã‚¹ãƒˆãƒ¬ã‚¤ãƒ³ãƒªãƒŸãƒ†ã‚£ãƒ³ã‚°ç”¨é…åˆ—ã®åˆæœŸåŒ–ã‚’è¿½åŠ
    lowRes_edgeStrains.resize(lowRes_edgeIds.size() / 2, 1.0f);
    lowRes_volStrains.resize(numLowTets, 1.0f);
    lowRes_edgeStrainLevel.resize(lowRes_edgeIds.size() / 2, 0);
    lowRes_volStrainLevel.resize(numLowTets, 0);
    lowRes_edgeStiffnessScale.resize(lowRes_edgeIds.size() / 2, 1.0f);
    lowRes_volStiffnessScale.resize(numLowTets, 1.0f);

    // ãƒ‡ãƒ•ã‚©ãƒ«ãƒˆãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿ã®è¨­å®š
    skinningAdjustParams.enabled = true;           // æœ‰åŠ¹åŒ–
    skinningAdjustParams.blendFactor = 0.5f;       // è£œæ­£å¼·åº¦50%
    skinningAdjustParams.detectionThreshold = 0.1f; // æ¤œå‡ºé–¾å€¤
    skinningAdjustParams.maxIterations = 3;        // ã‚¹ãƒ ãƒ¼ã‚¸ãƒ³ã‚°åå¾©3å›ž

    // ã‚ªãƒªã‚¸ãƒŠãƒ«ä½ç½®ã‚’ä¿å­˜
    original_highRes_positions = highRes_positions;

    // éš£æŽ¥ãƒªã‚¹ãƒˆã¯æœ€åˆã®æ›´æ–°æ™‚ã«è‡ªå‹•æ§‹ç¯‰ã•ã‚Œã‚‹
    adjacencyListComputed = false;

    std::cout << "Skinning adjustment initialized with default parameters" << std::endl;
    std::cout << "  Enabled: " << (skinningAdjustParams.enabled ? "Yes" : "No") << std::endl;
    std::cout << "  Blend factor: " << skinningAdjustParams.blendFactor << std::endl;
    std::cout << "  Max iterations: " << skinningAdjustParams.maxIterations << std::endl;

    std::cout << "=== Constructor Complete ===" << std::endl;
}


SoftBodyGPUDuo::SoftBodyGPUDuo(const MeshData& lowResTetMesh, const MeshData& highResTetMesh,
                               float edgeCompliance, float volCompliance,
                               MeshPreset preset)
    : lowResMeshData(lowResTetMesh)
    , highResMeshData(highResTetMesh)
    , lowResEdgeCompliance(edgeCompliance)
    , lowResVolCompliance(volCompliance)
    , lowRes_grabId(-1)
    , lowRes_grabInvMass(0.0f)
    , lowRes_damping(0.99f)
    , lowRes_grabOffset(0.0f, 0.0f, 0.0f)
    , useHighResMesh(true)
    , highResTetVAO(0)
    , highResTetVBO(0)
{
    std::cout << "\n=== SoftBody Constructor (Dual Tetrahedral Mesh) ===" << std::endl;
    std::cout << "LowRes TetMesh - Vertices: " << lowResTetMesh.verts.size() / 3
              << ", Tetrahedra: " << lowResTetMesh.tetIds.size() / 4 << std::endl;
    std::cout << "HighRes TetMesh - Vertices: " << highResTetMesh.verts.size() / 3
              << ", Tetrahedra: " << highResTetMesh.tetIds.size() / 4 << std::endl;

    numLowResParticles = lowResTetMesh.verts.size() / 3;
    numLowTets = lowResTetMesh.tetIds.size() / 4;
    numHighTets = highResTetMesh.tetIds.size() / 4;
    numHighResVerts = highResTetMesh.verts.size() / 3;

    lowRes_positions = lowResTetMesh.verts;
    lowRes_prevPositions = lowResTetMesh.verts;
    lowRes_velocities.resize(3 * numLowResParticles, 0.0f);

    lowRes_tetIds = lowResTetMesh.tetIds;
    lowRes_edgeIds = lowResTetMesh.tetEdgeIds;
    lowRes_restVols.resize(numLowTets, 0.0f);
    lowRes_edgeLengths.resize(lowRes_edgeIds.size() / 2, 0.0f);
    lowRes_invMasses.resize(numLowResParticles, 0.0f);

    highResTetIds = highResTetMesh.tetIds;
    highResEdgeIds = highResTetMesh.tetEdgeIds;

    size_t numHighResTets = highResTetMesh.tetIds.size() / 4;
    highResTetValid.resize(numHighResTets, true);

    highResValidTriangles = highResTetMesh.tetSurfaceTriIds;

    highResTetValid.resize(numHighTets, true);

    lowRes_tetValid.clear();
    lowRes_tetValid.resize(numLowTets, true);

    lowRes_tempBuffer.resize(4 * 3, 0.0f);
    lowRes_grads.resize(4 * 3, 0.0f);

    lowRes_edgeLambdas.resize(lowRes_edgeIds.size() / 2, 0.0f);
    lowRes_volLambdas.resize(numLowTets, 0.0f);

    highRes_positions = highResTetMesh.verts;
    skinningInfoLowToHigh.resize(4 * numHighResVerts, -1.0f);
    highResTetIds = highResTetMesh.tetIds;
    highResInvalidatedCount = 0;
    computeSkinningInfoLowToHigh(highResTetMesh.verts);

    initLowResSurfaceToTetMapping();

    if (!highResSurfaceTriToTet.empty() || highResMeshData.tetSurfaceTriIds.size() > 0) {
        initHighResSurfaceToTetMapping();
    }

    saveLowResInitialShape();

    initPhysicsLowRes();

    setupLowResMesh(lowResTetMesh.tetSurfaceTriIds);
    setupLowResTetMesh();

    setupHighResMesh();
    setupHighResTetMesh();

    setupNormalBuffer();

    showLowHighTetMesh = false;
    showHighResMesh = true;
    lowRes_modelMatrix = glm::mat4(1.0f);

    lowRes_edgeStrains.resize(lowRes_edgeIds.size() / 2, 1.0f);
    lowRes_volStrains.resize(numLowTets, 1.0f);
    lowRes_edgeStrainLevel.resize(lowRes_edgeIds.size() / 2, 0);
    lowRes_volStrainLevel.resize(numLowTets, 0);
    lowRes_edgeStiffnessScale.resize(lowRes_edgeIds.size() / 2, 1.0f);
    lowRes_volStiffnessScale.resize(numLowTets, 1.0f);

    skinningAdjustParams.enabled = true;
    skinningAdjustParams.blendFactor = 0.5f;
    skinningAdjustParams.detectionThreshold = 0.1f;
    skinningAdjustParams.maxIterations = 3;

    original_highRes_positions = highRes_positions;

    adjacencyListComputed = false;

    std::cout << "Skinning adjustment initialized with default parameters" << std::endl;
    std::cout << "  Enabled: " << (skinningAdjustParams.enabled ? "Yes" : "No") << std::endl;
    std::cout << "  Blend factor: " << skinningAdjustParams.blendFactor << std::endl;
    std::cout << "  Max iterations: " << skinningAdjustParams.maxIterations << std::endl;

    //==========================================================================
    // ★★★ プリセット処理（新規追加部分） ★★★
    //==========================================================================

    // HighResを持たないLowRes四面体を無効化（常に実行）
    invalidateLowResTetsWithoutHighRes();

    // プリセット適用
    if (preset != MeshPreset::NONE) {
        switch (preset) {
            case MeshPreset::LIVER:
                applyConfig(getPresetLiver());
                printCreationStats("liver");
                break;
            case MeshPreset::VESSEL:
                applyConfig(getPresetVessel());
                printCreationStats("vessel");
                break;
            default:
                break;
        }

        // エッジ有効性計算
        computeEdgeValidity(preset == MeshPreset::LIVER ? "liver" : "vessel");
    }

    std::cout << "=== Constructor Complete ===" << std::endl;
}


void SoftBodyGPUDuo::setupHighResMesh() {
    if (highResVAO != 0) {
        glDeleteVertexArrays(1, &highResVAO);
        highResVAO = 0;
    }
    if (highResVBO != 0) {
        glDeleteBuffers(1, &highResVBO);
        highResVBO = 0;
    }
    if (highResEBO != 0) {
        glDeleteBuffers(1, &highResEBO);
        highResEBO = 0;
    }
    if (highResNormalVBO != 0) {
        glDeleteBuffers(1, &highResNormalVBO);
        highResNormalVBO = 0;
    }

    glGenVertexArrays(1, &highResVAO);
    glGenBuffers(1, &highResVBO);
    glGenBuffers(1, &highResEBO);
    glGenBuffers(1, &highResNormalVBO);

    glBindVertexArray(highResVAO);

    glBindBuffer(GL_ARRAY_BUFFER, highResVBO);
    glBufferData(GL_ARRAY_BUFFER, highRes_positions.size() * sizeof(float),
                 highRes_positions.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, highResNormalVBO);
    computeHighResNormals();
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);

    // ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹ï¼ˆä¿®æ­£ç®‡æ‰€ï¼‰
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, highResEBO);
    // highResValidTrianglesãŒåˆæœŸåŒ–ã•ã‚Œã¦ã„ãªã„å ´åˆã¯å…¨ã¦ã®ä¸‰è§’å½¢ã‚’ä½¿ç”¨
    if (highResValidTriangles.empty()) {
        highResValidTriangles = highResMeshData.tetSurfaceTriIds;
    }
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 highResValidTriangles.size() * sizeof(int),
                 highResValidTriangles.data(),
                 GL_DYNAMIC_DRAW);  // GL_STATIC_DRAWã‹ã‚‰GL_DYNAMIC_DRAWã«å¤‰æ›´

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void SoftBodyGPUDuo::drawHighResMesh(ShaderProgram& shader) {
    if (!showHighResMesh || !useHighResMesh) return;

    shader.use();
    glBindVertexArray(highResVAO);
    glDrawElements(GL_TRIANGLES, highResValidTriangles.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void SoftBodyGPUDuo::setupHighResTetMesh() {
    if (highResTetVAO != 0) {
        glDeleteVertexArrays(1, &highResTetVAO);
        highResTetVAO = 0;
    }
    if (highResTetVBO != 0) {
        glDeleteBuffers(1, &highResTetVBO);
        highResTetVBO = 0;
    }

    glGenVertexArrays(1, &highResTetVAO);
    glGenBuffers(1, &highResTetVBO);

    glBindVertexArray(highResTetVAO);

    std::vector<float> edgeVertices;
    highResValidEdges.clear();

    std::set<std::pair<int, int>> processedEdges;  // é‡è¤‡ã‚’é¿ã‘ã‚‹ãŸã‚

    // å„å››é¢ä½“ã‹ã‚‰ã‚¨ãƒƒã‚¸ã‚’åŽé›†
    size_t numHighResTets = highResMeshData.tetIds.size() / 4;
    for (size_t t = 0; t < numHighResTets; t++) {
        // ç„¡åŠ¹ãªå››é¢ä½“ã¯ã‚¹ã‚­ãƒƒãƒ—
        if (!highResTetValid.empty() && !highResTetValid[t]) continue;

        int v0 = highResMeshData.tetIds[t * 4];
        int v1 = highResMeshData.tetIds[t * 4 + 1];
        int v2 = highResMeshData.tetIds[t * 4 + 2];
        int v3 = highResMeshData.tetIds[t * 4 + 3];

        // å››é¢ä½“ã®6ã¤ã®ã‚¨ãƒƒã‚¸
        std::vector<std::pair<int, int>> tetEdges = {
            {v0, v1}, {v0, v2}, {v0, v3},
            {v1, v2}, {v1, v3}, {v2, v3}
        };

        for (auto& edge : tetEdges) {
            // ã‚¨ãƒƒã‚¸ã‚’æ­£è¦åŒ–ï¼ˆå°ã•ã„æ–¹ã‚’å…ˆã«ï¼‰
            if (edge.first > edge.second) {
                std::swap(edge.first, edge.second);
            }

            // ã¾ã å‡¦ç†ã—ã¦ã„ãªã„ã‚¨ãƒƒã‚¸ãªã‚‰è¿½åŠ
            if (processedEdges.find(edge) == processedEdges.end()) {
                processedEdges.insert(edge);

                // é ‚ç‚¹ãƒ‡ãƒ¼ã‚¿ã‚’è¿½åŠ
                edgeVertices.push_back(highRes_positions[edge.first * 3]);
                edgeVertices.push_back(highRes_positions[edge.first * 3 + 1]);
                edgeVertices.push_back(highRes_positions[edge.first * 3 + 2]);

                edgeVertices.push_back(highRes_positions[edge.second * 3]);
                edgeVertices.push_back(highRes_positions[edge.second * 3 + 1]);
                edgeVertices.push_back(highRes_positions[edge.second * 3 + 2]);

                // ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹ã‚’è¨˜éŒ²
                highResValidEdges.push_back(edge.first);
                highResValidEdges.push_back(edge.second);
            }
        }
    }

    // VBOã«ãƒ‡ãƒ¼ã‚¿ã‚’ã‚¢ãƒƒãƒ—ãƒ­ãƒ¼ãƒ‰
    glBindBuffer(GL_ARRAY_BUFFER, highResTetVBO);
    glBufferData(GL_ARRAY_BUFFER, edgeVertices.size() * sizeof(float),
                 edgeVertices.data(), GL_DYNAMIC_DRAW);

    // é ‚ç‚¹å±žæ€§ã‚’è¨­å®š
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    std::cout << "High-res tet mesh edges: " << highResValidEdges.size() / 2
              << " (from " << processedEdges.size() << " unique edges)" << std::endl;
}

void SoftBodyGPUDuo::updateHighResTetMesh() {
    if (!useHighResMesh || highResTetVAO == 0) return;
    std::vector<float> edgeVertices;
    std::set<std::pair<int, int>> processedEdges;
    highResValidEdges.clear();
    size_t numHighResTets = highResMeshData.tetIds.size() / 4;
    for (size_t t = 0; t < numHighResTets; t++) {
        if (!highResTetValid.empty() && !highResTetValid[t]) continue;

        int v0 = highResMeshData.tetIds[t * 4];
        int v1 = highResMeshData.tetIds[t * 4 + 1];
        int v2 = highResMeshData.tetIds[t * 4 + 2];
        int v3 = highResMeshData.tetIds[t * 4 + 3];
        std::vector<std::pair<int, int>> tetEdges = {
            {v0, v1}, {v0, v2}, {v0, v3},
            {v1, v2}, {v1, v3}, {v2, v3}
        };

        for (auto& edge : tetEdges) {
            if (edge.first > edge.second) {
                std::swap(edge.first, edge.second);
            }
            if (processedEdges.find(edge) == processedEdges.end()) {
                processedEdges.insert(edge);
                edgeVertices.push_back(highRes_positions[edge.first * 3]);
                edgeVertices.push_back(highRes_positions[edge.first * 3 + 1]);
                edgeVertices.push_back(highRes_positions[edge.first * 3 + 2]);

                edgeVertices.push_back(highRes_positions[edge.second * 3]);
                edgeVertices.push_back(highRes_positions[edge.second * 3 + 1]);
                edgeVertices.push_back(highRes_positions[edge.second * 3 + 2]);

                highResValidEdges.push_back(edge.first);
                highResValidEdges.push_back(edge.second);
            }
        }
    }
    glBindBuffer(GL_ARRAY_BUFFER, highResTetVBO);
    glBufferData(GL_ARRAY_BUFFER, edgeVertices.size() * sizeof(float),
                 edgeVertices.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void SoftBodyGPUDuo::drawHighResTetMesh(ShaderProgram& shader) {
    if (!showHighResMesh || !useHighResMesh || highResTetVAO == 0) return;
    if (highResValidEdges.empty()) return;

    shader.use();
    glBindVertexArray(highResTetVAO);

    // ãƒ¯ã‚¤ãƒ¤ãƒ¼ãƒ•ãƒ¬ãƒ¼ãƒ ãƒ¢ãƒ¼ãƒ‰ã§æç”»
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDrawArrays(GL_LINES, 0, highResValidEdges.size());  // æœ‰åŠ¹ãªã‚¨ãƒƒã‚¸æ•°ã®ã¿æç”»
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glBindVertexArray(0);
}

void SoftBodyGPUDuo::deleteLowHighBuffers() {
    // ä½Žè§£åƒåº¦ãƒ¡ãƒƒã‚·ãƒ¥ãƒãƒƒãƒ•ã‚¡
    glDeleteVertexArrays(1, &lowResVAO);
    lowResVAO = 0;
    glDeleteBuffers(1, &lowResVBO);
    lowResVBO = 0;
    glDeleteBuffers(1, &lowResEBO);
    lowResEBO = 0;
    if (lowResNormalVBO != 0) {
        glDeleteBuffers(1, &lowResNormalVBO);
        lowResNormalVBO = 0;
    }

    // ä½Žè§£åƒåº¦ãƒ¯ã‚¤ãƒ¤ãƒ¼ãƒ•ãƒ¬ãƒ¼ãƒ ãƒãƒƒãƒ•ã‚¡
    if (lowResTetVAO != 0) {
        glDeleteVertexArrays(1, &lowResTetVAO);
        lowResTetVAO = 0;
    }
    if (lowResTetVBO != 0) {
        glDeleteBuffers(1, &lowResTetVBO);
        lowResTetVBO = 0;
    }
    if (lowResTetEBO != 0) {
        glDeleteBuffers(1, &lowResTetEBO);
        lowResTetEBO = 0;
    }

    // é«˜è§£åƒåº¦ãƒ¡ãƒƒã‚·ãƒ¥ãƒãƒƒãƒ•ã‚¡
    if (highResVAO != 0) {
        glDeleteVertexArrays(1, &highResVAO);
        highResVAO = 0;
    }
    if (highResVBO != 0) {
        glDeleteBuffers(1, &highResVBO);
        highResVBO = 0;
    }
    if (highResEBO != 0) {
        glDeleteBuffers(1, &highResEBO);
        highResEBO = 0;
    }
    if (highResNormalVBO != 0) {
        glDeleteBuffers(1, &highResNormalVBO);
        highResNormalVBO = 0;
    }

    // é«˜è§£åƒåº¦ãƒ¯ã‚¤ãƒ¤ãƒ¼ãƒ•ãƒ¬ãƒ¼ãƒ ãƒãƒƒãƒ•ã‚¡
    if (highResTetVAO != 0) {
        glDeleteVertexArrays(1, &highResTetVAO);
        highResTetVAO = 0;
    }
    if (highResTetVBO != 0) {
        glDeleteBuffers(1, &highResTetVBO);
        highResTetVBO = 0;
    }


}

void SoftBodyGPUDuo::initLowResSurfaceToTetMapping() {
    lowResSurfaceTriToTet.resize(lowResMeshData.tetSurfaceTriIds.size() / 3, -1);

    for (size_t i = 0; i < lowResMeshData.tetSurfaceTriIds.size(); i += 3) {
        int id0 = lowResMeshData.tetSurfaceTriIds[i];
        int id1 = lowResMeshData.tetSurfaceTriIds[i + 1];
        int id2 = lowResMeshData.tetSurfaceTriIds[i + 2];

        for (size_t t = 0; t < numLowTets; t++) {
            int count = 0;
            for (int j = 0; j < 4; j++) {
                int vid = lowRes_tetIds[4*t + j];
                if (vid == id0 || vid == id1 || vid == id2) count++;
            }
            if (count == 3) {
                lowResSurfaceTriToTet[i/3] = t;
                break;
            }
        }
    }
}

void SoftBodyGPUDuo::initHighResSurfaceToTetMapping() {

    size_t numTris = highResMeshData.tetSurfaceTriIds.size() / 3;
    size_t numTets = highResMeshData.tetIds.size() / 4;
    highResSurfaceTriToTet.resize(numTris, -1);

    std::vector<std::vector<int>> vertexToTets(numHighResVerts);

    for (size_t t = 0; t < numTets; t++) {
        for (int j = 0; j < 4; j++) {
            int vid = highResMeshData.tetIds[4*t + j];
            if (vid >= 0 && vid < numHighResVerts) {
                vertexToTets[vid].push_back(t);
            }
        }
    }

    // ã‚¹ãƒ†ãƒƒãƒ—2: å„ä¸‰è§’å½¢ã«å¯¾ã—ã¦ã€é–¢é€£ã™ã‚‹å››é¢ä½“ã ã‘ã‚’ãƒã‚§ãƒƒã‚¯
    int unmapped = 0;

    for (size_t i = 0; i < numTris; i++) {
        int id0 = highResMeshData.tetSurfaceTriIds[i * 3];
        int id1 = highResMeshData.tetSurfaceTriIds[i * 3 + 1];
        int id2 = highResMeshData.tetSurfaceTriIds[i * 3 + 2];

        bool found = false;
        for (int tetIdx : vertexToTets[id0]) {
            // ã“ã®å››é¢ä½“ãŒid1ã¨id2ã‚‚å«ã‚€ã‹ç¢ºèª
            int count = 0;
            for (int j = 0; j < 4; j++) {
                int vid = highResMeshData.tetIds[4*tetIdx + j];
                if (vid == id0 || vid == id1 || vid == id2) {
                    count++;
                }
            }

            if (count == 3) {
                highResSurfaceTriToTet[i] = tetIdx;
                found = true;
                break;
            }
        }

        if (!found) {
            unmapped++;
        }

        if (i % (numTris / 20) == 0) {
            std::cout << "\rProgress: " << (i * 100 / numTris) << "%" << std::flush;
        }
    }

    std::cout << "\rProgress: 100%" << std::endl;

    if (unmapped > 0) {
        std::cout << "Warning: " << unmapped << " triangles could not be mapped to tetrahedra" << std::endl;
    }

}


void SoftBodyGPUDuo::setupLowResTetMesh() {

    if (lowResTetVAO != 0) {
        glDeleteVertexArrays(1, &lowResTetVAO);
        lowResTetVAO = 0;
    }
    if (lowResTetVBO != 0) {
        glDeleteBuffers(1, &lowResTetVBO);
        lowResTetVBO = 0;
    }

    glGenVertexArrays(1, &lowResTetVAO);
    glGenBuffers(1, &lowResTetVBO);
    glBindVertexArray(lowResTetVAO);

    std::vector<float> edgeVertices;
    std::set<std::pair<int, int>> processedEdges;
    int totalEdges = 0;
    int validEdges = 0;

    for (size_t t = 0; t < numLowTets; t++) {
        if (!lowRes_tetValid.empty() && !lowRes_tetValid[t]) {
            //std::cout << "  Skipping invalid tetrahedron " << t << std::endl;
            continue;
        }

        int v0 = lowRes_tetIds[t * 4];
        int v1 = lowRes_tetIds[t * 4 + 1];
        int v2 = lowRes_tetIds[t * 4 + 2];
        int v3 = lowRes_tetIds[t * 4 + 3];

        // å››é¢ä½“ã®6ã¤ã®ã‚¨ãƒƒã‚¸
        std::vector<std::pair<int, int>> tetEdges = {
            {v0, v1}, {v0, v2}, {v0, v3},
            {v1, v2}, {v1, v3}, {v2, v3}
        };

        for (auto& edge : tetEdges) {
            totalEdges++;

            // ã‚¨ãƒƒã‚¸ã‚’æ­£è¦åŒ–ï¼ˆå°ã•ã„æ–¹ã‚’å…ˆã«ï¼‰
            if (edge.first > edge.second) {
                std::swap(edge.first, edge.second);
            }

            // ã¾ã å‡¦ç†ã—ã¦ã„ãªã„ã‚¨ãƒƒã‚¸ãªã‚‰è¿½åŠ
            if (processedEdges.find(edge) == processedEdges.end()) {
                processedEdges.insert(edge);
                validEdges++;

                // é ‚ç‚¹ãƒ‡ãƒ¼ã‚¿ã‚’è¿½åŠ
                edgeVertices.push_back(lowRes_positions[edge.first * 3]);
                edgeVertices.push_back(lowRes_positions[edge.first * 3 + 1]);
                edgeVertices.push_back(lowRes_positions[edge.first * 3 + 2]);

                edgeVertices.push_back(lowRes_positions[edge.second * 3]);
                edgeVertices.push_back(lowRes_positions[edge.second * 3 + 1]);
                edgeVertices.push_back(lowRes_positions[edge.second * 3 + 2]);
            }
        }
    }


    // VBOã«ãƒ‡ãƒ¼ã‚¿ã‚’ã‚¢ãƒƒãƒ—ãƒ­ãƒ¼ãƒ‰
    glBindBuffer(GL_ARRAY_BUFFER, lowResTetVBO);
    glBufferData(GL_ARRAY_BUFFER, edgeVertices.size() * sizeof(float),
                 edgeVertices.data(), GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void SoftBodyGPUDuo::updateLowResTetMeshes() {
    std::vector<float> edgeVertices;
    std::set<std::pair<int, int>> processedEdges;
    for (size_t t = 0; t < numLowTets; t++) {
        if (!lowRes_tetValid.empty() && !lowRes_tetValid[t]) continue;

        int v0 = lowRes_tetIds[t * 4];
        int v1 = lowRes_tetIds[t * 4 + 1];
        int v2 = lowRes_tetIds[t * 4 + 2];
        int v3 = lowRes_tetIds[t * 4 + 3];

        std::vector<std::pair<int, int>> tetEdges = {
            {v0, v1}, {v0, v2}, {v0, v3},
            {v1, v2}, {v1, v3}, {v2, v3}
        };

        for (auto& edge : tetEdges) {
            if (edge.first > edge.second) {
                std::swap(edge.first, edge.second);
            }

            if (processedEdges.find(edge) == processedEdges.end()) {
                processedEdges.insert(edge);

                edgeVertices.push_back(lowRes_positions[edge.first * 3]);
                edgeVertices.push_back(lowRes_positions[edge.first * 3 + 1]);
                edgeVertices.push_back(lowRes_positions[edge.first * 3 + 2]);

                edgeVertices.push_back(lowRes_positions[edge.second * 3]);
                edgeVertices.push_back(lowRes_positions[edge.second * 3 + 1]);
                edgeVertices.push_back(lowRes_positions[edge.second * 3 + 2]);
            }
        }
    }

    glBindBuffer(GL_ARRAY_BUFFER, lowResTetVBO);
    glBufferData(GL_ARRAY_BUFFER, edgeVertices.size() * sizeof(float),
                 edgeVertices.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void SoftBodyGPUDuo::drawLowResTetMesh(ShaderProgram& shader) {
    if (!showLowHighTetMesh) return;

    shader.use();
    glBindVertexArray(lowResTetVAO);
    std::set<std::pair<int, int>> validEdges;
    for (size_t t = 0; t < numLowTets; t++) {
        if (!lowRes_tetValid.empty() && !lowRes_tetValid[t]) continue;

        int v0 = lowRes_tetIds[t * 4];
        int v1 = lowRes_tetIds[t * 4 + 1];
        int v2 = lowRes_tetIds[t * 4 + 2];
        int v3 = lowRes_tetIds[t * 4 + 3];

        std::vector<std::pair<int, int>> tetEdges = {
            {std::min(v0, v1), std::max(v0, v1)},
            {std::min(v0, v2), std::max(v0, v2)},
            {std::min(v0, v3), std::max(v0, v3)},
            {std::min(v1, v2), std::max(v1, v2)},
            {std::min(v1, v3), std::max(v1, v3)},
            {std::min(v2, v3), std::max(v2, v3)}
        };

        for (const auto& edge : tetEdges) {
            validEdges.insert(edge);
        }
    }

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDrawArrays(GL_LINES, 0, validEdges.size() * 2);  // å„ã‚¨ãƒƒã‚¸ã¯2é ‚ç‚¹
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glBindVertexArray(0);
}

void SoftBodyGPUDuo::lowResPreSolve(float dt, const glm::vec3& gravity) {
    for (size_t i = 0; i < numLowResParticles; i++) {
        if (lowRes_invMasses[i] == 0.0f) continue;
        VectorMath::vecAdd(lowRes_velocities, i, {gravity.x, gravity.y, gravity.z}, 0, dt);
        VectorMath::vecScale(lowRes_velocities, i, lowRes_damping);
        VectorMath::vecCopy(lowRes_prevPositions, i, lowRes_positions, i);
        VectorMath::vecAdd(lowRes_positions, i, lowRes_velocities, i, dt);
        if (lowRes_positions[3 * i + 1] < -2.0f) {
            VectorMath::vecCopy(lowRes_positions, i, lowRes_prevPositions, i);
            lowRes_positions[3 * i + 1] = -2.0f;
            lowRes_velocities[3 * i + 1] = 0.0f;  // -2.0fã§ã¯ãªã0.0fã«
        }
    }
}

void SoftBodyGPUDuo::solveLowResEdges(float compliance, float dt) {
    float alpha = compliance / (dt * dt);

    for (size_t i = 0; i < lowRes_edgeLengths.size(); i++) {
        if (lowRes_edgeLengths[i] == 0.0f) continue;

        int id0 = lowRes_edgeIds[2 * i];
        int id1 = lowRes_edgeIds[2 * i + 1];

        // â˜…ã“ã®éƒ¨åˆ†ã‚’è¿½åŠ
        bool belongsToValidTet = false;
        for (size_t t = 0; t < numLowTets; t++) {
            if (!lowRes_tetValid[t]) continue;
            bool hasId0 = false, hasId1 = false;
            for (int j = 0; j < 4; j++) {
                int vid = lowRes_tetIds[t * 4 + j];
                if (vid == id0) hasId0 = true;
                if (vid == id1) hasId1 = true;
            }
            if (hasId0 && hasId1) {
                belongsToValidTet = true;
                break;
            }
        }
        if (!belongsToValidTet) continue;  // â˜…ç„¡åŠ¹ãªå››é¢ä½“ã®ã‚¨ãƒƒã‚¸ã¯ã‚¹ã‚­ãƒƒãƒ—


        float w0 = lowRes_invMasses[id0];
        float w1 = lowRes_invMasses[id1];
        if (w0 == 0.0f && w1 == 0.0f) continue;

        float w = w0 + w1;
        if (w == 0.0f) continue;

        VectorMath::vecSetDiff(lowRes_grads, 0, lowRes_positions, id0, lowRes_positions, id1);
        float len = std::sqrt(VectorMath::vecLengthSquared(lowRes_grads, 0));
        if (len == 0.0f) continue;

        VectorMath::vecScale(lowRes_grads, 0, 1.0f / len);
        float restLen = lowRes_edgeLengths[i];
        float C = len - restLen;

        float dLambda = -(C + alpha * lowRes_edgeLambdas[i]) / (w + alpha);
        lowRes_edgeLambdas[i] += dLambda;

        VectorMath::vecAdd(lowRes_positions, id0, lowRes_grads, 0, dLambda * w0);
        VectorMath::vecAdd(lowRes_positions, id1, lowRes_grads, 0, -dLambda * w1);
    }
}

void SoftBodyGPUDuo::initPhysicsLowRes() {
    std::fill(lowRes_invMasses.begin(), lowRes_invMasses.end(), 0.0f);
    std::fill(lowRes_restVols.begin(), lowRes_restVols.end(), 0.0f);
    std::fill(lowRes_edgeLambdas.begin(), lowRes_edgeLambdas.end(), 0.0f);
    std::fill(lowRes_volLambdas.begin(), lowRes_volLambdas.end(), 0.0f);
    std::vector<int> validTetCount(numLowResParticles, 0);

    for (size_t i = 0; i < numLowTets; i++) {
        if (!lowRes_tetValid[i]) {
            continue;
        }

        float vol = lowResGetTetVolume(i);
        lowRes_restVols[i] = vol;
        float pInvMass = vol > 0.0f ? 1.0f / (vol / 4000000.0f) : 1000000.0f;

        for (int j = 0; j < 4; j++) {
            int vid = lowRes_tetIds[4 * i + j];
            lowRes_invMasses[vid] += pInvMass;
            validTetCount[vid]++;  // ã“ã®é ‚ç‚¹ãŒå±žã™ã‚‹æœ‰åŠ¹ãªå››é¢ä½“ã‚’ã‚«ã‚¦ãƒ³ãƒˆ
        }
    }
    for (size_t i = 0; i < numLowResParticles; i++) {
        if (validTetCount[i] == 0 && lowRes_invMasses[i] == 0.0f) {
            lowRes_invMasses[i] = 0.0f;  // å®Œå…¨ã«å›ºå®š
        }
    }

    // ã‚¨ãƒƒã‚¸ã®é•·ã•ã‚’è¨ˆç®—
    for (size_t i = 0; i < lowRes_edgeLengths.size(); i++) {
        int id0 = lowRes_edgeIds[2 * i];
        int id1 = lowRes_edgeIds[2 * i + 1];

        // ä¸¡æ–¹ã®é ‚ç‚¹ãŒæœ‰åŠ¹ãªå››é¢ä½“ã«å±žã™ã‚‹å ´åˆã®ã¿ã‚¨ãƒƒã‚¸ã‚’æœ‰åŠ¹ã«ã™ã‚‹
        if (validTetCount[id0] > 0 && validTetCount[id1] > 0) {
            lowRes_edgeLengths[i] = std::sqrt(VectorMath::vecDistSquared(lowRes_positions, id0, lowRes_positions, id1));
        } else {
            lowRes_edgeLengths[i] = 0.0f;  // ç„¡åŠ¹ãªã‚¨ãƒƒã‚¸
        }
    }


    size_t numEdges = lowRes_edgeIds.size() / 2;
    lowRes_edgeValid.resize(numEdges, true);

    lowRes_pinnedVertices.clear();

    lowRes_oldInvMasses.resize(numLowResParticles, 0.0f);

    // ★ 初期の逆質量を保存（カット時の高速化用）
        if (originalInvMasses.empty()) {
            originalInvMasses = lowRes_invMasses;
        }

}


/*
void SoftBodyGPUDuo::invalidateLowResTetrahedra(const std::vector<int>& tetIndices) {
    // ========== 1. グラブ状態の保存 ==========
    bool grabWasActive = (lowRes_grabId >= 0);
    int savedGrabId = lowRes_grabId;
    glm::vec3 savedGrabPosition;
    glm::vec3 savedGrabOffset = lowRes_grabOffset;
    std::vector<int> savedActiveParticles = lowRes_activeParticles;
    std::vector<float> savedOldInvMasses = lowRes_oldInvMasses;

    if (grabWasActive) {
        savedGrabPosition = glm::vec3(
            lowRes_positions[lowRes_grabId * 3],
            lowRes_positions[lowRes_grabId * 3 + 1],
            lowRes_positions[lowRes_grabId * 3 + 2]
            );
        std::cout << "[DEBUG] Saving grab state before invalidation:" << std::endl;
        std::cout << "  Grabbed vertex: " << lowRes_grabId << std::endl;
        std::cout << "  Position: (" << savedGrabPosition.x << ", "
                  << savedGrabPosition.y << ", " << savedGrabPosition.z << ")" << std::endl;
        std::cout << "  Active particles: " << lowRes_activeParticles.size() << std::endl;
    }

    // ========== 2. 四面体を無効化 ==========
    int invalidatedCount = 0;
    for (int idx : tetIndices) {
        if (idx >= 0 && idx < static_cast<int>(lowRes_tetValid.size())) {
            if (lowRes_tetValid[idx]) {
                lowRes_tetValid[idx] = false;
                invalidatedCount++;
            }
        }
    }
    std::cout << "  Invalidated " << invalidatedCount << " tetrahedra" << std::endl;

    // ========== 3. 質量の再計算（有効な四面体のみ）==========
    std::vector<float> savedInvMasses = lowRes_invMasses;

    int fixedBefore = 0;
    for (size_t i = 0; i < lowRes_invMasses.size(); i++) {
        if (lowRes_invMasses[i] == 0.0f) fixedBefore++;
    }
    std::cout << "[DEBUG] Before mass recalc: fixed=" << fixedBefore << std::endl;

    std::fill(lowRes_invMasses.begin(), lowRes_invMasses.end(), 0.0f);
    for (size_t i = 0; i < numLowTets; i++) {
        if (!lowRes_tetValid[i]) continue;

        float vol = lowResGetTetVolume(i);
        float pInvMass = vol > 0.0f ? 1.0f / (vol / 4000000.0f) : 1000000.0f;

        for (int j = 0; j < 4; j++) {
            lowRes_invMasses[lowRes_tetIds[4 * i + j]] += pInvMass;
        }
    }

    int fixedAfterRecalc = 0;
    for (size_t i = 0; i < lowRes_invMasses.size(); i++) {
        if (lowRes_invMasses[i] == 0.0f) fixedAfterRecalc++;
    }
    std::cout << "[DEBUG] After mass recalc (before isolated fix): fixed=" << fixedAfterRecalc << std::endl;

    // 孤立した頂点を処理
    int isolatedVertexCount = 0;
    for (size_t i = 0; i < lowRes_invMasses.size(); i++) {
        if (lowRes_invMasses[i] == 0.0f && savedInvMasses[i] != 0.0f) {
            lowRes_invMasses[i] = savedInvMasses[i];
            isolatedVertexCount++;
        }
    }
    std::cout << "[DEBUG] Isolated vertices restored: " << isolatedVertexCount << std::endl;

    int fixedAfterIsolated = 0;
    for (size_t i = 0; i < lowRes_invMasses.size(); i++) {
        if (lowRes_invMasses[i] == 0.0f) fixedAfterIsolated++;
    }
    std::cout << "[DEBUG] After isolated fix: fixed=" << fixedAfterIsolated << std::endl;

    // 固定頂点の質量を0に維持
    for (int pinnedId : lowRes_pinnedVertices) {
        if (pinnedId >= 0 && static_cast<size_t>(pinnedId) < lowRes_invMasses.size()) {
            lowRes_invMasses[pinnedId] = 0.0f;
        }
    }

    // ★★★ 追加: ハンドルグループの頂点も固定を維持 ★★★
    int handleGroupFixedCount = 0;
    for (const auto& group : handleGroups) {
        for (int vertexId : group.vertices) {
            if (vertexId >= 0 && static_cast<size_t>(vertexId) < lowRes_invMasses.size()) {
                lowRes_invMasses[vertexId] = 0.0f;
                handleGroupFixedCount++;
            }
        }
    }
    std::cout << "[DEBUG] HandleGroup vertices fixed: " << handleGroupFixedCount << std::endl;

    int fixedAfterPinned = 0;
    for (size_t i = 0; i < lowRes_invMasses.size(); i++) {
        if (lowRes_invMasses[i] == 0.0f) fixedAfterPinned++;
    }
    std::cout << "[DEBUG] After pinned + handleGroups: fixed=" << fixedAfterPinned << std::endl;

    // 初期制約を復元
    restoreLowResInitialConstraints();

    int fixedAfterRestore = 0;
    for (size_t i = 0; i < lowRes_invMasses.size(); i++) {
        if (lowRes_invMasses[i] == 0.0f) fixedAfterRestore++;
    }
    std::cout << "[DEBUG] After restoreLowResInitialConstraints: fixed=" << fixedAfterRestore << std::endl;

    // エッジ有効性を更新
    updateLowResEdgeValidity();

    // ========== 5. グラブ状態の復元 ==========
    if (grabWasActive) {
        lowRes_grabId = savedGrabId;
        lowRes_grabOffset = savedGrabOffset;
        lowRes_activeParticles = savedActiveParticles;
        lowRes_oldInvMasses = savedOldInvMasses;

        lowRes_positions[lowRes_grabId * 3] = savedGrabPosition.x;
        lowRes_positions[lowRes_grabId * 3 + 1] = savedGrabPosition.y;
        lowRes_positions[lowRes_grabId * 3 + 2] = savedGrabPosition.z;

        lowRes_invMasses[lowRes_grabId] = 0.0f;

        for (int id : lowRes_activeParticles) {
            if (id != lowRes_grabId && static_cast<size_t>(id) < lowRes_invMasses.size()) {
                if (static_cast<size_t>(id) < lowRes_oldInvMasses.size()) {
                    lowRes_invMasses[id] = lowRes_oldInvMasses[id];
                }
            }
        }

        std::cout << "[DEBUG] Restored grab state after invalidation" << std::endl;
    }

    int fixedFinal = 0;
    for (size_t i = 0; i < lowRes_invMasses.size(); i++) {
        if (lowRes_invMasses[i] == 0.0f) fixedFinal++;
    }
    std::cout << "[DEBUG] FINAL in invalidateLowResTetrahedra: fixed=" << fixedFinal << std::endl;


    // スムースメッシュの再生成
    if (smoothDisplayMode) {
        std::cout << "  Regenerating smooth surface..." << std::endl;
        generateSmoothSurface();
        applySmoothingToSurface();
        updateSmoothBuffers();
        std::cout << "  Smooth surface regenerated" << std::endl;
    }

    // ★ これだけ追加
    invalidateSurfaceCache();

     // ★★★ XPBD隣接キャッシュを無効化（カット後に再構築される）★★★
    lowResNeighborsCacheBuilt_ = false;

    std::cout << "[DEBUG] invalidateLowResTetrahedra completed" << std::endl;
}
*/


//==============================================================================
// invalidateLowResTetrahedra 完全デバッグ版
// - initialInvMasses vs recalculated の比較
// - タイミング測定
//
// 【置き換え場所】SoftBodyGPUDuo.cpp の 991行目〜
//==============================================================================

//==============================================================================
// invalidateLowResTetrahedra 完全計測版
//
// 変更点:
// - restoreLowResInitialConstraints()の時間を計測
// - MassRecalc部分も計測
//
// 【置き換え場所】SoftBodyGPUDuo.cpp の invalidateLowResTetrahedra 関数全体
//==============================================================================

void SoftBodyGPUDuo::invalidateLowResTetrahedra(const std::vector<int>& tetIndices) {
    auto totalStart = std::chrono::high_resolution_clock::now();

    // タイミング変数
    double timeGrabSave = 0, timeInvalidate = 0, timeMassSet = 0;
    double timeRestoreConstraints = 0, timeEdgeValidity = 0, timeGrabRestore = 0;
    double timeGenerateSmooth = 0, timeApplySmooth = 0, timeUpdateBuffers = 0;
    double timeOther = 0;

    // ========== 1. グラブ状態の保存 ==========
    auto t0 = std::chrono::high_resolution_clock::now();

    bool grabWasActive = (lowRes_grabId >= 0);
    int savedGrabId = lowRes_grabId;
    glm::vec3 savedGrabPosition;
    glm::vec3 savedGrabOffset = lowRes_grabOffset;
    std::vector<int> savedActiveParticles = lowRes_activeParticles;
    std::vector<float> savedOldInvMasses = lowRes_oldInvMasses;

    if (grabWasActive) {
        savedGrabPosition = glm::vec3(
            lowRes_positions[lowRes_grabId * 3],
            lowRes_positions[lowRes_grabId * 3 + 1],
            lowRes_positions[lowRes_grabId * 3 + 2]
        );
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    timeGrabSave = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ========== 2. 四面体を無効化 ==========
    auto t2 = std::chrono::high_resolution_clock::now();

    int invalidatedCount = 0;
    for (int idx : tetIndices) {
        if (idx >= 0 && idx < static_cast<int>(lowRes_tetValid.size())) {
            if (lowRes_tetValid[idx]) {
                lowRes_tetValid[idx] = false;
                invalidatedCount++;
            }
        }
    }
    std::cout << "  Invalidated " << invalidatedCount << " tetrahedra" << std::endl;

    auto t3 = std::chrono::high_resolution_clock::now();
    timeInvalidate = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // ========== 3. 質量の設定（初期値を使用）==========
    auto t4 = std::chrono::high_resolution_clock::now();

    if (!originalInvMasses.empty()) {
        lowRes_invMasses = originalInvMasses;
    }

    // 固定頂点の質量を0に維持
    for (int pinnedId : lowRes_pinnedVertices) {
        if (pinnedId >= 0 && static_cast<size_t>(pinnedId) < lowRes_invMasses.size()) {
            lowRes_invMasses[pinnedId] = 0.0f;
        }
    }

    // ハンドルグループの頂点も固定を維持
    for (const auto& group : handleGroups) {
        for (int vertexId : group.vertices) {
            if (vertexId >= 0 && static_cast<size_t>(vertexId) < lowRes_invMasses.size()) {
                lowRes_invMasses[vertexId] = 0.0f;
            }
        }
    }

    auto t5 = std::chrono::high_resolution_clock::now();
    timeMassSet = std::chrono::duration<double, std::milli>(t5 - t4).count();

    // ========== 4. 初期制約を復元 ==========
    auto t5_1 = std::chrono::high_resolution_clock::now();

    restoreLowResInitialConstraints();

    auto t5_2 = std::chrono::high_resolution_clock::now();
    timeRestoreConstraints = std::chrono::duration<double, std::milli>(t5_2 - t5_1).count();

    // ========== 5. エッジ有効性を更新 ==========
    auto t6 = std::chrono::high_resolution_clock::now();

    updateLowResEdgeValidity();

    auto t7 = std::chrono::high_resolution_clock::now();
    timeEdgeValidity = std::chrono::duration<double, std::milli>(t7 - t6).count();

    // ========== 6. グラブ状態の復元 ==========
    auto t8 = std::chrono::high_resolution_clock::now();

    if (grabWasActive) {
        lowRes_grabId = savedGrabId;
        lowRes_grabOffset = savedGrabOffset;
        lowRes_activeParticles = savedActiveParticles;
        lowRes_oldInvMasses = savedOldInvMasses;

        lowRes_positions[lowRes_grabId * 3] = savedGrabPosition.x;
        lowRes_positions[lowRes_grabId * 3 + 1] = savedGrabPosition.y;
        lowRes_positions[lowRes_grabId * 3 + 2] = savedGrabPosition.z;

        lowRes_invMasses[lowRes_grabId] = 0.0f;

        for (int id : lowRes_activeParticles) {
            if (id != lowRes_grabId && static_cast<size_t>(id) < lowRes_invMasses.size()) {
                if (static_cast<size_t>(id) < lowRes_oldInvMasses.size()) {
                    lowRes_invMasses[id] = lowRes_oldInvMasses[id];
                }
            }
        }
    }

    int fixedFinal = 0;
    for (size_t i = 0; i < lowRes_invMasses.size(); i++) {
        if (lowRes_invMasses[i] == 0.0f) fixedFinal++;
    }
    std::cout << "[DEBUG] FINAL in invalidateLowResTetrahedra: fixed=" << fixedFinal << std::endl;

    auto t9 = std::chrono::high_resolution_clock::now();
    timeGrabRestore = std::chrono::duration<double, std::milli>(t9 - t8).count();

    // ========== 7. スムースメッシュの再生成 ==========
    if (smoothDisplayMode) {
        std::cout << "  Regenerating smooth surface..." << std::endl;

        auto tSmooth1 = std::chrono::high_resolution_clock::now();
        generateSmoothSurface();
        auto tSmooth2 = std::chrono::high_resolution_clock::now();
        timeGenerateSmooth = std::chrono::duration<double, std::milli>(tSmooth2 - tSmooth1).count();

        auto tSmooth3 = std::chrono::high_resolution_clock::now();
        applySmoothingToSurface();
        auto tSmooth4 = std::chrono::high_resolution_clock::now();
        timeApplySmooth = std::chrono::duration<double, std::milli>(tSmooth4 - tSmooth3).count();

        auto tSmooth5 = std::chrono::high_resolution_clock::now();
        updateSmoothBuffers();
        auto tSmooth6 = std::chrono::high_resolution_clock::now();
        timeUpdateBuffers = std::chrono::duration<double, std::milli>(tSmooth6 - tSmooth5).count();

        std::cout << "  Smooth surface regenerated" << std::endl;
    }

    // ========== 8. キャッシュ無効化 ==========
    auto t10 = std::chrono::high_resolution_clock::now();

    invalidateSurfaceCache();
    lowResNeighborsCacheBuilt_ = false;

    auto t11 = std::chrono::high_resolution_clock::now();
    timeOther = std::chrono::duration<double, std::milli>(t11 - t10).count();

    // ========== タイミング出力 ==========
    auto totalEnd = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    std::cout << "\n  [InvalidateLowRes TIMING BREAKDOWN]" << std::endl;
    std::cout << "    GrabSave:           " << timeGrabSave << " ms" << std::endl;
    std::cout << "    Invalidate:         " << timeInvalidate << " ms" << std::endl;
    std::cout << "    MassSet:            " << timeMassSet << " ms" << std::endl;
    std::cout << "    RestoreConstraints: " << timeRestoreConstraints << " ms" << std::endl;
    std::cout << "    EdgeValidity:       " << timeEdgeValidity << " ms" << std::endl;
    std::cout << "    GrabRestore:        " << timeGrabRestore << " ms" << std::endl;
    if (smoothDisplayMode) {
        std::cout << "    GenerateSmooth:     " << timeGenerateSmooth << " ms" << std::endl;
        std::cout << "    ApplySmooth:        " << timeApplySmooth << " ms" << std::endl;
        std::cout << "    UpdateBuffers:      " << timeUpdateBuffers << " ms" << std::endl;
    }
    std::cout << "    Other:              " << timeOther << " ms" << std::endl;
    std::cout << "    ----------------------" << std::endl;
    std::cout << "    TOTAL:              " << totalTime << " ms" << std::endl;

    std::cout << "[DEBUG] invalidateLowResTetrahedra completed" << std::endl;
}


void SoftBodyGPUDuo::updateLowResEdgeValidity() {
    size_t numEdges = lowRes_edgeIds.size() / 2;
    lowRes_edgeValid.resize(numEdges);
    std::fill(lowRes_edgeValid.begin(), lowRes_edgeValid.end(), false);

    // Step 1: 有効な四面体から頂点ペアのセットを構築 O(T × 6)
    std::set<std::pair<int, int>> validEdgePairs;

    for (size_t t = 0; t < numLowTets; t++) {
        if (!lowRes_tetValid[t]) continue;

        int v0 = lowRes_tetIds[t * 4 + 0];
        int v1 = lowRes_tetIds[t * 4 + 1];
        int v2 = lowRes_tetIds[t * 4 + 2];
        int v3 = lowRes_tetIds[t * 4 + 3];

        validEdgePairs.insert({std::min(v0, v1), std::max(v0, v1)});
        validEdgePairs.insert({std::min(v0, v2), std::max(v0, v2)});
        validEdgePairs.insert({std::min(v0, v3), std::max(v0, v3)});
        validEdgePairs.insert({std::min(v1, v2), std::max(v1, v2)});
        validEdgePairs.insert({std::min(v1, v3), std::max(v1, v3)});
        validEdgePairs.insert({std::min(v2, v3), std::max(v2, v3)});
    }

    // Step 2: 各エッジの有効性をO(log N)でチェック
    for (size_t i = 0; i < numEdges; i++) {
        int id0 = lowRes_edgeIds[2 * i];
        int id1 = lowRes_edgeIds[2 * i + 1];
        lowRes_edgeValid[i] = (validEdgePairs.count({std::min(id0, id1), std::max(id0, id1)}) > 0);
    }
}



void SoftBodyGPUDuo::enableSmoothDisplay(bool enable) {
    smoothDisplayMode = enable;

    if (enable) {
        // æ¯Žå›žå†ç”Ÿæˆï¼ˆç„¡åŠ¹ãªå››é¢ä½“ãŒã‚ã‚‹å ´åˆã«å¯¾å¿œï¼‰
        smoothedVertices = highRes_positions;
        generateSmoothSurface();  // æœ‰åŠ¹ãªå››é¢ä½“ã‹ã‚‰è¡¨é¢ã‚’ç”Ÿæˆ

        if (!smoothSurfaceTriIds.empty()) {
            applySmoothingToSurface();
            setupSmoothBuffers();
        } else {
            std::cout << "Warning: Cannot enable smooth display - no valid surface" << std::endl;
            smoothDisplayMode = false;
        }
    }
}

SoftBodyGPUDuo::BoundingBox SoftBodyGPUDuo::calculateBoundingBox(const std::vector<float>& vertices,
                                                     const std::set<int>& surfaceVertices) {
    BoundingBox bbox;
    bbox.min = glm::vec3(FLT_MAX);
    bbox.max = glm::vec3(-FLT_MAX);

    for (int vid : surfaceVertices) {
        glm::vec3 v(vertices[vid * 3], vertices[vid * 3 + 1], vertices[vid * 3 + 2]);
        bbox.min = glm::min(bbox.min, v);
        bbox.max = glm::max(bbox.max, v);
    }

    bbox.center = (bbox.min + bbox.max) * 0.5f;
    bbox.size = bbox.max - bbox.min;

    return bbox;
}

void SoftBodyGPUDuo::adjustMeshSize(const BoundingBox& originalBBox, const BoundingBox& smoothedBBox,
                              const std::set<int>& surfaceVertices) {
    glm::vec3 scaleFactors;
    scaleFactors.x = (smoothedBBox.size.x > 0.001f) ? originalBBox.size.x / smoothedBBox.size.x : 1.0f;
    scaleFactors.y = (smoothedBBox.size.y > 0.001f) ? originalBBox.size.y / smoothedBBox.size.y : 1.0f;
    scaleFactors.z = (smoothedBBox.size.z > 0.001f) ? originalBBox.size.z / smoothedBBox.size.z : 1.0f;

    if (scalingMethod == 2) {
        // éžä¸€æ§˜ã‚¹ã‚±ãƒ¼ãƒªãƒ³ã‚°
        for (int vid : surfaceVertices) {
            glm::vec3 vertex(smoothedVertices[vid * 3],
                             smoothedVertices[vid * 3 + 1],
                             smoothedVertices[vid * 3 + 2]);

            glm::vec3 relPos = vertex - smoothedBBox.center;
            relPos.x *= scaleFactors.x;
            relPos.y *= scaleFactors.y;
            relPos.z *= scaleFactors.z;

            vertex = originalBBox.center + relPos;

            smoothedVertices[vid * 3] = vertex.x;
            smoothedVertices[vid * 3 + 1] = vertex.y;
            smoothedVertices[vid * 3 + 2] = vertex.z;
        }
    }
}

void SoftBodyGPUDuo::updateSmoothMesh() {
    if (!smoothDisplayMode) return;
smoothedVertices = highRes_positions;

    // ã‚¹ãƒ ãƒ¼ã‚¸ãƒ³ã‚°ã‚’å†é©ç”¨
    applySmoothingToSurface();

    // OpenGLãƒãƒƒãƒ•ã‚¡ã‚’æ›´æ–°
    updateSmoothBuffers();
}

// SoftBodyGPUDuo.cpp の drawSmoothMesh (1228-1240行)
void SoftBodyGPUDuo::drawSmoothMesh(ShaderProgram& shader) {
    shader.use();

    if (smoothDisplayMode && smoothVAO != 0) {
        glBindVertexArray(smoothVAO);
        glDrawElements(GL_TRIANGLES, smoothSurfaceTriIds.size(), GL_UNSIGNED_INT, 0);
    } else {
        // 修正: lowResVAO → highResVAO
        glBindVertexArray(highResVAO);
        glDrawElements(GL_TRIANGLES, highResMeshData.tetSurfaceTriIds.size(), GL_UNSIGNED_INT, 0);
    }

    glBindVertexArray(0);
}

void SoftBodyGPUDuo::setupSmoothBuffers() {
    deleteSmoothBuffers();

    glGenVertexArrays(1, &smoothVAO);
    glGenBuffers(1, &smoothVBO);
    glGenBuffers(1, &smoothEBO);
    glGenBuffers(1, &smoothNormalVBO);

    glBindVertexArray(smoothVAO);

    // é ‚ç‚¹ãƒãƒƒãƒ•ã‚¡
    glBindBuffer(GL_ARRAY_BUFFER, smoothVBO);
    glBufferData(GL_ARRAY_BUFFER, smoothedVertices.size() * sizeof(float),
                 smoothedVertices.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // æ³•ç·šãƒãƒƒãƒ•ã‚¡
    glBindBuffer(GL_ARRAY_BUFFER, smoothNormalVBO);
    std::vector<float> normals = computeSmoothNormals();
    glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float),
                 normals.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);

    // ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹ãƒãƒƒãƒ•ã‚¡
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, smoothEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, smoothSurfaceTriIds.size() * sizeof(int),
                 smoothSurfaceTriIds.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void SoftBodyGPUDuo::updateSmoothBuffers() {
    if (smoothVAO == 0) {
        setupSmoothBuffers();
        return;
    }

    // é ‚ç‚¹ãƒãƒƒãƒ•ã‚¡æ›´æ–°
    glBindBuffer(GL_ARRAY_BUFFER, smoothVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0,
                    smoothedVertices.size() * sizeof(float),
                    smoothedVertices.data());

    // æ³•ç·šãƒãƒƒãƒ•ã‚¡æ›´æ–°
    glBindBuffer(GL_ARRAY_BUFFER, smoothNormalVBO);
    std::vector<float> normals = computeSmoothNormals();
    glBufferSubData(GL_ARRAY_BUFFER, 0,
                    normals.size() * sizeof(float),
                    normals.data());

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void SoftBodyGPUDuo::deleteSmoothBuffers() {
    if (smoothVAO != 0) {
        glDeleteVertexArrays(1, &smoothVAO);
        smoothVAO = 0;
    }
    if (smoothVBO != 0) {
        glDeleteBuffers(1, &smoothVBO);
        smoothVBO = 0;
    }
    if (smoothEBO != 0) {
        glDeleteBuffers(1, &smoothEBO);
        smoothEBO = 0;
    }
    if (smoothNormalVBO != 0) {
        glDeleteBuffers(1, &smoothNormalVBO);
        smoothNormalVBO = 0;
    }
}

void SoftBodyGPUDuo::generateSmoothSurface() {
    smoothSurfaceTriIds.clear();
    smoothTriangleTetMap.clear();

    struct Face {
        int v0, v1, v2;
        int origV0, origV1, origV2;
        int tetIndex;
        bool isOriginalSurface;  // å…ƒã®è¡¨é¢ã‹æ–­é¢ã‹ã‚’åŒºåˆ¥

        Face(int a, int b, int c, int tet, bool isOrig = false) : tetIndex(tet), isOriginalSurface(isOrig) {
            origV0 = a;
            origV1 = b;
            origV2 = c;

            int vertices[3] = {a, b, c};
            if (vertices[0] > vertices[1]) std::swap(vertices[0], vertices[1]);
            if (vertices[1] > vertices[2]) std::swap(vertices[1], vertices[2]);
            if (vertices[0] > vertices[1]) std::swap(vertices[0], vertices[1]);

            v0 = vertices[0];
            v1 = vertices[1];
            v2 = vertices[2];
        }

        bool operator<(const Face& other) const {
            if (v0 != other.v0) return v0 < other.v0;
            if (v1 != other.v1) return v1 < other.v1;
            return v2 < other.v2;
        }

        bool operator==(const Face& other) const {
            return v0 == other.v0 && v1 == other.v1 && v2 == other.v2;
        }
    };

    std::vector<Face> allFaces;
    allFaces.reserve(numHighTets * 4);

    // å…ƒã®è¡¨é¢ã‹ã©ã†ã‹ã‚’åˆ¤å®šã™ã‚‹ãŸã‚ã®ã‚»ãƒƒãƒˆ
    std::set<std::tuple<int, int, int>> originalSurfaceFaces;
    for (size_t i = 0; i < highResMeshData.tetSurfaceTriIds.size(); i += 3) {
        int v0 = highResMeshData.tetSurfaceTriIds[i];
        int v1 = highResMeshData.tetSurfaceTriIds[i + 1];
        int v2 = highResMeshData.tetSurfaceTriIds[i + 2];

        // ã‚½ãƒ¼ãƒˆã—ã¦æ­£è¦åŒ–
        int vertices[3] = {v0, v1, v2};
        if (vertices[0] > vertices[1]) std::swap(vertices[0], vertices[1]);
        if (vertices[1] > vertices[2]) std::swap(vertices[1], vertices[2]);
        if (vertices[0] > vertices[1]) std::swap(vertices[0], vertices[1]);

        originalSurfaceFaces.insert(std::make_tuple(vertices[0], vertices[1], vertices[2]));
    }

    // æœ‰åŠ¹ãªå››é¢ä½“ã®ã¿ã‹ã‚‰é¢ã‚’åŽé›†
    for (size_t i = 0; i < numHighTets; i++) {
        // ç„¡åŠ¹ãªå››é¢ä½“ã¯ã‚¹ã‚­ãƒƒãƒ—
        if (!highResTetValid.empty() && !highResTetValid[i]) {
            continue;
        }

        int v0 = highResTetIds[i * 4];
        int v1 = highResTetIds[i * 4 + 1];
        int v2 = highResTetIds[i * 4 + 2];
        int v3 = highResTetIds[i * 4 + 3];

        int tetVerts[4] = {v0, v1, v2, v3};

        // â˜…ä¿®æ­£ï¼šå››é¢ä½“å®šç¾©ã«åŸºã¥ãæ­£ã—ã„é¢é †åºã§ç”Ÿæˆ
        for (int faceIdx = 0; faceIdx < 4; faceIdx++) {
            int fv0 = tetVerts[TET_FACE_INDICES[faceIdx][0]];
            int fv1 = tetVerts[TET_FACE_INDICES[faceIdx][1]];
            int fv2 = tetVerts[TET_FACE_INDICES[faceIdx][2]];

            allFaces.emplace_back(fv0, fv1, fv2, i);
        }
    }

    std::sort(allFaces.begin(), allFaces.end());

    smoothSurfaceTriIds.reserve(allFaces.size() / 10);
    smoothTriangleTetMap.reserve(allFaces.size() / 10);

    // è¡¨é¢ä¸‰è§’å½¢ï¼ˆå…±æœ‰ã•ã‚Œã¦ã„ãªã„é¢ï¼‰ã‚’æŠ½å‡º
    for (size_t i = 0; i < allFaces.size(); ) {
        if (i + 1 < allFaces.size() && allFaces[i] == allFaces[i + 1]) {
            // å…±æœ‰é¢ã¯ã‚¹ã‚­ãƒƒãƒ—
            i += 2;
        } else {
            const Face& face = allFaces[i];

            // ä¸‰è§’å½¢æƒ…å ±ã‚’ä¿å­˜
            TriangleTetInfo triInfo;
            triInfo.tetIndex = face.tetIndex;

            // SimpleTetMeshæ–¹å¼: å››é¢ä½“å®šç¾©ã«ã‚ˆã‚Šå‘ãã¯ä¿è¨¼æ¸ˆã¿
            // å…ƒã®è¡¨é¢ã‚‚ã‚«ãƒƒãƒˆæ–­é¢ã‚‚ã€åŒã˜å››é¢ä½“ã‹ã‚‰ç”Ÿæˆã•ã‚Œã‚‹ãŸã‚å‘ãåˆ¤å®šä¸è¦
            smoothSurfaceTriIds.push_back(face.origV0);
            smoothSurfaceTriIds.push_back(face.origV1);
            smoothSurfaceTriIds.push_back(face.origV2);
            triInfo.v0 = face.origV0;
            triInfo.v1 = face.origV1;
            triInfo.v2 = face.origV2;

            smoothTriangleTetMap.push_back(triInfo);
            i++;
        }
    }

    std::cout << "Generated smooth surface with " << smoothSurfaceTriIds.size() / 3
              << " triangles from " << numHighTets << " valid tetrahedra" << std::endl;

    // ã‚¹ãƒ ãƒ¼ã‚ºè¡¨é¢ãŒç©ºã®å ´åˆã®è­¦å‘Š
    if (smoothSurfaceTriIds.empty()) {
        std::cout << "Warning: No surface triangles generated from valid tetrahedra!" << std::endl;
        // ãƒ•ã‚©ãƒ¼ãƒ«ãƒãƒƒã‚¯ï¼šé«˜è§£åƒåº¦ãƒ¡ãƒƒã‚·ãƒ¥ã®æœ‰åŠ¹ãªä¸‰è§’å½¢ã‚’ä½¿ç”¨
        smoothSurfaceTriIds = highResValidTriangles;
    }
    neighborsComputed = false;
}

void SoftBodyGPUDuo::setSmoothingParameters(int iterations, float factor, bool adjustSize, int method) {
    // å®‰å…¨ãªç¯„å›²ã«åˆ¶é™
    smoothingIterations = std::min(std::max(1, iterations), 10);
    smoothingFactor = std::min(std::max(0.1f, factor), 0.8f);  // æœ€å¤§0.8ã«åˆ¶é™
    enableSizeAdjustment = adjustSize;
    scalingMethod = method;

    // è­¦å‘Šã‚’è¡¨ç¤º
    if (factor > 0.7f && iterations > 3) {
        std::cout << "Warning: High smoothing parameters may cause surface inversion" << std::endl;
        std::cout << "  Consider reducing factor or iterations for stable results" << std::endl;
    }

    if (smoothDisplayMode) {
        smoothedVertices = highRes_positions;
        applySmoothingToSurface();
        updateSmoothBuffers();
    }
}


void SoftBodyGPUDuo::clearSmoothingData() {
    smoothDisplayMode = false;
    smoothedVertices.clear();
    smoothSurfaceTriIds.clear();
    smoothTriangleTetMap.clear();
    deleteSmoothBuffers();
}

std::vector<int> SoftBodyGPUDuo::getVertexNeighbors(int vertexId) {
    std::set<int> neighbors;
    for (size_t i = 0; i < smoothSurfaceTriIds.size(); i += 3) {
        int v0 = smoothSurfaceTriIds[i];
        int v1 = smoothSurfaceTriIds[i + 1];
        int v2 = smoothSurfaceTriIds[i + 2];

        if (v0 == vertexId) {
            neighbors.insert(v1);
            neighbors.insert(v2);
        } else if (v1 == vertexId) {
            neighbors.insert(v0);
            neighbors.insert(v2);
        } else if (v2 == vertexId) {
            neighbors.insert(v0);
            neighbors.insert(v1);
        }
    }

    return std::vector<int>(neighbors.begin(), neighbors.end());
}

glm::vec3 SoftBodyGPUDuo::calculateMeshCenter() {
    glm::vec3 center(0.0f);
    for (size_t i = 0; i < numHighResVerts; i++) {
        center.x += smoothedVertices[i * 3];
        center.y += smoothedVertices[i * 3 + 1];
        center.z += smoothedVertices[i * 3 + 2];
    }
    return center / float(numHighResVerts);
}

void SoftBodyGPUDuo::computeVertexNeighborsCache() {
    if (neighborsComputed) return;

    vertexNeighborsCache.clear();
    vertexNeighborsCache.resize(numHighResVerts);

    for (size_t i = 0; i < smoothSurfaceTriIds.size(); i += 3) {
        int v0 = smoothSurfaceTriIds[i];
        int v1 = smoothSurfaceTriIds[i + 1];
        int v2 = smoothSurfaceTriIds[i + 2];

        vertexNeighborsCache[v0].push_back(v1);
        vertexNeighborsCache[v0].push_back(v2);
        vertexNeighborsCache[v1].push_back(v0);
        vertexNeighborsCache[v1].push_back(v2);
        vertexNeighborsCache[v2].push_back(v0);
        vertexNeighborsCache[v2].push_back(v1);
    }

    // é‡è¤‡ã‚’å‰Šé™¤
    for (auto& neighbors : vertexNeighborsCache) {
        if (neighbors.empty()) continue;
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(
            std::unique(neighbors.begin(), neighbors.end()),
            neighbors.end()
            );
    }

    neighborsComputed = true;
    std::cout << "Vertex neighbors cache computed for valid surface vertices" << std::endl;
}

void SoftBodyGPUDuo::clearNeighborsCache() {
    vertexNeighborsCache.clear();
    neighborsComputed = false;
}

std::vector<float> SoftBodyGPUDuo::computeSmoothNormals() {
    if (!neighborsComputed) {
        computeVertexNeighborsCache();
    }

    std::vector<glm::vec3> normals(numHighResVerts, glm::vec3(0.0f));

    // â˜… ä¿®æ­£ï¼šsmoothSurfaceTriIdsï¼ˆã‚«ãƒƒãƒˆæ–­é¢å«ã‚€ï¼‰ã‚’ä½¿ç”¨
    for (size_t i = 0; i < smoothSurfaceTriIds.size(); i += 3) {
        int id0 = smoothSurfaceTriIds[i];
        int id1 = smoothSurfaceTriIds[i + 1];
        int id2 = smoothSurfaceTriIds[i + 2];

        glm::vec3 v0(smoothedVertices[id0 * 3], smoothedVertices[id0 * 3 + 1], smoothedVertices[id0 * 3 + 2]);
        glm::vec3 v1(smoothedVertices[id1 * 3], smoothedVertices[id1 * 3 + 1], smoothedVertices[id1 * 3 + 2]);
        glm::vec3 v2(smoothedVertices[id2 * 3], smoothedVertices[id2 * 3 + 1], smoothedVertices[id2 * 3 + 2]);

        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;
        glm::vec3 normal = glm::cross(edge1, edge2);

        // æ­£è¦åŒ–ä¸è¦ï¼ˆé¢ç©åŠ é‡ï¼‰
        normals[id0] += normal;
        normals[id1] += normal;
        normals[id2] += normal;
    }

    std::vector<float> normalBuffer;
    normalBuffer.reserve(numHighResVerts * 3);

    for (size_t i = 0; i < numHighResVerts; i++) {
        if (glm::length(normals[i]) > 0.0f) {
            normals[i] = glm::normalize(normals[i]);
        }
        normalBuffer.push_back(normals[i].x);
        normalBuffer.push_back(normals[i].y);
        normalBuffer.push_back(normals[i].z);
    }

    return normalBuffer;
}

int SoftBodyGPUDuo::getInvalidatedLowResTetCount() const {
    return std::count(lowRes_tetValid.begin(), lowRes_tetValid.end(), false);
}

int SoftBodyGPUDuo::getInvalidatedHighResTetCount() const {
    return std::count(highResTetValid.begin(), highResTetValid.end(), false);
}

void SoftBodyGPUDuo::validateHighResTetrahedra(const std::vector<int>& tetIndices) {
    for (int idx : tetIndices) {
        if (idx >= 0 && idx < highResTetValid.size()) {
            highResTetValid[idx] = true;
        }
    }

    initPhysicsLowRes();

    if (smoothDisplayMode) {
        generateSmoothSurface();
        applySmoothingToSurface();
        updateSmoothBuffers();
    }
}

void SoftBodyGPUDuo::validateAllHighResTetrahedra() {
    std::fill(highResTetValid.begin(), highResTetValid.end(), true);

    initPhysicsLowRes();

    if (smoothDisplayMode) {
        generateSmoothSurface();
        applySmoothingToSurface();
        updateSmoothBuffers();
    }
    invalidateSurfaceCache();  // ★ 追加
}


void SoftBodyGPUDuo::invalidateLowResTetsWithoutHighRes() {
    std::cout << "\n=== Invalidating Low-Res Tets Without High-Res Coverage ===" << std::endl;

    if (!tetMappingComputed) {
        computeTetToTetMappingLowToHigh();
    }

    int invalidatedCount = 0;
    std::vector<bool> lowTetHasCoverage(numLowTets, false);

    // ★修正: 有効な高解像度四面体のみをチェック
    for (size_t highTetIdx = 0; highTetIdx < numHighTets; highTetIdx++) {
        // ★この行を追加: 無効な高解像度四面体はスキップ
        if (!highResTetValid.empty() && !highResTetValid[highTetIdx]) {
            continue;
        }

        std::set<int> correspondingLowTets = getLowResTetsFromHighResTet(highTetIdx);
        for (int lowTetIdx : correspondingLowTets) {
            if (lowTetIdx >= 0 && lowTetIdx < (int)numLowTets) {
                lowTetHasCoverage[lowTetIdx] = true;
            }
        }
    }

    for (size_t lowTetIdx = 0; lowTetIdx < numLowTets; lowTetIdx++) {
        // 既に無効化されているものはスキップ
        if (!lowRes_tetValid[lowTetIdx]) {
            continue;
        }

        if (!lowTetHasCoverage[lowTetIdx]) {
            lowRes_tetValid[lowTetIdx] = false;
            invalidatedCount++;
        }
    }

    std::cout << "Invalidated " << invalidatedCount << " / " << numLowTets
              << " Low-res tets without High-res coverage" << std::endl;

    int validTets = std::count(lowRes_tetValid.begin(), lowRes_tetValid.end(), true);
    std::cout << "Remaining valid Low-res tets: " << validTets << std::endl;
    std::cout << "Coverage ratio: "
              << (100.0f * validTets / numLowTets) << "%" << std::endl;

    // 質量と制約を再計算
    //initPhysicsLowRes();

    setupLowResTetMesh();
    updateLowResMesh();

    invalidateSurfaceCache();
}


void SoftBodyGPUDuo::updateHighResEdgeValidity() {
    highResEdgeValid.resize(highResEdgeIds.size() / 2, false);

    for (size_t i = 0; i < highResEdgeIds.size() / 2; i++) {
        int id0 = highResEdgeIds[2 * i];
        int id1 = highResEdgeIds[2 * i + 1];

        // ã“ã®ã‚¨ãƒƒã‚¸ãŒæœ‰åŠ¹ãªå››é¢ä½“ã«å±žã™ã‚‹ã‹ãƒã‚§ãƒƒã‚¯
        for (size_t t = 0; t < numHighTets; t++) {
            if (!highResTetValid[t]) continue;

            bool hasId0 = false, hasId1 = false;
            for (int j = 0; j < 4; j++) {
                int vid = highResTetIds[t * 4 + j];
                if (vid == id0) hasId0 = true;
                if (vid == id1) hasId1 = true;
            }

            if (hasId0 && hasId1) {
                highResEdgeValid[i] = true;
                break;
            }
        }
    }
}

void SoftBodyGPUDuo::invalidateHighResTetrahedra(const std::vector<int>& tetIndices) {
    // ç„¡åŠ¹åŒ–å‡¦ç†
    for (int tetIdx : tetIndices) {
        if (tetIdx >= 0 && tetIdx < static_cast<int>(highResTetValid.size())) {
            if (highResTetValid[tetIdx]) {
                highResTetValid[tetIdx] = false;
                highResInvalidatedCount++;
            }
        }
    }

    std::cout << "Total invalidated: " << highResInvalidatedCount
              << " / " << highResTetValid.size() << std::endl;

    updateHighResValidSurface();

    //updateHighResTetMesh();

}

void SoftBodyGPUDuo::applySmoothedDamping() {
    if (!lowRes_useBoundaryDamping) return;
    std::set<int> boundaryVertices;
    if (getInvalidatedLowResTetCount() > 0) {
        for (size_t i = 0; i < numLowResParticles; i++) {
            bool touchesValid = false;
            bool touchesInvalid = false;

            for (size_t t = 0; t < numLowTets; t++) {
                for (int j = 0; j < 4; j++) {
                    if (lowRes_tetIds[t * 4 + j] == i) {
                        if (lowRes_tetValid[t]) touchesValid = true;
                        else touchesInvalid = true;
                        break;
                    }
                }
            }

            if (touchesValid && touchesInvalid) {
                boundaryVertices.insert(i);
            }
        }
    }

    for (size_t i = 0; i < numLowResParticles; i++) {
        if (lowRes_invMasses[i] == 0.0f) continue;

        // é€Ÿåº¦ã«é€£ç¶šçš„ãªæ¸›è¡°ã‚’é©ç”¨
        float dampFactor = lowRes_motionDampingFactor;

        // å¢ƒç•Œé ‚ç‚¹ã«ã¯ã‚ˆã‚Šå¼·ã„æ¸›è¡°
        if (boundaryVertices.count(i) > 0) {
            dampFactor = lowRes_boundaryDampingFactor;
        }

        // é€Ÿåº¦ã‚’æ¸›è¡°ï¼ˆæ€¥æ¿€ã«æ­¢ã‚ã‚‹ã®ã§ã¯ãªãå¾ã€…ã«æ¸›é€Ÿï¼‰
        lowRes_velocities[i * 3] *= dampFactor;
        lowRes_velocities[i * 3 + 1] *= dampFactor;
        lowRes_velocities[i * 3 + 2] *= dampFactor;

        // éžå¸¸ã«å°ã•ã„é€Ÿåº¦ã«ãªã£ãŸã‚‰0ã«ã™ã‚‹ï¼ˆã‚ªãƒ—ã‚·ãƒ§ãƒ³ï¼‰
        float vx = lowRes_velocities[i * 3];
        float vy = lowRes_velocities[i * 3 + 1];
        float vz = lowRes_velocities[i * 3 + 2];
        float speed = std::sqrt(vx*vx + vy*vy + vz*vz);

        if (speed < 0.0001f) {  // éžå¸¸ã«å°ã•ã„é–¾å€¤
            lowRes_velocities[i * 3] = 0.0f;
            lowRes_velocities[i * 3 + 1] = 0.0f;
            lowRes_velocities[i * 3 + 2] = 0.0f;
        }
    }
}


#include <unordered_map>
void SoftBodyGPUDuo::computeTetToTetMappingLowToHigh() {
    std::cout << "\n=== Computing Tet-to-Tet Mapping for Cut Operations ===" << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();

    highToLowTetMapping.clear();
    lowToHighTetMapping.clear();

    // 同じメッシュかどうかをチェック
    bool sameMesh = (numHighTets == numLowTets);
    std::cout << "  Same mesh detected: " << (sameMesh ? "YES" : "NO") << std::endl;
    std::cout << "  High-res tets: " << numHighTets << ", Low-res tets: " << numLowTets << std::endl;

    if (sameMesh) {
        // 同じメッシュの場合：直接IDマッピング
        std::cout << "  Using direct ID mapping for same mesh" << std::endl;
        for (size_t tetIdx = 0; tetIdx < numHighTets; tetIdx++) {
            highToLowTetMapping[tetIdx].insert(tetIdx);
            lowToHighTetMapping[tetIdx].insert(tetIdx);
        }
    } else {
        // 異なるメッシュの場合：空間ハッシュを使用
        std::cout << "  Using spatial hash for different meshes (OPTIMIZED)" << std::endl;

        float distanceThreshold = 0.05f;

        // ---------------------------------------------------------------------
        // Step 1: 低解像度tetの重心を計算
        // ---------------------------------------------------------------------
        std::vector<glm::vec3> lowTetCenters(numLowTets);
        glm::vec3 minBound(FLT_MAX), maxBound(-FLT_MAX);

        for (size_t i = 0; i < numLowTets; i++) {
            glm::vec3 center(0.0f);
            for (int j = 0; j < 4; j++) {
                int vid = lowRes_tetIds[i * 4 + j];
                center.x += lowRes_positions[vid * 3];
                center.y += lowRes_positions[vid * 3 + 1];
                center.z += lowRes_positions[vid * 3 + 2];
            }
            center /= 4.0f;
            lowTetCenters[i] = center;

            // バウンディングボックスを更新
            minBound = glm::min(minBound, center);
            maxBound = glm::max(maxBound, center);
        }

        // ---------------------------------------------------------------------
        // Step 2: セルサイズを決定（閾値の2倍程度が適切）
        // ---------------------------------------------------------------------
        float cellSize = distanceThreshold * 2.5f;

        // セルサイズが小さすぎると効率が悪いので、最小値を設定
        glm::vec3 meshSize = maxBound - minBound;
        float avgDim = (meshSize.x + meshSize.y + meshSize.z) / 3.0f;
        float minCellSize = avgDim / 100.0f;  // メッシュサイズの1%を最小値
        cellSize = std::max(cellSize, minCellSize);

        std::cout << "  Cell size: " << cellSize << std::endl;
        std::cout << "  Distance threshold: " << distanceThreshold << std::endl;

        // ---------------------------------------------------------------------
        // Step 3: 空間ハッシュ関数
        // ---------------------------------------------------------------------
        // グリッドのセル座標を計算
        auto getCellCoord = [cellSize, &minBound](const glm::vec3& p) -> glm::ivec3 {
            return glm::ivec3(
                static_cast<int>(std::floor((p.x - minBound.x) / cellSize)),
                static_cast<int>(std::floor((p.y - minBound.y) / cellSize)),
                static_cast<int>(std::floor((p.z - minBound.z) / cellSize))
            );
        };

        // セル座標からハッシュ値を計算（大きな素数を使用）
        auto hashCell = [](const glm::ivec3& cell) -> int64_t {
            // 負の座標にも対応するため、オフセットを追加
            int64_t x = static_cast<int64_t>(cell.x) + 1000000;
            int64_t y = static_cast<int64_t>(cell.y) + 1000000;
            int64_t z = static_cast<int64_t>(cell.z) + 1000000;
            return (x * 73856093LL) ^ (y * 19349663LL) ^ (z * 83492791LL);
        };

        // ---------------------------------------------------------------------
        // Step 4: 低解像度tetをグリッドに登録
        // ---------------------------------------------------------------------
        std::unordered_map<int64_t, std::vector<int>> grid;
        grid.reserve(numLowTets);  // メモリ確保の最適化

        for (size_t i = 0; i < numLowTets; i++) {
            glm::ivec3 cell = getCellCoord(lowTetCenters[i]);
            int64_t hash = hashCell(cell);
            grid[hash].push_back(static_cast<int>(i));
        }

        std::cout << "  Grid cells created: " << grid.size() << std::endl;

        // ---------------------------------------------------------------------
        // Step 5: 高解像度tetごとに近傍セルのみを検索
        // ---------------------------------------------------------------------
        size_t totalComparisons = 0;
        float thresholdSq = distanceThreshold * distanceThreshold;

        for (size_t highTetIdx = 0; highTetIdx < numHighTets; highTetIdx++) {
            // この高解像度四面体の重心を計算
            glm::vec3 highTetCenter(0.0f);
            for (int j = 0; j < 4; j++) {
                int vid = highResTetIds[highTetIdx * 4 + j];
                highTetCenter.x += highRes_positions[vid * 3];
                highTetCenter.y += highRes_positions[vid * 3 + 1];
                highTetCenter.z += highRes_positions[vid * 3 + 2];
            }
            highTetCenter /= 4.0f;

            glm::ivec3 centerCell = getCellCoord(highTetCenter);

            float minDistSq = FLT_MAX;
            int closestLowTet = -1;

            // 近傍27セル（3×3×3）のみを検索
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dz = -1; dz <= 1; dz++) {
                        glm::ivec3 neighborCell = centerCell + glm::ivec3(dx, dy, dz);
                        int64_t hash = hashCell(neighborCell);

                        auto it = grid.find(hash);
                        if (it == grid.end()) continue;

                        for (int lowTetIdx : it->second) {
                            totalComparisons++;

                            glm::vec3 diff = highTetCenter - lowTetCenters[lowTetIdx];
                            float distSq = glm::dot(diff, diff);

                            if (distSq < minDistSq) {
                                minDistSq = distSq;
                                closestLowTet = lowTetIdx;
                            }

                            if (distSq < thresholdSq) {
                                highToLowTetMapping[highTetIdx].insert(lowTetIdx);
                                lowToHighTetMapping[lowTetIdx].insert(highTetIdx);
                            }
                        }
                    }
                }
            }

            // 閾値内に四面体がない場合、最も近い1つだけを登録
            if (highToLowTetMapping[highTetIdx].empty() && closestLowTet >= 0) {
                highToLowTetMapping[highTetIdx].insert(closestLowTet);
                lowToHighTetMapping[closestLowTet].insert(highTetIdx);
            }
        }

        // 比較回数の統計
        size_t bruteForceComparisons = numHighTets * numLowTets;
        float speedup = static_cast<float>(bruteForceComparisons) /
                       static_cast<float>(std::max(totalComparisons, size_t(1)));
        std::cout << "  Total comparisons: " << totalComparisons
                  << " (brute force would be: " << bruteForceComparisons << ")" << std::endl;
        std::cout << "  Speedup factor: " << std::fixed << std::setprecision(1)
                  << speedup << "x" << std::endl;
    }

    // -------------------------------------------------------------------------
    // 統計情報
    // -------------------------------------------------------------------------
    std::cout << "Mapping statistics:" << std::endl;
    std::cout << "  High-res tets with mapping: " << highToLowTetMapping.size()
              << " / " << numHighTets << std::endl;
    std::cout << "  Low-res tets with mapping: " << lowToHighTetMapping.size()
              << " / " << numLowTets << std::endl;

    // 平均マッピング数
    float avgHighToLow = 0, avgLowToHigh = 0;
    int maxHighToLow = 0, maxLowToHigh = 0;

    for (const auto& [tet, mappings] : highToLowTetMapping) {
        avgHighToLow += mappings.size();
        maxHighToLow = std::max(maxHighToLow, (int)mappings.size());
    }
    for (const auto& [tet, mappings] : lowToHighTetMapping) {
        avgLowToHigh += mappings.size();
        maxLowToHigh = std::max(maxLowToHigh, (int)mappings.size());
    }

    if (!highToLowTetMapping.empty()) avgHighToLow /= highToLowTetMapping.size();
    if (!lowToHighTetMapping.empty()) avgLowToHigh /= lowToHighTetMapping.size();

    std::cout << "  Avg low-res tets per high-res tet: " << avgHighToLow
              << " (max: " << maxHighToLow << ")" << std::endl;
    std::cout << "  Avg high-res tets per low-res tet: " << avgLowToHigh
              << " (max: " << maxLowToHigh << ")" << std::endl;

    // デバッグ：最初の5つの詳細
    if (!sameMesh) {
        std::cout << "  Sample mappings (first 5):" << std::endl;
        int count = 0;
        for (const auto& [highTet, lowTets] : highToLowTetMapping) {
            if (count++ >= 5) break;
            std::cout << "    High-res tet " << highTet << " -> Low-res tets: ";
            for (int lowTet : lowTets) {
                std::cout << lowTet << " ";
            }
            std::cout << std::endl;
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "  Mapping computation time: " << duration.count() << " ms" << std::endl;

    tetMappingComputed = true;
}

std::set<int> SoftBodyGPUDuo::getLowResTetsFromHighResTet(int highResTetIdx) {
    if (!tetMappingComputed) {
        computeTetToTetMappingLowToHigh();
    }

    auto it = highToLowTetMapping.find(highResTetIdx);
    if (it != highToLowTetMapping.end()) {
        return it->second;
    }
    return std::set<int>();
}

std::set<int> SoftBodyGPUDuo::getHighResTetsFromLowResTet(int lowResTetIdx) {
    if (!tetMappingComputed) {
        computeTetToTetMappingLowToHigh();
    }

    auto it = lowToHighTetMapping.find(lowResTetIdx);
    if (it != lowToHighTetMapping.end()) {
        return it->second;
    }
    return std::set<int>();
}

void SoftBodyGPUDuo::saveLowResInitialShape() {
    // åˆæœŸä½ç½®ã‚’ä¿å­˜
    lowRes_initialPositions = lowResMeshData.verts;  // å…ƒã®ãƒ¡ãƒƒã‚·ãƒ¥ãƒ‡ãƒ¼ã‚¿ã‹ã‚‰

    // åˆæœŸã®å››é¢ä½“ä½“ç©ã‚’è¨ˆç®—ã—ã¦ä¿å­˜
    lowRes_initialRestVols.resize(numLowTets);
    for (size_t i = 0; i < numLowTets; i++) {
        // å…ƒã®é ‚ç‚¹ä½ç½®ã‚’ä½¿ã£ã¦ä½“ç©ã‚’è¨ˆç®—
        float vol = getTetVolumeFromInitial(i);
        lowRes_initialRestVols[i] = vol;
    }

    // åˆæœŸã®ã‚¨ãƒƒã‚¸é•·ã‚’è¨ˆç®—ã—ã¦ä¿å­˜
    lowRes_initialEdgeLengths.resize(lowRes_edgeIds.size() / 2);
    for (size_t i = 0; i < lowRes_edgeLengths.size(); i++) {
        int id0 = lowRes_edgeIds[2 * i];
        int id1 = lowRes_edgeIds[2 * i + 1];

        // åˆæœŸä½ç½®ã‹ã‚‰ã‚¨ãƒƒã‚¸é•·ã‚’è¨ˆç®—
        float dx = lowRes_initialPositions[id0 * 3] - lowRes_initialPositions[id1 * 3];
        float dy = lowRes_initialPositions[id0 * 3 + 1] - lowRes_initialPositions[id1 * 3 + 1];
        float dz = lowRes_initialPositions[id0 * 3 + 2] - lowRes_initialPositions[id1 * 3 + 2];
        lowRes_initialEdgeLengths[i] = std::sqrt(dx*dx + dy*dy + dz*dz);
    }
}

float SoftBodyGPUDuo::getTetVolumeFromInitial(int nr) {
    int id0 = lowRes_tetIds[4 * nr];
    int id1 = lowRes_tetIds[4 * nr + 1];
    int id2 = lowRes_tetIds[4 * nr + 2];
    int id3 = lowRes_tetIds[4 * nr + 3];

    // åˆæœŸä½ç½®ã‚’ä½¿ç”¨
    glm::vec3 v0(lowRes_initialPositions[id0 * 3], lowRes_initialPositions[id0 * 3 + 1], lowRes_initialPositions[id0 * 3 + 2]);
    glm::vec3 v1(lowRes_initialPositions[id1 * 3], lowRes_initialPositions[id1 * 3 + 1], lowRes_initialPositions[id1 * 3 + 2]);
    glm::vec3 v2(lowRes_initialPositions[id2 * 3], lowRes_initialPositions[id2 * 3 + 1], lowRes_initialPositions[id2 * 3 + 2]);
    glm::vec3 v3(lowRes_initialPositions[id3 * 3], lowRes_initialPositions[id3 * 3 + 1], lowRes_initialPositions[id3 * 3 + 2]);

    glm::vec3 e1 = v1 - v0;
    glm::vec3 e2 = v2 - v0;
    glm::vec3 e3 = v3 - v0;

    return glm::dot(glm::cross(e1, e2), e3) / 6.0f;
}


void SoftBodyGPUDuo::restoreLowResInitialConstraints() {
    // 1. 四面体の静止体積を復元 O(T)
    for (size_t i = 0; i < numLowTets; i++) {
        if (lowRes_tetValid[i] && i < lowRes_initialRestVols.size()) {
            lowRes_restVols[i] = lowRes_initialRestVols[i];
        }
    }

    // 2. 有効なエッジペアのセットを構築 O(T × 6)
    std::set<std::pair<int, int>> validEdgePairs;
    for (size_t t = 0; t < numLowTets; t++) {
        if (!lowRes_tetValid[t]) continue;

        int v0 = lowRes_tetIds[t * 4 + 0];
        int v1 = lowRes_tetIds[t * 4 + 1];
        int v2 = lowRes_tetIds[t * 4 + 2];
        int v3 = lowRes_tetIds[t * 4 + 3];

        validEdgePairs.insert({std::min(v0, v1), std::max(v0, v1)});
        validEdgePairs.insert({std::min(v0, v2), std::max(v0, v2)});
        validEdgePairs.insert({std::min(v0, v3), std::max(v0, v3)});
        validEdgePairs.insert({std::min(v1, v2), std::max(v1, v2)});
        validEdgePairs.insert({std::min(v1, v3), std::max(v1, v3)});
        validEdgePairs.insert({std::min(v2, v3), std::max(v2, v3)});
    }

    // 3. 有効なエッジの初期長さを復元 O(E × log N)
    for (size_t i = 0; i < lowRes_edgeLengths.size(); i++) {
        if (i < lowRes_initialEdgeLengths.size()) {
            int id0 = lowRes_edgeIds[2 * i];
            int id1 = lowRes_edgeIds[2 * i + 1];

            if (validEdgePairs.count({std::min(id0, id1), std::max(id0, id1)}) > 0) {
                lowRes_edgeLengths[i] = lowRes_initialEdgeLengths[i];
            }
        }
    }
}


SoftBodyGPUDuo::~SoftBodyGPUDuo() {
    deleteLowHighBuffers();
    deleteSmoothBuffers();
}

float SoftBodyGPUDuo::lowResGetTetVolume(int nr) {
    int id0 = lowRes_tetIds[4 * nr];
    int id1 = lowRes_tetIds[4 * nr + 1];
    int id2 = lowRes_tetIds[4 * nr + 2];
    int id3 = lowRes_tetIds[4 * nr + 3];
    VectorMath::vecSetDiff(lowRes_tempBuffer, 0, lowRes_positions, id1, lowRes_positions, id0);
    VectorMath::vecSetDiff(lowRes_tempBuffer, 1, lowRes_positions, id2, lowRes_positions, id0);
    VectorMath::vecSetDiff(lowRes_tempBuffer, 2, lowRes_positions, id3, lowRes_positions, id0);
    VectorMath::vecSetCross(lowRes_tempBuffer, 3, lowRes_tempBuffer, 0, lowRes_tempBuffer, 1);
    return VectorMath::vecDot(lowRes_tempBuffer, 3, lowRes_tempBuffer, 2) / 6.0f;
}

void SoftBodyGPUDuo::lowResSolve(float dt) {
    computeLowResAllStrainLevels();

    if(LowRes_enableStrainLimiting) {
        // ã‚¹ãƒˆãƒ¬ã‚¤ãƒ³ãƒªãƒŸãƒ†ã‚£ãƒ³ã‚°æœ‰åŠ¹æ™‚
        for (int i = 0; i < 5; i++) {
            solveLowResEdgesStrainLimits(lowResEdgeCompliance, dt);
            for (int j = 0; j < 2; j++) {
                solveLowResVolumesStrainLimits(lowResVolCompliance, dt);
                    //solveVolumes(volCompliance, dt);
            }
        }
    } else {
        // é€šå¸¸ã®solve
        for (int i = 0; i < 5; i++) {
            solveLowResEdges(lowResEdgeCompliance, dt);
            for (int j = 0; j < 2; j++) {
                lowResSolveVolumes(lowResVolCompliance, dt);
            }
        }
    }
}

void SoftBodyGPUDuo::lowResSolveVolumes(float compliance, float dt) {
    float alpha = compliance / (dt * dt);
    for (size_t i = 0; i < numLowTets; i++) {
        if (!lowRes_tetValid.empty() && !lowRes_tetValid[i]) continue;
        float w = 0.0f;
        for (int j = 0; j < 4; j++) {
            int id0 = lowRes_tetIds[4 * i + volIdOrder[j][0]];
            int id1 = lowRes_tetIds[4 * i + volIdOrder[j][1]];
            int id2 = lowRes_tetIds[4 * i + volIdOrder[j][2]];
            VectorMath::vecSetDiff(lowRes_tempBuffer, 0, lowRes_positions, id1, lowRes_positions, id0);
            VectorMath::vecSetDiff(lowRes_tempBuffer, 1, lowRes_positions, id2, lowRes_positions, id0);
            VectorMath::vecSetCross(lowRes_grads, j, lowRes_tempBuffer, 0, lowRes_tempBuffer, 1);
            VectorMath::vecScale(lowRes_grads, j, 1.0f/6.0f);
            w += lowRes_invMasses[lowRes_tetIds[4 * i + j]] * VectorMath::vecLengthSquared(lowRes_grads, j);
        }
        if (w == 0.0f) continue;
        float vol = lowResGetTetVolume(i);
        float restVol = lowRes_restVols[i];
        float C = vol - restVol;
        float dLambda = -(C + alpha * lowRes_volLambdas[i]) / (w + alpha);
        lowRes_volLambdas[i] += dLambda;
        for (int j = 0; j < 4; j++) {
            int id = lowRes_tetIds[4 * i + j];
            VectorMath::vecAdd(lowRes_positions, id, lowRes_grads, j, dLambda * lowRes_invMasses[id]);
        }
    }
}

void SoftBodyGPUDuo::lowResPostSolve(float dt) {
    for (size_t i = 0; i < numLowResParticles; i++) {
        if (lowRes_invMasses[i] == 0.0f) continue;
        VectorMath::vecSetDiff(lowRes_velocities, i, lowRes_positions, i, lowRes_prevPositions, i, 1.0f / dt);
    }
}

void SoftBodyGPUDuo::setupLowResMesh(const std::vector<int>& surfaceTriIds) {
    deleteLowHighBuffers();
    glGenVertexArrays(1, &lowResVAO);
    glGenBuffers(1, &lowResVBO);
    glGenBuffers(1, &lowResEBO);
    glGenBuffers(1, &lowResNormalVBO);
    glBindVertexArray(lowResVAO);
    glBindBuffer(GL_ARRAY_BUFFER, lowResVBO);
    glBufferData(GL_ARRAY_BUFFER, lowRes_positions.size() * sizeof(float), lowRes_positions.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, lowResNormalVBO);
    computeLowResNormals();
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, lowResEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, surfaceTriIds.size() * sizeof(int), surfaceTriIds.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void SoftBodyGPUDuo::setupNormalBuffer() {
    normalLines.clear();
    computeLowResNormals();
}

void SoftBodyGPUDuo::computeLowResNormals() {
    std::vector<glm::vec3> normals(numLowResParticles, glm::vec3(0.0f));

    for (size_t i = 0; i < lowResMeshData.tetSurfaceTriIds.size(); i += 3) {
        int id0 = lowResMeshData.tetSurfaceTriIds[i];
        int id1 = lowResMeshData.tetSurfaceTriIds[i + 1];
        int id2 = lowResMeshData.tetSurfaceTriIds[i + 2];

        glm::vec3 p0(lowRes_positions[id0 * 3], lowRes_positions[id0 * 3 + 1], lowRes_positions[id0 * 3 + 2]);
        glm::vec3 p1(lowRes_positions[id1 * 3], lowRes_positions[id1 * 3 + 1], lowRes_positions[id1 * 3 + 2]);
        glm::vec3 p2(lowRes_positions[id2 * 3], lowRes_positions[id2 * 3 + 1], lowRes_positions[id2 * 3 + 2]);

        glm::vec3 edge1 = p1 - p0;
        glm::vec3 edge2 = p2 - p0;
        glm::vec3 normal = glm::cross(edge1, edge2);

        // æ­£è¦åŒ–ä¸è¦ï¼ˆé¢ç©åŠ é‡ï¼‰
        normals[id0] += normal;
        normals[id1] += normal;
        normals[id2] += normal;
    }
    std::vector<float> normalBuffer;
    normalBuffer.reserve(numLowResParticles * 3);
    for (auto& n : normals) {
        float lenSq = glm::dot(n, n);
        if (lenSq > 1e-12f) {
            n *= glm::inversesqrt(lenSq);  // é«˜é€Ÿãªé€†å¹³æ–¹æ ¹
        }
        normalBuffer.push_back(n.x);
        normalBuffer.push_back(n.y);
        normalBuffer.push_back(n.z);
    }
    glBindBuffer(GL_ARRAY_BUFFER, lowResNormalVBO);
    glBufferData(GL_ARRAY_BUFFER, normalBuffer.size() * sizeof(float), normalBuffer.data(), GL_DYNAMIC_DRAW);
}

void SoftBodyGPUDuo::drawLowResTet(ShaderProgram& shader) {
    shader.use();
    glBindVertexArray(lowResVAO);
    glDrawElements(GL_TRIANGLES, lowResMeshData.tetSurfaceTriIds.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void SoftBodyGPUDuo::startLowResGrab(const glm::vec3& pos) {
    float minD2 = std::numeric_limits<float>::max();
    lowRes_grabId = -1;

    struct ParticleDistance {
        int id;
        float distance;
    };
    std::vector<ParticleDistance> sortedParticles;
    lowRes_activeParticles.clear();

    // Step 1: 有効な四面体に含まれる頂点を高速に収集（O(m)）
    std::vector<bool> vertexIsValid(numLowResParticles, false);
    for (size_t t = 0; t < numLowTets; t++) {
        if (!lowRes_tetValid[t]) continue;
        for (int j = 0; j < 4; j++) {
            int vIdx = lowRes_tetIds[t * 4 + j];
            if (vIdx >= 0 && vIdx < static_cast<int>(numLowResParticles)) {
                vertexIsValid[vIdx] = true;
            }
        }
    }

    // Step 2: 有効な頂点のみをループ（O(n)）
    for (size_t i = 0; i < numLowResParticles; i++) {
        if (!vertexIsValid[i]) continue;

        glm::vec3 particlePos(lowRes_positions[i * 3],
                              lowRes_positions[i * 3 + 1],
                              lowRes_positions[i * 3 + 2]);
        glm::vec3 diff = particlePos - pos;
        float d2 = glm::dot(diff, diff);

        if (d2 < minD2) {
            minD2 = d2;
            lowRes_grabId = i;
        }
        sortedParticles.push_back({static_cast<int>(i), d2});
    }

    if (lowRes_grabId < 0) return;

    glm::vec3 grabbedVertexPos(lowRes_positions[lowRes_grabId * 3],
                                lowRes_positions[lowRes_grabId * 3 + 1],
                                lowRes_positions[lowRes_grabId * 3 + 2]);
    lowRes_grabOffset = pos - grabbedVertexPos;

    std::sort(sortedParticles.begin(), sortedParticles.end(),
              [](const ParticleDistance& a, const ParticleDistance& b) {
                  return a.distance < b.distance;
              });

    int numToSelect = static_cast<int>(sortedParticles.size() * 1.0);
    numToSelect = std::max(1, numToSelect);

    for (int i = 0; i < numToSelect && i < static_cast<int>(sortedParticles.size()); i++) {
        lowRes_activeParticles.push_back(sortedParticles[i].id);
    }

    lowRes_oldInvMasses = lowRes_invMasses;
    std::fill(lowRes_invMasses.begin(), lowRes_invMasses.end(), 0.0f);

    for (int id : lowRes_activeParticles) {
        lowRes_invMasses[id] = lowRes_oldInvMasses[id];
    }

    lowRes_invMasses[lowRes_grabId] = 0.0f;

    glm::vec3 correctedPos = pos - lowRes_grabOffset;
    lowRes_positions[lowRes_grabId * 3] = correctedPos.x;
    lowRes_positions[lowRes_grabId * 3 + 1] = correctedPos.y;
    lowRes_positions[lowRes_grabId * 3 + 2] = correctedPos.z;
}

void SoftBodyGPUDuo::moveLowResGrabbed(const glm::vec3& pos, const glm::vec3& vel) {
    if (lowRes_grabId >= 0) {
        glm::vec3 correctedPos = pos - lowRes_grabOffset;
        lowRes_positions[lowRes_grabId * 3] = correctedPos.x;
        lowRes_positions[lowRes_grabId * 3 + 1] = correctedPos.y;
        lowRes_positions[lowRes_grabId * 3 + 2] = correctedPos.z;
    }
}

void SoftBodyGPUDuo::endLowResGrab(const glm::vec3& pos, const glm::vec3& vel) {
    if (lowRes_grabId >= 0) {
        for (int id : lowRes_activeParticles) {
            lowRes_invMasses[id] = lowRes_oldInvMasses[id];
        }
        //lowRes_invMasses[lowRes_grabId] = 0.0f;
        lowRes_velocities[lowRes_grabId * 3] = vel.x;
        lowRes_velocities[lowRes_grabId * 3 + 1] = vel.y;
        lowRes_velocities[lowRes_grabId * 3 + 2] = vel.z;
        lowRes_grabId = -1;
        lowRes_activeParticles.clear();
        lowRes_grabOffset = glm::vec3(0.0f);
    }
}

SoftBodyGPUDuo::MeshData SoftBodyGPUDuo::loadTetMesh(const std::string& filename) {
    SoftBodyGPUDuo::MeshData meshData;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open file: " << filename << std::endl;
        return meshData;
    }
    std::string line;
    bool readingVertices = false, readingTetrahedra = false, readingEdges = false, readingSurfaceTris = false;
    while (std::getline(file, line)) {
        if (line == "VERTICES") {
            readingVertices = true;
            readingTetrahedra = false;
            readingEdges = false;
            readingSurfaceTris = false;
            continue;
        }
        if (line == "TETRAHEDRA") {
            readingVertices = false;
            readingTetrahedra = true;
            readingEdges = false;
            readingSurfaceTris = false;
            continue;
        }
        if (line == "EDGES") {
            readingVertices = false;
            readingTetrahedra = false;
            readingEdges = true;
            readingSurfaceTris = false;
            continue;
        }
        if (line == "SURFACE_TRIANGLES") {
            readingVertices = false;
            readingTetrahedra = false;
            readingEdges = false;
            readingSurfaceTris = true;
            continue;
        }
        std::istringstream ss(line);
        if (readingVertices) {
            float x, y, z;
            if (ss >> x >> y >> z) {
                meshData.verts.push_back(x);
                meshData.verts.push_back(y);
                meshData.verts.push_back(z);
            }
        }
        else if (readingTetrahedra) {
            int v0, v1, v2, v3;
            if (ss >> v0 >> v1 >> v2 >> v3) {
                meshData.tetIds.push_back(v0);
                meshData.tetIds.push_back(v1);
                meshData.tetIds.push_back(v2);
                meshData.tetIds.push_back(v3);
            }
        }
        else if (readingEdges) {
            int e0, e1;
            if (ss >> e0 >> e1) {
                meshData.tetEdgeIds.push_back(e0);
                meshData.tetEdgeIds.push_back(e1);
            }
        }
        else if (readingSurfaceTris) {
            int t0, t1, t2;
            if (ss >> t0 >> t1 >> t2) {
                meshData.tetSurfaceTriIds.push_back(t0);
                meshData.tetSurfaceTriIds.push_back(t1);
                meshData.tetSurfaceTriIds.push_back(t2);
            }
        }
    }
    file.close();
    return meshData;
}

void SoftBodyGPUDuo::updateLowResMesh() {
    glBindBuffer(GL_ARRAY_BUFFER, lowResVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, lowRes_positions.size() * sizeof(float), lowRes_positions.data());
    computeLowResNormals();
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

int SoftBodyGPUDuo::lowResgetMaxEdgeStrainLevel() const {
    int maxLevel = 0;
    for (int level : lowRes_edgeStrainLevel) {
        maxLevel = std::max(maxLevel, level);
    }
    return maxLevel;
}

int SoftBodyGPUDuo::lowResgetMaxVolStrainLevel() const {
    int maxLevel = 0;
    for (size_t i = 0; i < lowRes_volStrainLevel.size(); i++) {
        if (lowRes_tetValid[i]) {
            maxLevel = std::max(maxLevel, lowRes_volStrainLevel[i]);
        }
    }
    return maxLevel;
}

std::vector<int> SoftBodyGPUDuo::lowResgetCriticalTetrahedra() const {
    std::vector<int> criticalTets;
    for (size_t i = 0; i < lowRes_volStrainLevel.size(); i++) {
        if (lowRes_tetValid[i] && lowRes_volStrainLevel[i] >= 2) {  // ãƒãƒ¼ãƒ‰ãƒªãƒŸãƒƒãƒˆä»¥ä¸Š
            criticalTets.push_back(i);
        }
    }
    return criticalTets;
}

void SoftBodyGPUDuo::solveLowResEdgesStrainLimits(float compliance, float dt) {
    float alpha = compliance / (dt * dt);

    for (size_t i = 0; i < lowRes_edgeLengths.size(); i++) {
        if (lowRes_edgeLengths[i] == 0.0f) continue;
        int id0 = lowRes_edgeIds[2 * i];
        int id1 = lowRes_edgeIds[2 * i + 1];

        bool belongsToValidTet = false;
        for (size_t t = 0; t < numLowTets; t++) {
            if (!lowRes_tetValid[t]) continue;
            bool hasId0 = false, hasId1 = false;
            for (int j = 0; j < 4; j++) {
                int vid = lowRes_tetIds[t * 4 + j];
                if (vid == id0) hasId0 = true;
                if (vid == id1) hasId1 = true;
            }
            if (hasId0 && hasId1) {
                belongsToValidTet = true;
                break;
            }
        }

        // â˜…ç„¡åŠ¹ãªå››é¢ä½“ã®ã‚¨ãƒƒã‚¸ã¯ã‚¹ã‚­ãƒƒãƒ—ï¼ˆã“ã‚ŒãŒé‡è¦ï¼ï¼‰
        if (!belongsToValidTet) continue;

        float w0 = lowRes_invMasses[id0];
        float w1 = lowRes_invMasses[id1];
        float w = w0 + w1;
        if (w == 0.0f) continue;

        VectorMath::vecSetDiff(lowRes_grads, 0, lowRes_positions, id0, lowRes_positions, id1);
        float len = std::sqrt(VectorMath::vecLengthSquared(lowRes_grads, 0));
        if (len == 0.0f) continue;

        VectorMath::vecScale(lowRes_grads, 0, 1.0f / len);
        float restLen = lowRes_edgeLengths[i];
        float C = len - restLen;

        float dLambda = -(C + alpha * lowRes_edgeLambdas[i]) / (w + alpha);
        dLambda *= lowRes_edgeStiffnessScale[i];  // äº‹å‰è¨ˆç®—ã•ã‚ŒãŸã‚¹ã‚±ãƒ¼ãƒ«ã‚’ä½¿ç”¨

        lowRes_edgeLambdas[i] += dLambda;

        VectorMath::vecAdd(lowRes_positions, id0, lowRes_grads, 0, dLambda * w0);
        VectorMath::vecAdd(lowRes_positions, id1, lowRes_grads, 0, -dLambda * w1);
    }
}

void SoftBodyGPUDuo::solveLowResVolumesStrainLimits(float compliance, float dt) {
    float alpha = compliance / (dt * dt);

    for (size_t i = 0; i < numLowTets; i++) {
        if (!lowRes_tetValid[i]) continue;
        if (lowRes_volStiffnessScale[i] == 0.0f) continue; // å®Œå…¨ã«å‰›ä½“åŒ–

        float w = 0.0f;

        for (int j = 0; j < 4; j++) {
            int id0 = lowRes_tetIds[4 * i + volIdOrder[j][0]];
            int id1 = lowRes_tetIds[4 * i + volIdOrder[j][1]];
            int id2 = lowRes_tetIds[4 * i + volIdOrder[j][2]];

            VectorMath::vecSetDiff(lowRes_tempBuffer, 0, lowRes_positions, id1, lowRes_positions, id0);
            VectorMath::vecSetDiff(lowRes_tempBuffer, 1, lowRes_positions, id2, lowRes_positions, id0);
            VectorMath::vecSetCross(lowRes_grads, j, lowRes_tempBuffer, 0, lowRes_tempBuffer, 1);
            VectorMath::vecScale(lowRes_grads, j, 1.0f/6.0f);

            w += lowRes_invMasses[lowRes_tetIds[4 * i + j]] * VectorMath::vecLengthSquared(lowRes_grads, j);
        }

        if (w == 0.0f) continue;

        float vol = lowResGetTetVolume(i);
        float restVol = lowRes_restVols[i];
        float C = vol - restVol;

        // äº‹å‰è¨ˆç®—ã•ã‚ŒãŸã‚¹ã‚±ãƒ¼ãƒ«ã‚’ä½¿ç”¨
        float dLambda = -(C + alpha * lowRes_volLambdas[i]) / (w + alpha);
        dLambda *= lowRes_volStiffnessScale[i];

        lowRes_volLambdas[i] += dLambda;

        for (int j = 0; j < 4; j++) {
            int id = lowRes_tetIds[4 * i + j];
            VectorMath::vecAdd(lowRes_positions, id, lowRes_grads, j, dLambda * lowRes_invMasses[id]);
        }
    }
}

void SoftBodyGPUDuo::computeLowResAllStrainLevels() {
    for (size_t i = 0; i < lowRes_edgeLengths.size(); i++) {
        if (lowRes_edgeLengths[i] == 0.0f) {
            lowRes_edgeStrainLevel[i] = 0;
            lowRes_edgeStrains[i] = 1.0f;
            lowRes_edgeStiffnessScale[i] = 1.0f;
            continue;
        }

        int id0 = lowRes_edgeIds[2 * i];
        int id1 = lowRes_edgeIds[2 * i + 1];

        // æœ‰åŠ¹ãªå››é¢ä½“ã«å±žã™ã‚‹ã‹ã®åˆ¤å®šã‚‚ä¸€åº¦ã ã‘
        bool belongsToValidTet = false;
        for (size_t t = 0; t < numLowTets; t++) {
            if (!lowRes_tetValid[t]) continue;
            bool hasId0 = false, hasId1 = false;
            for (int j = 0; j < 4; j++) {
                int vid = lowRes_tetIds[t * 4 + j];
                if (vid == id0) hasId0 = true;
                if (vid == id1) hasId1 = true;
            }
            if (hasId0 && hasId1) {
                belongsToValidTet = true;
                break;
            }
        }

        if (!belongsToValidTet) {
            lowRes_edgeStrainLevel[i] = 0;
            lowRes_edgeStrains[i] = 1.0f;
            lowRes_edgeStiffnessScale[i] = 1.0f;
            continue;
        }

        // ã‚¹ãƒˆãƒ¬ã‚¤ãƒ³è¨ˆç®—
        float len = std::sqrt(VectorMath::vecDistSquared(lowRes_positions, id0, lowRes_positions, id1));
        float restLen = lowRes_edgeLengths[i];
        float strain = len / restLen;
        lowRes_edgeStrains[i] = strain;

        float strainRatio = std::abs(strain);

        // ãƒ¬ãƒ™ãƒ«ã¨ã‚¹ã‚±ãƒ¼ãƒ«ã‚’è¨­å®š
        if (strainRatio > edgeStrainMaxLimit) {
            lowRes_edgeStrainLevel[i] = 3;
            lowRes_edgeStiffnessScale[i] = 0.05f;
        } else if (strainRatio > edgeStrainHardLimit) {
            lowRes_edgeStrainLevel[i] = 2;
            float t = (strainRatio - edgeStrainHardLimit) / (edgeStrainMaxLimit - edgeStrainHardLimit);
            lowRes_edgeStiffnessScale[i] = 0.3f - t * 0.25f;
        } else if (strainRatio > edgeStrainSoftLimit) {
            lowRes_edgeStrainLevel[i] = 1;
            float t = (strainRatio - edgeStrainSoftLimit) / (edgeStrainHardLimit - edgeStrainSoftLimit);
            lowRes_edgeStiffnessScale[i] = 1.0f - t * 0.7f;
        } else {
            lowRes_edgeStrainLevel[i] = 0;
            lowRes_edgeStiffnessScale[i] = 1.0f;
        }
    }

    // ãƒœãƒªãƒ¥ãƒ¼ãƒ ã‚¹ãƒˆãƒ¬ã‚¤ãƒ³ã®è¨ˆç®—
    for (size_t i = 0; i < numLowTets; i++) {
        if (!lowRes_tetValid[i]) {
            lowRes_volStrainLevel[i] = 0;
            lowRes_volStrains[i] = 1.0f;
            lowRes_volStiffnessScale[i] = 1.0f;
            continue;
        }

        float vol = lowResGetTetVolume(i);
        float restVol = lowRes_restVols[i];
        float strain = (restVol != 0.0f) ? vol / restVol : 1.0f;
        lowRes_volStrains[i] = strain;

        float strainRatio = (strain < 1.0f) ? 1.0f / strain : strain;

        // ãƒ¬ãƒ™ãƒ«ã¨ã‚¹ã‚±ãƒ¼ãƒ«ã‚’è¨­å®š

        if (strainRatio > volStrainMaxLimit) {
            lowRes_volStrainLevel[i] = 3;
            lowRes_volStiffnessScale[i] = 0.05f;
        } else if (strainRatio > volStrainHardLimit) {
            lowRes_volStrainLevel[i] = 2;
            float t = (strainRatio - volStrainHardLimit) / (volStrainMaxLimit - volStrainHardLimit);
            lowRes_volStiffnessScale[i] = 0.3f - t * 0.25f;
        } else if (strainRatio > volStrainSoftLimit) {
            lowRes_volStrainLevel[i] = 1;
            float t = (strainRatio - volStrainSoftLimit) / (volStrainHardLimit - volStrainSoftLimit);
            lowRes_volStiffnessScale[i] = 1.0f - t * 0.7f;
        } else {
            lowRes_volStrainLevel[i] = 0;
            lowRes_volStiffnessScale[i] = 1.0f;
        }
    }
}

void SoftBodyGPUDuo::detectAbnormalVerticesLowToHigh(std::vector<bool>& isAbnormal) {
    isAbnormal.clear();
    isAbnormal.resize(numHighResVerts, false);

    int noSkining = 0;

    // ã‚¹ã‚­ãƒ‹ãƒ³ã‚°æƒ…å ±ãŒãªã„é ‚ç‚¹ã‚’æ¤œå‡º
    for (size_t i = 0; i < numHighResVerts; i++) {
        if (8 * i < skinningInfoLowToHigh.size()) {
            float tetIndex = skinningInfoLowToHigh[8 * i];

            // tetIndexãŒè² ã®å ´åˆã€ã‚¹ã‚­ãƒ‹ãƒ³ã‚°æƒ…å ±ãŒãªã„
            if (tetIndex < 0.0f) {
                isAbnormal[i] = true;
                noSkining++;
            }
        }
    }

    // â˜…â˜…â˜… ãƒ‡ãƒãƒƒã‚°: åˆå›žã®ã¿è©³ç´°å‡ºåŠ› â˜…â˜…â˜…
    static bool firstTime = true;
    if (firstTime) {
        std::cout << "\n=== Detection Debug ===" << std::endl;
        std::cout << "  Total vertices: " << numHighResVerts << std::endl;
        std::cout << "  Detected as abnormal (no skinning): " << noSkining << std::endl;
        std::cout << "=======================\n" << std::endl;
        firstTime = false;
    }

    int abnormalCount = std::count(isAbnormal.begin(), isAbnormal.end(), true);
    if (abnormalCount > 0) {
        std::cout << "Detected " << abnormalCount << " unskinned vertices out of "
                  << numHighResVerts << " vertices" << std::endl;
    }
}

void SoftBodyGPUDuo::buildAdjacencyListLowToHigh(std::vector<std::vector<int>>& adjacencyList) {
    adjacencyList.clear();
    adjacencyList.resize(numHighResVerts);

    // é«˜è§£åƒåº¦ãƒ¡ãƒƒã‚·ãƒ¥ã®è¡¨é¢ä¸‰è§’å½¢ã‹ã‚‰éš£æŽ¥é–¢ä¿‚ã‚’æ§‹ç¯‰
    for (size_t i = 0; i < highResMeshData.tetSurfaceTriIds.size(); i += 3) {
        int v0 = highResMeshData.tetSurfaceTriIds[i];
        int v1 = highResMeshData.tetSurfaceTriIds[i + 1];
        int v2 = highResMeshData.tetSurfaceTriIds[i + 2];

        // ç¯„å›²ãƒã‚§ãƒƒã‚¯
        if (v0 < 0 || v0 >= numHighResVerts ||
            v1 < 0 || v1 >= numHighResVerts ||
            v2 < 0 || v2 >= numHighResVerts) {
            continue;
        }

        // åŒæ–¹å‘ã®éš£æŽ¥é–¢ä¿‚ã‚’è¿½åŠ ï¼ˆé‡è¤‡ãƒã‚§ãƒƒã‚¯ä»˜ãï¼‰
        // v0ã®éš£æŽ¥é ‚ç‚¹ã¨ã—ã¦v1, v2ã‚’è¿½åŠ
        if (std::find(adjacencyList[v0].begin(), adjacencyList[v0].end(), v1) == adjacencyList[v0].end()) {
            adjacencyList[v0].push_back(v1);
        }
        if (std::find(adjacencyList[v0].begin(), adjacencyList[v0].end(), v2) == adjacencyList[v0].end()) {
            adjacencyList[v0].push_back(v2);
        }

        // v1ã®éš£æŽ¥é ‚ç‚¹ã¨ã—ã¦v0, v2ã‚’è¿½åŠ
        if (std::find(adjacencyList[v1].begin(), adjacencyList[v1].end(), v0) == adjacencyList[v1].end()) {
            adjacencyList[v1].push_back(v0);
        }
        if (std::find(adjacencyList[v1].begin(), adjacencyList[v1].end(), v2) == adjacencyList[v1].end()) {
            adjacencyList[v1].push_back(v2);
        }

        // v2ã®éš£æŽ¥é ‚ç‚¹ã¨ã—ã¦v0, v1ã‚’è¿½åŠ
        if (std::find(adjacencyList[v2].begin(), adjacencyList[v2].end(), v0) == adjacencyList[v2].end()) {
            adjacencyList[v2].push_back(v0);
        }
        if (std::find(adjacencyList[v2].begin(), adjacencyList[v2].end(), v1) == adjacencyList[v2].end()) {
            adjacencyList[v2].push_back(v1);
        }
    }

    std::cout << "Built adjacency list for " << numHighResVerts << " vertices" << std::endl;
}

void SoftBodyGPUDuo::applySmoothingToVertex(int vertexIdx,
                                      std::vector<float>& corrected_positions,
                                      const std::vector<std::vector<int>>& adjacencyList) {
    if (vertexIdx < 0 || vertexIdx >= numHighResVerts || adjacencyList[vertexIdx].empty()) {
        return;
    }

    const std::vector<int>& neighbors = adjacencyList[vertexIdx];

    // æ­£å¸¸ãªéš£æŽ¥é ‚ç‚¹ã‚’ä½¿ã£ã¦ä½ç½®ã‚’æŽ¨å®š
    glm::vec3 avgPosition(0.0f);
    float totalWeight = 0.0f;
    int validNeighbors = 0;

    for (int neighborIdx : neighbors) {
        if (neighborIdx < 0 || neighborIdx >= numHighResVerts) continue;

        // ã‚¹ã‚­ãƒ‹ãƒ³ã‚°æƒ…å ±ãŒã‚ã‚‹ï¼ˆæ­£å¸¸ãªï¼‰éš£æŽ¥é ‚ç‚¹ã‚’å„ªå…ˆ
        bool hasValidSkinning = (8 * neighborIdx < skinningInfoLowToHigh.size() &&
                                 skinningInfoLowToHigh[8 * neighborIdx] >= 0.0f);

        glm::vec3 neighborPos(
            highRes_positions[3 * neighborIdx],
            highRes_positions[3 * neighborIdx + 1],
            highRes_positions[3 * neighborIdx + 2]
            );

        // å…ƒã®ãƒ¡ãƒƒã‚·ãƒ¥ã§ã®è·é›¢ã«åŸºã¥ãé‡ã¿ä»˜ã‘
        float weight = 1.0f;
        if (!original_highRes_positions.empty() &&
            original_highRes_positions.size() >= 3 * numHighResVerts) {
            glm::vec3 originalPos(
                original_highRes_positions[3 * vertexIdx],
                original_highRes_positions[3 * vertexIdx + 1],
                original_highRes_positions[3 * vertexIdx + 2]
                );
            glm::vec3 originalNeighborPos(
                original_highRes_positions[3 * neighborIdx],
                original_highRes_positions[3 * neighborIdx + 1],
                original_highRes_positions[3 * neighborIdx + 2]
                );
            float distance = glm::length(originalPos - originalNeighborPos);
            weight = (distance > 0.001f) ? 1.0f / distance : 1.0f;
        }

        // æ­£å¸¸ãªé ‚ç‚¹ã«ã‚ˆã‚Šé«˜ã„é‡ã¿ã‚’ä¸Žãˆã‚‹
        if (hasValidSkinning) {
            weight *= 2.0f;
        }

        avgPosition += neighborPos * weight;
        totalWeight += weight;
        validNeighbors++;
    }

    // å¹³å‡ä½ç½®ã‚’è¨ˆç®—
    if (totalWeight > 0.0f && validNeighbors > 0) {
        avgPosition /= totalWeight;

        // è£œæ­£ä½ç½®ã‚’è¨­å®š
        corrected_positions[3 * vertexIdx] = avgPosition.x;
        corrected_positions[3 * vertexIdx + 1] = avgPosition.y;
        corrected_positions[3 * vertexIdx + 2] = avgPosition.z;
    }
}

void SoftBodyGPUDuo::correctUnskinnedVerticesLowToHigh() {
    if (!skinningAdjustParams.enabled) {
        return;
    }

    if (skinningInfoLowToHigh.empty() || numHighResVerts == 0) {
        return;
    }

    // ã‚ªãƒªã‚¸ãƒŠãƒ«ä½ç½®ã‚’ä¿å­˜ï¼ˆåˆå›žã®ã¿ï¼‰
    if (original_highRes_positions.empty()) {
        original_highRes_positions = highResMeshData.verts; // â† åˆæœŸãƒ¡ãƒƒã‚·ãƒ¥ã®ä½ç½®ã‚’ä½¿ç”¨
    }

    // ç•°å¸¸é ‚ç‚¹ã‚’æ¤œå‡º
    std::vector<bool> isAbnormal;
    detectAbnormalVerticesLowToHigh(isAbnormal);

    int abnormalCount = std::count(isAbnormal.begin(), isAbnormal.end(), true);
    if (abnormalCount == 0) {
        return;
    }

    std::cout << "Correcting " << abnormalCount << " unskinned vertices..." << std::endl;

    // â˜…â˜…â˜… ä¿®æ­£: 5%åˆ¶é™ã‚’å‰Šé™¤ï¼ˆå…¨ã¦ã®æœªã‚¹ã‚­ãƒ‹ãƒ³ã‚°é ‚ç‚¹ã‚’è£œæ­£ï¼‰ â˜…â˜…â˜…

    // éš£æŽ¥ãƒªã‚¹ãƒˆã‚’æ§‹ç¯‰
    if (!adjacencyListComputed || cachedAdjacencyList.empty()) {
        buildAdjacencyListLowToHigh(cachedAdjacencyList);
        adjacencyListComputed = true;
    }

    // è£œæ­£ä½ç½®ã‚’è¨ˆç®—
    std::vector<float> corrected_positions = highRes_positions;

    // â˜…â˜…â˜… åˆæœŸåŒ–: ã‚¹ã‚­ãƒ‹ãƒ³ã‚°ã•ã‚Œã¦ã„ãªã„é ‚ç‚¹ã¯å…ƒã®ä½ç½®ã‹ã‚‰é–‹å§‹ â˜…â˜…â˜…
    for (size_t i = 0; i < numHighResVerts; i++) {
        if (isAbnormal[i] && 3 * i + 2 < original_highRes_positions.size()) {
            corrected_positions[3 * i] = original_highRes_positions[3 * i];
            corrected_positions[3 * i + 1] = original_highRes_positions[3 * i + 1];
            corrected_positions[3 * i + 2] = original_highRes_positions[3 * i + 2];
        }
    }

    // è¤‡æ•°å›žã®ã‚¹ãƒ ãƒ¼ã‚¸ãƒ³ã‚°åå¾©
    for (int iter = 0; iter < skinningAdjustParams.maxIterations; iter++) {
        for (size_t i = 0; i < numHighResVerts; i++) {
            if (isAbnormal[i]) {
                applySmoothingToVertex(i, corrected_positions, cachedAdjacencyList);
            }
        }
    }

    // è£œæ­£ã‚’é©ç”¨ï¼ˆãƒ–ãƒ¬ãƒ³ãƒ‰ï¼‰
    for (size_t i = 0; i < numHighResVerts; i++) {
        if (isAbnormal[i]) {
            float blendFactor = skinningAdjustParams.blendFactor;

            // ç•°å¸¸ãªéš£æŽ¥é ‚ç‚¹ãŒå¤šã„å ´åˆã¯è£œæ­£ã‚’å¼·åŒ–
            int abnormalNeighbors = 0;
            if (i < cachedAdjacencyList.size()) {
                for (int neighborIdx : cachedAdjacencyList[i]) {
                    if (neighborIdx >= 0 && neighborIdx < isAbnormal.size() && isAbnormal[neighborIdx]) {
                        abnormalNeighbors++;
                    }
                }
            }

            if (abnormalNeighbors > 2) {
                blendFactor = std::min(blendFactor + 0.3f, 1.0f); // ã‚ˆã‚Šå¼·ãè£œæ­£
            }

            // è£œæ­£ä½ç½®ã‚’é©ç”¨
            highRes_positions[3 * i] = corrected_positions[3 * i];
            highRes_positions[3 * i + 1] = corrected_positions[3 * i + 1];
            highRes_positions[3 * i + 2] = corrected_positions[3 * i + 2];
        }
    }

    std::cout << "Correction applied with blend factor: " << skinningAdjustParams.blendFactor << std::endl;
}



void SoftBodyGPUDuo::updateHighResMesh() {
    if (!useHighResMesh) return;

    for (size_t i = 0; i < numHighResVerts; i++) {
        int tetIdx = static_cast<int>(skinningInfoLowToHigh[8 * i]);

        if (tetIdx < 0 || tetIdx >= numLowTets) {
            continue;  // æœªã‚¹ã‚­ãƒ‹ãƒ³ã‚°é ‚ç‚¹ã¯ã‚¹ã‚­ãƒƒãƒ—
        }

        float b0 = skinningInfoLowToHigh[8 * i + 1];
        float b1 = skinningInfoLowToHigh[8 * i + 2];
        float b2 = skinningInfoLowToHigh[8 * i + 3];
        float b3 = skinningInfoLowToHigh[8 * i + 4];

        int id0 = lowRes_tetIds[4 * tetIdx];
        int id1 = lowRes_tetIds[4 * tetIdx + 1];
        int id2 = lowRes_tetIds[4 * tetIdx + 2];
        int id3 = lowRes_tetIds[4 * tetIdx + 3];

        highRes_positions[i * 3] =
            b0 * lowRes_positions[id0 * 3] + b1 * lowRes_positions[id1 * 3] +
            b2 * lowRes_positions[id2 * 3] + b3 * lowRes_positions[id3 * 3];
        highRes_positions[i * 3 + 1] =
            b0 * lowRes_positions[id0 * 3 + 1] + b1 * lowRes_positions[id1 * 3 + 1] +
            b2 * lowRes_positions[id2 * 3 + 1] + b3 * lowRes_positions[id3 * 3 + 1];
        highRes_positions[i * 3 + 2] =
            b0 * lowRes_positions[id0 * 3 + 2] + b1 * lowRes_positions[id1 * 3 + 2] +
            b2 * lowRes_positions[id2 * 3 + 2] + b3 * lowRes_positions[id3 * 3 + 2];
    }

    // â˜…â˜…â˜… æœªã‚¹ã‚­ãƒ‹ãƒ³ã‚°é ‚ç‚¹ã‚’è£œæ­£ â˜…â˜…â˜…
    if (skinningAdjustParams.enabled) {
        correctUnskinnedVerticesLowToHigh();
    }

    // OpenGLãƒãƒƒãƒ•ã‚¡ã‚’æ›´æ–°
    glBindBuffer(GL_ARRAY_BUFFER, highResVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0,
                    highRes_positions.size() * sizeof(float),
                    highRes_positions.data());

    static int normalUpdateCounter = 0;
    if (++normalUpdateCounter % 3 == 0) {
        computeHighResNormals();
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void SoftBodyGPUDuo::updateHighResValidSurface() {
    highResValidTriangles.clear();

    struct Face {
        int v0, v1, v2;
        int origV0, origV1, origV2;
        int tetIndex;

        Face(int a, int b, int c, int tet) : tetIndex(tet) {
            origV0 = a; origV1 = b; origV2 = c;
            int vertices[3] = {a, b, c};
            std::sort(vertices, vertices + 3);
            v0 = vertices[0]; v1 = vertices[1]; v2 = vertices[2];
        }

        bool operator<(const Face& other) const {
            if (v0 != other.v0) return v0 < other.v0;
            if (v1 != other.v1) return v1 < other.v1;
            return v2 < other.v2;
        }

        bool operator==(const Face& other) const {
            return v0 == other.v0 && v1 == other.v1 && v2 == other.v2;
        }
    };

    std::vector<Face> allFaces;
    size_t numHighResTets = highResMeshData.tetIds.size() / 4;

    for (size_t i = 0; i < numHighResTets; i++) {
        if (!highResTetValid[i]) continue;

        int v0 = highResMeshData.tetIds[i * 4];
        int v1 = highResMeshData.tetIds[i * 4 + 1];
        int v2 = highResMeshData.tetIds[i * 4 + 2];
        int v3 = highResMeshData.tetIds[i * 4 + 3];

        int tetVerts[4] = {v0, v1, v2, v3};

        for (int faceIdx = 0; faceIdx < 4; faceIdx++) {
            int fv0 = tetVerts[TET_FACE_INDICES[faceIdx][0]];
            int fv1 = tetVerts[TET_FACE_INDICES[faceIdx][1]];
            int fv2 = tetVerts[TET_FACE_INDICES[faceIdx][2]];

            allFaces.emplace_back(fv0, fv1, fv2, i);
        }
    }

    std::sort(allFaces.begin(), allFaces.end());

    for (size_t i = 0; i < allFaces.size(); ) {
        if (i + 1 < allFaces.size() && allFaces[i] == allFaces[i + 1]) {
            i += 2;
        } else {
            const Face& face = allFaces[i];
            highResValidTriangles.push_back(face.origV0);
            highResValidTriangles.push_back(face.origV1);
            highResValidTriangles.push_back(face.origV2);
            i++;
        }
    }

    // ★★★ ここから setupHighResMesh() の内容を統合 ★★★

    // 既存のバッファをクリーンアップ
    if (highResVAO != 0) {
        glDeleteVertexArrays(1, &highResVAO);
        highResVAO = 0;
    }
    if (highResVBO != 0) {
        glDeleteBuffers(1, &highResVBO);
        highResVBO = 0;
    }
    if (highResEBO != 0) {
        glDeleteBuffers(1, &highResEBO);
        highResEBO = 0;
    }
    if (highResNormalVBO != 0) {
        glDeleteBuffers(1, &highResNormalVBO);
        highResNormalVBO = 0;
    }

    // 新しいバッファを生成
    glGenVertexArrays(1, &highResVAO);
    glGenBuffers(1, &highResVBO);
    glGenBuffers(1, &highResEBO);
    glGenBuffers(1, &highResNormalVBO);

    glBindVertexArray(highResVAO);

    // 頂点位置
    glBindBuffer(GL_ARRAY_BUFFER, highResVBO);
    glBufferData(GL_ARRAY_BUFFER, highRes_positions.size() * sizeof(float),
                 highRes_positions.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // 法線
    glBindBuffer(GL_ARRAY_BUFFER, highResNormalVBO);
    computeHighResNormals();
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);

    // インデックス（カット後の新しい表面）
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, highResEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 highResValidTriangles.size() * sizeof(int),
                 highResValidTriangles.data(),
                 GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    std::cout << "HighResMesh updated with " << highResValidTriangles.size() / 3
              << " triangles (including cut surfaces)" << std::endl;
}

void SoftBodyGPUDuo::computeHighResNormals() {
    std::vector<glm::vec3> normals(numHighResVerts, glm::vec3(0.0f));

    // ★修正：highResValidTriangles（カット後の表面）を使用
    const std::vector<int>& triangles = highResValidTriangles.empty()
                                            ? highResMeshData.tetSurfaceTriIds
                                            : highResValidTriangles;

    for (size_t i = 0; i < triangles.size(); i += 3) {
        int id0 = triangles[i];
        int id1 = triangles[i + 1];
        int id2 = triangles[i + 2];

        glm::vec3 p0(highRes_positions[id0 * 3], highRes_positions[id0 * 3 + 1], highRes_positions[id0 * 3 + 2]);
        glm::vec3 p1(highRes_positions[id1 * 3], highRes_positions[id1 * 3 + 1], highRes_positions[id1 * 3 + 2]);
        glm::vec3 p2(highRes_positions[id2 * 3], highRes_positions[id2 * 3 + 1], highRes_positions[id2 * 3 + 2]);

        glm::vec3 edge1 = p1 - p0;
        glm::vec3 edge2 = p2 - p0;
        glm::vec3 normal = glm::cross(edge1, edge2);

        normals[id0] += normal;
        normals[id1] += normal;
        normals[id2] += normal;
    }

    std::vector<float> normalBuffer;
    normalBuffer.reserve(numHighResVerts * 3);

    for (auto& n : normals) {
        float lenSq = glm::dot(n, n);
        if (lenSq > 1e-12f) {
            n *= glm::inversesqrt(lenSq);
        }
        normalBuffer.push_back(n.x);
        normalBuffer.push_back(n.y);
        normalBuffer.push_back(n.z);
    }

    glBindBuffer(GL_ARRAY_BUFFER, highResNormalVBO);
    glBufferData(GL_ARRAY_BUFFER, normalBuffer.size() * sizeof(float),
                 normalBuffer.data(), GL_DYNAMIC_DRAW);
}

#include "VoxelSkeletonSegmentation.h"

//==============================================================================
// スケルトンバインディング実装
//==============================================================================
void SoftBodyGPUDuo::bindToSkeleton(const VoxelSkeleton::VesselSegmentation& skeleton) {
    const auto& nodes = skeleton.getNodes();
    const auto& segments = skeleton.getSegments();

    if (nodes.empty() || segments.empty()) {
        std::cerr << "Warning: Empty skeleton data" << std::endl;
        return;
    }

    skeletonBinding.tetToSegmentId.resize(numHighTets, -1);

    std::cout << "Binding " << numHighTets << " tetrahedra to "
              << segments.size() << " segments..." << std::endl;

#pragma omp parallel for
    for (int tetIdx = 0; tetIdx < static_cast<int>(numHighTets); ++tetIdx) {  // OK: int
        glm::vec3 centroid = computeTetCentroid(static_cast<int>(tetIdx));
        int nearestNodeId = findNearestSkeletonNode(centroid, nodes);

        if (nearestNodeId >= 0 && nearestNodeId < static_cast<int>(nodes.size())) {
            skeletonBinding.tetToSegmentId[tetIdx] = nodes[nearestNodeId].segmentId;
        }
    }

    propagateSegmentIdsToVertices();
    assignSegmentColors(static_cast<int>(segments.size()));

    skeletonBinding.isBound = true;
    setupSegmentColorBuffer();

    printSkeletonBindingStats();
}

void SoftBodyGPUDuo::setTetSegmentIds(const std::vector<int>& tetSegmentIds) {
    if (tetSegmentIds.size() != numHighTets) {
        std::cerr << "Error: tetSegmentIds size mismatch" << std::endl;
        return;
    }

    skeletonBinding.tetToSegmentId = tetSegmentIds;
    propagateSegmentIdsToVertices();

    int maxSegId = *std::max_element(tetSegmentIds.begin(), tetSegmentIds.end());
    assignSegmentColors(maxSegId + 1);

    skeletonBinding.isBound = true;
    setupSegmentColorBuffer();
}

void SoftBodyGPUDuo::unbindSkeleton() {
    skeletonBinding.clear();
    deleteSegmentColorBuffer();
}

//==============================================================================
// セグメント選択
//==============================================================================

void SoftBodyGPUDuo::selectSegmentWithDownstream(int segmentId,
                                           const VoxelSkeleton::VesselSegmentation& skeleton) {
    skeletonBinding.selectedSegments.clear();
    skeletonBinding.selectedSegments.insert(segmentId);

    const auto& segments = skeleton.getSegments();

    std::function<void(int)> collectDownstream = [&](int segId) {
        if (segId < 0 || segId >= static_cast<int>(segments.size())) return;

        for (int childId : segments[segId].childIds) {
            if (skeletonBinding.selectedSegments.find(childId) ==
                skeletonBinding.selectedSegments.end()) {
                skeletonBinding.selectedSegments.insert(childId);
                collectDownstream(childId);
            }
        }
    };

    collectDownstream(segmentId);

    segmentColorsNeedUpdate = true;
    updateVertexColors();

    if (skeletonBinding.onSelectionChanged) {
        skeletonBinding.onSelectionChanged(skeletonBinding.selectedSegments);
    }

    std::cout << "Selected segment " << segmentId << " with "
              << skeletonBinding.selectedSegments.size() << " total segments" << std::endl;
}

void SoftBodyGPUDuo::selectSegments(const std::set<int>& segmentIds) {
    skeletonBinding.selectedSegments = segmentIds;
    segmentColorsNeedUpdate = true;
    updateVertexColors();

    if (skeletonBinding.onSelectionChanged) {
        skeletonBinding.onSelectionChanged(skeletonBinding.selectedSegments);
    }
}

void SoftBodyGPUDuo::clearSegmentSelection() {
    skeletonBinding.selectedSegments.clear();
    segmentColorsNeedUpdate = true;
    updateVertexColors();
}

//=============================================================================
// セグメント選択（追加選択対応版）
//=============================================================================
void SoftBodyGPUDuo::selectSegmentWithDownstream(int segmentId,
                                           const VoxelSkeleton::VesselSegmentation& skeleton,
                                           bool addToSelection) {
    if (!addToSelection) {
        skeletonBinding.selectedSegments.clear();
    }
    if (segmentId < 0) return;
    std::set<int> downstream = skeleton.getDownstreamSegments(segmentId);
    downstream.insert(segmentId);
    for (int id : downstream) {
        skeletonBinding.selectedSegments.insert(id);
    }
    updateVertexColors();
    segmentColorsNeedUpdate = true;
    if (skeletonBinding.onSelectionChanged) {
        skeletonBinding.onSelectionChanged(skeletonBinding.selectedSegments);
    }
}

//=============================================================================
// セグメント選択のトグル（Shift+クリック用）
//=============================================================================
void SoftBodyGPUDuo::toggleSegmentSelection(int segmentId,
                                      const VoxelSkeleton::VesselSegmentation& skeleton) {
    if (segmentId < 0) return;
    std::set<int> downstream = skeleton.getDownstreamSegments(segmentId);
    downstream.insert(segmentId);
    bool alreadySelected = (skeletonBinding.selectedSegments.count(segmentId) > 0);
    if (alreadySelected) {
        for (int id : downstream) {
            skeletonBinding.selectedSegments.erase(id);
        }
    } else {
        for (int id : downstream) {
            skeletonBinding.selectedSegments.insert(id);
        }
    }
    updateVertexColors();
    segmentColorsNeedUpdate = true;
    if (skeletonBinding.onSelectionChanged) {
        skeletonBinding.onSelectionChanged(skeletonBinding.selectedSegments);
    }
}
//==============================================================================
// 色設定
//==============================================================================

void SoftBodyGPUDuo::setSegmentColor(int segmentId, const glm::vec4& color) {
    skeletonBinding.segmentColors[segmentId] = color;
    segmentColorsNeedUpdate = true;
}

void SoftBodyGPUDuo::assignSegmentColors(int totalSegments) {
    skeletonBinding.segmentColors.clear();

    for (int i = 0; i < totalSegments; ++i) {
        float hue = static_cast<float>(i) / totalSegments;

        // HSV to RGB 変換
        float h = hue * 6.0f;
        float c = 0.8f;
        float x = c * (1.0f - std::abs(fmod(h, 2.0f) - 1.0f));

        glm::vec3 rgb;
        if (h < 1) rgb = glm::vec3(c, x, 0);
        else if (h < 2) rgb = glm::vec3(x, c, 0);
        else if (h < 3) rgb = glm::vec3(0, c, x);
        else if (h < 4) rgb = glm::vec3(0, x, c);
        else if (h < 5) rgb = glm::vec3(x, 0, c);
        else rgb = glm::vec3(c, 0, x);

        rgb += glm::vec3(0.2f);

        skeletonBinding.segmentColors[i] = glm::vec4(rgb, 1.0f);
    }
}

void SoftBodyGPUDuo::setUnselectedColor(const glm::vec4& color) {
    skeletonBinding.unselectedColor = color;
    segmentColorsNeedUpdate = true;
}

//==============================================================================
// 内部実装
//==============================================================================

glm::vec3 SoftBodyGPUDuo::computeTetCentroid(int tetIndex) const {
    if (tetIndex < 0 || tetIndex >= static_cast<int>(numHighTets)) {
        return glm::vec3(0.0f);
    }

    glm::vec3 centroid(0.0f);
    for (int i = 0; i < 4; ++i) {
        int vIdx = highResTetIds[tetIndex * 4 + i];
        centroid.x += highRes_positions[vIdx * 3 + 0];
        centroid.y += highRes_positions[vIdx * 3 + 1];
        centroid.z += highRes_positions[vIdx * 3 + 2];
    }
    return centroid * 0.25f;
}

int SoftBodyGPUDuo::findNearestSkeletonNode(const glm::vec3& position,
                                      const std::vector<VoxelSkeleton::SkeletonNode>& nodes) const {
    int nearestId = -1;
    float minDistSq = std::numeric_limits<float>::max();

    for (const auto& node : nodes) {
        float distSq = glm::dot(position - node.position, position - node.position);
        if (distSq < minDistSq) {
            minDistSq = distSq;
            nearestId = node.id;
        }
    }

    return nearestId;
}

void SoftBodyGPUDuo::propagateSegmentIdsToVertices() {
    skeletonBinding.vertexToSegmentId.resize(numHighResVerts, -1);

    std::vector<std::map<int, int>> vertexSegmentVotes(numHighResVerts);

    for (size_t tetIdx = 0; tetIdx < numHighTets; ++tetIdx) {
        int segId = skeletonBinding.tetToSegmentId[tetIdx];
        if (segId < 0) continue;

        for (int i = 0; i < 4; ++i) {
            int vIdx = highResTetIds[tetIdx * 4 + i];
            vertexSegmentVotes[vIdx][segId]++;
        }
    }

    for (size_t v = 0; v < numHighResVerts; ++v) {
        int maxVotes = 0;
        int bestSegId = -1;

        for (const auto& pair : vertexSegmentVotes[v]) {
            if (pair.second > maxVotes) {
                maxVotes = pair.second;
                bestSegId = pair.first;
            }
        }

        skeletonBinding.vertexToSegmentId[v] = bestSegId;
    }
}

void SoftBodyGPUDuo::updateVertexColors() {
    if (!skeletonBinding.isBound) return;

    vertexColors.resize(numHighResVerts * 4);

    bool hasSelection = !skeletonBinding.selectedSegments.empty();

    for (size_t v = 0; v < numHighResVerts; ++v) {
        int segId = skeletonBinding.vertexToSegmentId[v];
        glm::vec4 color;

        if (hasSelection) {
            if (skeletonBinding.selectedSegments.count(segId) > 0) {
                auto it = skeletonBinding.segmentColors.find(segId);
                color = (it != skeletonBinding.segmentColors.end())
                            ? it->second : glm::vec4(1.0f, 0.5f, 0.0f, 1.0f);
            } else {
                color = skeletonBinding.unselectedColor;
            }
        } else {
            auto it = skeletonBinding.segmentColors.find(segId);
            color = (it != skeletonBinding.segmentColors.end())
                        ? it->second : glm::vec4(0.7f, 0.7f, 0.7f, 1.0f);
        }

        vertexColors[v * 4 + 0] = color.r;
        vertexColors[v * 4 + 1] = color.g;
        vertexColors[v * 4 + 2] = color.b;
        vertexColors[v * 4 + 3] = color.a;
    }

    segmentColorsNeedUpdate = true;
}

//==============================================================================
// GPU関連
//==============================================================================

void SoftBodyGPUDuo::setupSegmentColorBuffer() {
    if (segmentColorVBO == 0) {
        glGenBuffers(1, &segmentColorVBO);
    }

    updateVertexColors();
    updateSegmentColorBuffer();
}

void SoftBodyGPUDuo::updateSegmentColorBuffer() {
    if (!segmentColorsNeedUpdate || vertexColors.empty()) return;

    glBindBuffer(GL_ARRAY_BUFFER, segmentColorVBO);
    glBufferData(GL_ARRAY_BUFFER,
                 vertexColors.size() * sizeof(float),
                 vertexColors.data(),
                 GL_DYNAMIC_DRAW);

    segmentColorsNeedUpdate = false;
}

void SoftBodyGPUDuo::deleteSegmentColorBuffer() {
    if (segmentColorVBO != 0) {
        glDeleteBuffers(1, &segmentColorVBO);
        segmentColorVBO = 0;
    }
    vertexColors.clear();
}

//==============================================================================
// 描画
//==============================================================================

void SoftBodyGPUDuo::drawSmoothMeshWithSegments(ShaderProgram& shader) {
    if (!skeletonBinding.isBound || smoothedVertices.empty()) {
        drawSmoothMesh(shader);
        return;
    }

    if (segmentColorsNeedUpdate) {
        updateSegmentColorBuffer();
    }

    glBindVertexArray(smoothVAO);

    // 頂点カラー属性を有効化（location = 3）
    glBindBuffer(GL_ARRAY_BUFFER, segmentColorVBO);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(3);

    // 描画
    glDrawElements(GL_TRIANGLES,
                   static_cast<GLsizei>(smoothSurfaceTriIds.size()),
                   GL_UNSIGNED_INT,
                   nullptr);

    // クリーンアップ
    glDisableVertexAttribArray(3);
    glBindVertexArray(0);
}

void SoftBodyGPUDuo::drawHighResMeshWithSegments(ShaderProgram& shader) {
    if (!skeletonBinding.isBound) {
        drawHighResMesh(shader);
        return;
    }

    if (segmentColorsNeedUpdate) {
        updateSegmentColorBuffer();
    }

    glBindVertexArray(highResVAO);

    glBindBuffer(GL_ARRAY_BUFFER, segmentColorVBO);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(3);

    glDrawElements(GL_TRIANGLES,
                   static_cast<GLsizei>(highResMeshData.tetSurfaceTriIds.size()),
                   GL_UNSIGNED_INT,
                   nullptr);

    glDisableVertexAttribArray(3);
    glBindVertexArray(0);
}

//==============================================================================
// ユーティリティ
//==============================================================================

int SoftBodyGPUDuo::getTetSegmentId(int tetIndex) const {
    if (!skeletonBinding.isBound ||
        tetIndex < 0 ||
        tetIndex >= static_cast<int>(skeletonBinding.tetToSegmentId.size())) {
        return -1;
    }
    return skeletonBinding.tetToSegmentId[tetIndex];
}

int SoftBodyGPUDuo::getVertexSegmentId(int vertexIndex) const {
    if (!skeletonBinding.isBound ||
        vertexIndex < 0 ||
        vertexIndex >= static_cast<int>(skeletonBinding.vertexToSegmentId.size())) {
        return -1;
    }
    return skeletonBinding.vertexToSegmentId[vertexIndex];
}

std::vector<int> SoftBodyGPUDuo::getTetsInSegment(int segmentId) const {
    std::vector<int> result;
    if (!skeletonBinding.isBound) return result;

    for (size_t i = 0; i < skeletonBinding.tetToSegmentId.size(); ++i) {
        if (skeletonBinding.tetToSegmentId[i] == segmentId) {
            result.push_back(static_cast<int>(i));
        }
    }
    return result;
}

std::vector<int> SoftBodyGPUDuo::getSurfaceTrianglesInSegment(int segmentId) const {
    std::vector<int> result;
    if (!skeletonBinding.isBound) return result;

    // 表面三角形を走査し、その頂点のセグメントIDをチェック
    const auto& triIds = highResMeshData.tetSurfaceTriIds;
    for (size_t i = 0; i < triIds.size(); i += 3) {
        int v0 = triIds[i];
        int v1 = triIds[i + 1];
        int v2 = triIds[i + 2];

        // 3頂点のうち2つ以上が同じセグメントなら含める
        int seg0 = getVertexSegmentId(v0);
        int seg1 = getVertexSegmentId(v1);
        int seg2 = getVertexSegmentId(v2);

        int matchCount = 0;
        if (seg0 == segmentId) matchCount++;
        if (seg1 == segmentId) matchCount++;
        if (seg2 == segmentId) matchCount++;

        if (matchCount >= 2) {
            result.push_back(static_cast<int>(i / 3));
        }
    }
    return result;
}

void SoftBodyGPUDuo::printSkeletonBindingStats() const {
    if (!skeletonBinding.isBound) {
        std::cout << "Skeleton not bound" << std::endl;
        return;
    }

    std::map<int, int> segmentTetCounts;
    int unboundTets = 0;

    for (int segId : skeletonBinding.tetToSegmentId) {
        if (segId < 0) {
            unboundTets++;
        } else {
            segmentTetCounts[segId]++;
        }
    }

    std::cout << "=== Skeleton Binding Statistics ===" << std::endl;
    std::cout << "  Total tetrahedra: " << numHighTets << std::endl;
    std::cout << "  Bound segments: " << segmentTetCounts.size() << std::endl;
    std::cout << "  Unbound tetrahedra: " << unboundTets << std::endl;

    // 最初の10セグメントだけ表示
    int count = 0;
    for (const auto& pair : segmentTetCounts) {
        if (count++ >= 10) {
            std::cout << "  ... and " << (segmentTetCounts.size() - 10) << " more segments" << std::endl;
            break;
        }
        std::cout << "  Segment " << pair.first << ": " << pair.second << " tets" << std::endl;
    }
    std::cout << "===================================" << std::endl;
}

// =====================================
// 親ソフトボディの設定
// =====================================
void SoftBodyGPUDuo::setParentSoftBody(SoftBodyGPUDuo* parent, const FollowParams& params) {
    if (parent == this) {
        std::cerr << "Error: Cannot set self as parent!" << std::endl;
        return;
    }

    parentSoftBody = parent;
    followParams = params;

    if (parent != nullptr) {
        followMode = FOLLOW_PARENT_SOFTBODY;
        std::cout << "\n=== Setting Parent SoftBody ===" << std::endl;
        std::cout << "  Child LowRes particles: " << numLowResParticles << std::endl;
        std::cout << "  Parent LowRes tets: " << parent->numLowTets << std::endl;

        // スキニング情報を計算
        computeSkinningToParent();
    } else {
        clearParentSoftBody();
    }
}

// =====================================
// 親ソフトボディのクリア
// =====================================
void SoftBodyGPUDuo::clearParentSoftBody() {
    parentSoftBody = nullptr;
    followMode = FOLLOW_NONE;
    skinningToParent.clear();
    isAnchoredToParent.clear();
    numAnchoredVertices = 0;

    // アンカーされていた頂点の質量を復元
    if (!originalInvMasses.empty()) {
        for (size_t i = 0; i < numLowResParticles; i++) {
            lowRes_invMasses[i] = originalInvMasses[i];
        }
    }

    std::cout << "Parent SoftBody cleared" << std::endl;
}

// =====================================
// 親に対するスキニング計算
// =====================================
void SoftBodyGPUDuo::computeSkinningToParent() {
    if (parentSoftBody == nullptr) {
        std::cerr << "Error: No parent SoftBody set!" << std::endl;
        return;
    }

    std::cout << "\n=== Computing Skinning to Parent ===" << std::endl;

    float baryEpsilon = followParams.barycentricEpsilon;
    float maxAcceptableDist = followParams.maxAcceptableDist;
    float lowQualityFactor = followParams.lowQualityFactor;
    float border = followParams.border;

    std::cout << "  Params: epsilon=" << baryEpsilon
              << ", maxDist=" << maxAcceptableDist
              << ", factor=" << lowQualityFactor
              << ", border=" << border << std::endl;

    // スキニング情報を初期化
    // [tetIdx, b0, b1, b2] per vertex（b3は 1-b0-b1-b2 で計算）
    skinningToParent.clear();
    skinningToParent.resize(numLowResParticles * 4, -1.0f);

    isAnchoredToParent.clear();
    isAnchoredToParent.resize(numLowResParticles, false);

    // 元の質量を保存（まだ保存されていなければ）
    if (originalInvMasses.empty()) {
        originalInvMasses = lowRes_invMasses;
    }

    // 親のバウンディングボックスを計算
    glm::vec3 tetMin(std::numeric_limits<float>::max());
    glm::vec3 tetMax(std::numeric_limits<float>::lowest());

    for (size_t i = 0; i < parentSoftBody->numLowResParticles; i++) {
        float x = parentSoftBody->lowRes_positions[i * 3 + 0];
        float y = parentSoftBody->lowRes_positions[i * 3 + 1];
        float z = parentSoftBody->lowRes_positions[i * 3 + 2];
        tetMin.x = std::min(tetMin.x, x);
        tetMin.y = std::min(tetMin.y, y);
        tetMin.z = std::min(tetMin.z, z);
        tetMax.x = std::max(tetMax.x, x);
        tetMax.y = std::max(tetMax.y, y);
        tetMax.z = std::max(tetMax.z, z);
    }

    glm::vec3 tetSize = tetMax - tetMin;
    float maxSize = std::max({tetSize.x, tetSize.y, tetSize.z});
    float spacing = maxSize * 1.0f;

    // 空間ハッシュを作成（子の頂点用）
    Hash hash(spacing, numLowResParticles);
    hash.create(lowRes_positions);

    std::vector<float> minDist(numLowResParticles, std::numeric_limits<float>::max());
    std::vector<int> skinQuality(numLowResParticles, 0);

    // 一時バッファ
    std::vector<float> tetCenter(3, 0.0f);
    std::vector<float> mat(9, 0.0f);
    std::vector<float> bary(4, 0.0f);

    // 各親の四面体について処理
    for (size_t i = 0; i < parentSoftBody->numLowTets; i++) {
        // 無効な四面体はスキップ
        if (!parentSoftBody->lowRes_tetValid[i]) continue;

        // 四面体の中心を計算
        std::fill(tetCenter.begin(), tetCenter.end(), 0.0f);
        for (int j = 0; j < 4; j++) {
            int vid = parentSoftBody->lowRes_tetIds[4 * i + j];
            VectorMath::vecAdd(tetCenter, 0, parentSoftBody->lowRes_positions, vid, 0.25f);
        }

        // バウンディングスフィアの半径
        float rMax = 0.0f;
        for (int j = 0; j < 4; j++) {
            int vid = parentSoftBody->lowRes_tetIds[4 * i + j];
            float r2 = VectorMath::vecDistSquared(tetCenter, 0, parentSoftBody->lowRes_positions, vid);
            rMax = std::max(rMax, std::sqrt(r2));
        }
        rMax += border;

        // 近傍頂点を検索（子の頂点から）
        hash.query(tetCenter, 0, rMax);
        if (hash.querySize == 0) continue;

        // 四面体の頂点インデックス
        int id0 = parentSoftBody->lowRes_tetIds[4 * i + 0];
        int id1 = parentSoftBody->lowRes_tetIds[4 * i + 1];
        int id2 = parentSoftBody->lowRes_tetIds[4 * i + 2];
        int id3 = parentSoftBody->lowRes_tetIds[4 * i + 3];

        // 逆行列を計算（重心座標計算用）
        VectorMath::vecSetDiff(mat, 0, parentSoftBody->lowRes_positions, id0, parentSoftBody->lowRes_positions, id3);
        VectorMath::vecSetDiff(mat, 1, parentSoftBody->lowRes_positions, id1, parentSoftBody->lowRes_positions, id3);
        VectorMath::vecSetDiff(mat, 2, parentSoftBody->lowRes_positions, id2, parentSoftBody->lowRes_positions, id3);
        VectorMath::matSetInverse(mat);

        // 検索された各頂点（子の頂点）に対して処理
        for (int j = 0; j < hash.querySize; j++) {
            int id = hash.queryIds[j];

            // すでに高品質スキニングがある場合はスキップ
            if (skinQuality[id] == 3) continue;

            // 距離チェック
            if (VectorMath::vecDistSquared(lowRes_positions, id, tetCenter, 0) > rMax * rMax)
                continue;

            // 重心座標を計算
            VectorMath::vecSetDiff(bary, 0, lowRes_positions, id, parentSoftBody->lowRes_positions, id3);
            VectorMath::matSetMult(mat, bary, 0, bary, 0);
            bary[3] = 1.0f - bary[0] - bary[1] - bary[2];

            // 品質判定
            bool allPositive = true;
            float minNegative = 0.0f;

            for (int k = 0; k < 4; k++) {
                if (bary[k] < -baryEpsilon) {
                    allPositive = false;
                    minNegative = std::min(minNegative, bary[k]);
                }
            }

            float dist = -minNegative;

            // 品質レベルを決定
            int quality = 0;
            if (allPositive) {
                quality = 3;  // 高品質：四面体内部
            } else if (dist < maxAcceptableDist) {
                quality = 2;  // 中品質：近い外部
            } else if (dist < maxAcceptableDist * lowQualityFactor) {
                quality = 1;  // 低品質：遠い外部
            }

            // より良い結果が見つかった場合は更新
            if (quality > skinQuality[id] || (quality == skinQuality[id] && dist < minDist[id])) {
                skinQuality[id] = quality;
                minDist[id] = dist;

                skinningToParent[4 * id + 0] = static_cast<float>(i);
                skinningToParent[4 * id + 1] = bary[0];
                skinningToParent[4 * id + 2] = bary[1];
                skinningToParent[4 * id + 3] = bary[2];
            }
        }
    }

    // アンカー頂点を設定（高品質・中品質のみ）
    int highQuality = 0, mediumQuality = 0, lowQuality = 0, noSkinning = 0;
    numAnchoredVertices = 0;

    for (size_t i = 0; i < numLowResParticles; i++) {
        if (skinningToParent[4 * i] < 0) {
            noSkinning++;
        } else if (skinQuality[i] == 3) {
            highQuality++;
            isAnchoredToParent[i] = true;
            lowRes_invMasses[i] = 0.0f;  // 固定
            numAnchoredVertices++;
        } else if (skinQuality[i] == 2) {
            mediumQuality++;
            isAnchoredToParent[i] = true;
            lowRes_invMasses[i] = 0.0f;  // 固定
            numAnchoredVertices++;
        } else {
            lowQuality++;
        }
    }

    std::cout << "\n=== Skinning Results ===" << std::endl;
    std::cout << "  Total child vertices: " << numLowResParticles << std::endl;
    std::cout << "  Skinning quality:" << std::endl;
    std::cout << "    High (inside tet): " << highQuality << " -> anchored" << std::endl;
    std::cout << "    Medium (near tet): " << mediumQuality << " -> anchored" << std::endl;
    std::cout << "    Low (far from tet): " << lowQuality << " -> free" << std::endl;
    std::cout << "    None: " << noSkinning << " -> free" << std::endl;
    std::cout << "  Total anchored: " << numAnchoredVertices << std::endl;
    std::cout << "========================\n" << std::endl;
}

// =====================================
// 親の動きに追従
// =====================================
void SoftBodyGPUDuo::updateFromParent() {
    if (parentSoftBody == nullptr || followMode != FOLLOW_PARENT_SOFTBODY) {
        return;
    }

    if (skinningToParent.empty()) {
        return;
    }

    // アンカー頂点の位置を親の四面体に基づいて更新
    for (size_t i = 0; i < numLowResParticles; i++) {
        if (!isAnchoredToParent[i]) continue;

        int tetIdx = static_cast<int>(skinningToParent[4 * i + 0]);
        if (tetIdx < 0) continue;

        // 親の四面体が無効になった場合
        if (!parentSoftBody->lowRes_tetValid[tetIdx]) {
            // アンカーを解除して自由に動けるようにする
            isAnchoredToParent[i] = false;
            lowRes_invMasses[i] = originalInvMasses[i];
            numAnchoredVertices--;
            continue;
        }

        float b0 = skinningToParent[4 * i + 1];
        float b1 = skinningToParent[4 * i + 2];
        float b2 = skinningToParent[4 * i + 3];
        float b3 = 1.0f - b0 - b1 - b2;

        // 親の四面体の頂点インデックス
        int id0 = parentSoftBody->lowRes_tetIds[tetIdx * 4 + 0];
        int id1 = parentSoftBody->lowRes_tetIds[tetIdx * 4 + 1];
        int id2 = parentSoftBody->lowRes_tetIds[tetIdx * 4 + 2];
        int id3 = parentSoftBody->lowRes_tetIds[tetIdx * 4 + 3];

        // 親の四面体の現在位置から補間
        float newX = b0 * parentSoftBody->lowRes_positions[id0 * 3 + 0]
                     + b1 * parentSoftBody->lowRes_positions[id1 * 3 + 0]
                     + b2 * parentSoftBody->lowRes_positions[id2 * 3 + 0]
                     + b3 * parentSoftBody->lowRes_positions[id3 * 3 + 0];

        float newY = b0 * parentSoftBody->lowRes_positions[id0 * 3 + 1]
                     + b1 * parentSoftBody->lowRes_positions[id1 * 3 + 1]
                     + b2 * parentSoftBody->lowRes_positions[id2 * 3 + 1]
                     + b3 * parentSoftBody->lowRes_positions[id3 * 3 + 1];

        float newZ = b0 * parentSoftBody->lowRes_positions[id0 * 3 + 2]
                     + b1 * parentSoftBody->lowRes_positions[id1 * 3 + 2]
                     + b2 * parentSoftBody->lowRes_positions[id2 * 3 + 2]
                     + b3 * parentSoftBody->lowRes_positions[id3 * 3 + 2];

        // 位置を更新
        lowRes_positions[i * 3 + 0] = newX;
        lowRes_positions[i * 3 + 1] = newY;
        lowRes_positions[i * 3 + 2] = newZ;

        // prevPositionsも更新（速度計算のため）
        lowRes_prevPositions[i * 3 + 0] = newX;
        lowRes_prevPositions[i * 3 + 1] = newY;
        lowRes_prevPositions[i * 3 + 2] = newZ;

        // 速度をゼロに（固定されているため）
        lowRes_velocities[i * 3 + 0] = 0.0f;
        lowRes_velocities[i * 3 + 1] = 0.0f;
        lowRes_velocities[i * 3 + 2] = 0.0f;
    }
}

//=============================================================================
// OBJセグメントIDに基づいて四面体を選択
//=============================================================================
void SoftBodyGPUDuo::selectByOBJSegment(int objSegmentId, const std::vector<int>& nodeToOBJSegmentId) {
    if (!isSkeletonBound() || nodeToOBJSegmentId.empty()) {
        return;
    }

    // 該当するスケルトンセグメントを収集
    std::set<int> matchingSegments;

    // 各四面体をチェック
    for (size_t tetIdx = 0; tetIdx < numHighTets; tetIdx++) {
        int segId = skeletonBinding.tetToSegmentId[tetIdx];
        if (segId < 0) continue;

        // このセグメントのノードがOBJセグメント内にあるか確認
        // （skeletonBindingからセグメント情報を取得）
        // 簡易実装：segIdに対応するノードを探す
        matchingSegments.insert(segId);
    }

    // 選択を更新
    skeletonBinding.selectedSegments.clear();

    for (int segId : matchingSegments) {
        // このセグメントがOBJセグメント内にあるか確認
        // 実際にはVesselSegmentationから情報を取得する必要がある
        skeletonBinding.selectedSegments.insert(segId);
    }

    std::cout << "Selected " << skeletonBinding.selectedSegments.size()
              << " segments for OBJ segment S" << objSegmentId << std::endl;
}

//=============================================================================
// OBJセグメント色を更新（バインド済みのIDを使用）
//=============================================================================
void SoftBodyGPUDuo::updateOBJSegmentColors(const VoxelSkeleton::VesselSegmentation& skeleton) {
    if (!skeletonBinding.objSegmentsBound) {
        // まだバインドされていない場合はバインドを実行
        bindOBJSegments(skeleton);
    }

    if (!skeleton.isUsingOBJSegmentation()) {
        useOBJSegmentColors_ = false;
        return;
    }

    // 頂点色配列を初期化
    objSegmentVertexColors_.resize(numHighResVerts * 3, 0.5f);

    // バインド済みのセグメントIDから色を設定
    for (size_t vid = 0; vid < numHighResVerts; vid++) {
        int objSegId = skeletonBinding.vertexToOBJSegmentId[vid];

        glm::vec3 color(0.5f, 0.5f, 0.5f);  // デフォルト
        if (objSegId > 0) {
            color = skeleton.getOBJSegmentColor(objSegId);
        }

        objSegmentVertexColors_[vid * 3] = color.r;
        objSegmentVertexColors_[vid * 3 + 1] = color.g;
        objSegmentVertexColors_[vid * 3 + 2] = color.b;
    }

    useOBJSegmentColors_ = true;
    std::cout << "OBJ segment colors updated from bound data" << std::endl;
}

//=============================================================================
// 選択ハイライト付き
//=============================================================================
void SoftBodyGPUDuo::updateOBJSegmentColorsWithSelection(
    const VoxelSkeleton::VesselSegmentation& skeleton,
    int selectedSegmentId)
{
    if (!skeletonBinding.objSegmentsBound) {
        bindOBJSegments(skeleton);
    }

    if (!skeleton.isUsingOBJSegmentation()) {
        useOBJSegmentColors_ = false;
        return;
    }

    objSegmentVertexColors_.resize(numHighResVerts * 3, 0.2f);

    for (size_t vid = 0; vid < numHighResVerts; vid++) {
        int objSegId = skeletonBinding.vertexToOBJSegmentId[vid];

        glm::vec3 color;
        if (objSegId == selectedSegmentId) {
            // 選択されたセグメント：明るい色
            color = skeleton.getOBJSegmentColor(objSegId);
        } else if (objSegId > 0) {
            // 非選択：暗い色
            glm::vec3 baseColor = skeleton.getOBJSegmentColor(objSegId);
            color = baseColor * 0.3f;
        } else {
            color = glm::vec3(0.2f);
        }

        objSegmentVertexColors_[vid * 3] = color.r;
        objSegmentVertexColors_[vid * 3 + 1] = color.g;
        objSegmentVertexColors_[vid * 3 + 2] = color.b;
    }

    useOBJSegmentColors_ = true;
}

//=============================================================================
// アクセサ
//=============================================================================
int SoftBodyGPUDuo::getTetOBJSegmentId(int tetIndex) const {
    if (!skeletonBinding.objSegmentsBound ||
        tetIndex < 0 ||
        tetIndex >= static_cast<int>(skeletonBinding.tetToOBJSegmentId.size())) {
        return -1;
    }
    return skeletonBinding.tetToOBJSegmentId[tetIndex];
}

int SoftBodyGPUDuo::getVertexOBJSegmentId(int vertexIndex) const {
    if (!skeletonBinding.objSegmentsBound ||
        vertexIndex < 0 ||
        vertexIndex >= static_cast<int>(skeletonBinding.vertexToOBJSegmentId.size())) {
        return -1;
    }
    return skeletonBinding.vertexToOBJSegmentId[vertexIndex];
}

//=============================================================================
// セグメント色の強制更新（スケルトンモードに戻すとき）
//=============================================================================
void SoftBodyGPUDuo::forceUpdateSegmentColors() {
    segmentColorsNeedUpdate = true;
    updateSegmentColorBuffer();
    std::cout << "Segment colors forced update" << std::endl;
}

//=============================================================================
// OBJセグメント色でスムーズメッシュを描画
//=============================================================================
void SoftBodyGPUDuo::drawSmoothMeshWithOBJSegments(ShaderProgram& shader) {
    if (!useOBJSegmentColors_ || objSegmentVertexColors_.empty()) {
        drawSmoothMesh(shader);
        return;
    }

    if (!smoothDisplayMode || smoothedVertices.empty()) {
        return;
    }

    // ★ objSegmentVertexColors_ (RGB) を RGBA に変換して一時バッファに
    size_t numVerts = smoothedVertices.size() / 3;
    std::vector<float> tempColors(numVerts * 4);

    for (size_t i = 0; i < numVerts; i++) {
        if (i * 3 + 2 < objSegmentVertexColors_.size()) {
            tempColors[i * 4 + 0] = objSegmentVertexColors_[i * 3 + 0];
            tempColors[i * 4 + 1] = objSegmentVertexColors_[i * 3 + 1];
            tempColors[i * 4 + 2] = objSegmentVertexColors_[i * 3 + 2];
            tempColors[i * 4 + 3] = 1.0f;
        } else {
            tempColors[i * 4 + 0] = 0.5f;
            tempColors[i * 4 + 1] = 0.5f;
            tempColors[i * 4 + 2] = 0.5f;
            tempColors[i * 4 + 3] = 1.0f;
        }
    }

    // ★ OBJ専用のVBOを使う（vertexColorsを変更しない）
    static GLuint objColorVBO = 0;
    if (objColorVBO == 0) {
        glGenBuffers(1, &objColorVBO);
    }

    glBindBuffer(GL_ARRAY_BUFFER, objColorVBO);
    glBufferData(GL_ARRAY_BUFFER, tempColors.size() * sizeof(float),
                 tempColors.data(), GL_DYNAMIC_DRAW);

    // 既存のsmoothVAOを使って描画
    glBindVertexArray(smoothVAO);

    // 頂点カラー属性を有効化（location = 3）
    glBindBuffer(GL_ARRAY_BUFFER, objColorVBO);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(3);

    // 描画
    glDrawElements(GL_TRIANGLES,
                   static_cast<GLsizei>(smoothSurfaceTriIds.size()),
                   GL_UNSIGNED_INT,
                   nullptr);

    // クリーンアップ
    glDisableVertexAttribArray(3);
    glBindVertexArray(0);
}

//=============================================================================
// OBJセグメント色でHighResメッシュを描画
//=============================================================================
void SoftBodyGPUDuo::drawHighResMeshWithOBJSegments(ShaderProgram& shader) {
    if (!useOBJSegmentColors_ || objSegmentVertexColors_.empty()) {
        drawHighResMesh(shader);
        return;
    }

    // ★ objSegmentVertexColors_ (RGB) を RGBA に変換して一時バッファに
    std::vector<float> tempColors(numHighResVerts * 4);

    for (size_t i = 0; i < numHighResVerts; i++) {
        if (i * 3 + 2 < objSegmentVertexColors_.size()) {
            tempColors[i * 4 + 0] = objSegmentVertexColors_[i * 3 + 0];
            tempColors[i * 4 + 1] = objSegmentVertexColors_[i * 3 + 1];
            tempColors[i * 4 + 2] = objSegmentVertexColors_[i * 3 + 2];
            tempColors[i * 4 + 3] = 1.0f;
        } else {
            tempColors[i * 4 + 0] = 0.5f;
            tempColors[i * 4 + 1] = 0.5f;
            tempColors[i * 4 + 2] = 0.5f;
            tempColors[i * 4 + 3] = 1.0f;
        }
    }

    // ★ OBJ専用のVBOを使う
    static GLuint objHighResColorVBO = 0;
    if (objHighResColorVBO == 0) {
        glGenBuffers(1, &objHighResColorVBO);
    }

    glBindBuffer(GL_ARRAY_BUFFER, objHighResColorVBO);
    glBufferData(GL_ARRAY_BUFFER, tempColors.size() * sizeof(float),
                 tempColors.data(), GL_DYNAMIC_DRAW);

    // 既存のhighResVAOを使って描画
    glBindVertexArray(highResVAO);

    glBindBuffer(GL_ARRAY_BUFFER, objHighResColorVBO);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(3);

    glDrawElements(GL_TRIANGLES,
                   static_cast<GLsizei>(highResMeshData.tetSurfaceTriIds.size()),
                   GL_UNSIGNED_INT,
                   nullptr);

    glDisableVertexAttribArray(3);
    glBindVertexArray(0);
}

//=============================================================================
// OBJセグメントを四面体にバインド（内部判定のみ版）
//=============================================================================
void SoftBodyGPUDuo::bindOBJSegments(const VoxelSkeleton::VesselSegmentation& skeleton) {
    const auto& objSegments = skeleton.getOBJSegments();

    if (objSegments.empty()) {
        std::cout << "No OBJ segments to bind" << std::endl;
        return;
    }

    std::cout << "\n=== Binding OBJ Segments to Tetrahedra ===" << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();

    skeletonBinding.tetToOBJSegmentId.resize(numHighTets, -1);

    // Phase 1: 内部判定のみで割り当て
#pragma omp parallel for schedule(dynamic, 100)
    for (int tetIdx = 0; tetIdx < static_cast<int>(numHighTets); tetIdx++) {
        if (!highResTetValid[tetIdx]) continue;

        glm::vec3 tetCenter(0.0f);
        for (int j = 0; j < 4; j++) {
            int vid = highResMeshData.tetIds[tetIdx * 4 + j];
            tetCenter += glm::vec3(highRes_positions[vid * 3],
                                   highRes_positions[vid * 3 + 1],
                                   highRes_positions[vid * 3 + 2]);
        }
        tetCenter /= 4.0f;

        // ★ 内部判定が真のセグメントのみを候補に
        std::vector<std::pair<int, float>> insideSegments;  // (segId, distToCenter)

        for (size_t s = 0; s < objSegments.size(); s++) {
            const auto& seg = objSegments[s];

            // バウンディングボックスチェック
            if (tetCenter.x < seg.boundMin.x || tetCenter.x > seg.boundMax.x ||
                tetCenter.y < seg.boundMin.y || tetCenter.y > seg.boundMax.y ||
                tetCenter.z < seg.boundMin.z || tetCenter.z > seg.boundMax.z) {
                continue;
            }

            // ★ 内部判定が真の場合のみ候補に追加
            if (skeleton.isInsideOBJSegment(tetCenter, static_cast<int>(s))) {
                glm::vec3 segCenter = (seg.boundMin + seg.boundMax) * 0.5f;
                float dist = glm::length(tetCenter - segCenter);
                insideSegments.push_back({seg.id, dist});
            }
        }

        // 内部と判定されたセグメントがある場合のみ割り当て
        if (!insideSegments.empty()) {
            // 複数ある場合は中心に最も近いものを選ぶ
            int bestId = insideSegments[0].first;
            float bestDist = insideSegments[0].second;
            for (size_t i = 1; i < insideSegments.size(); i++) {
                if (insideSegments[i].second < bestDist) {
                    bestDist = insideSegments[i].second;
                    bestId = insideSegments[i].first;
                }
            }
            skeletonBinding.tetToOBJSegmentId[tetIdx] = bestId;
        }
        // ★ 内部でなければ -1 のまま（割り当てなし）
    }

    auto phase1Time = std::chrono::high_resolution_clock::now();

    // 統計（Phase 1後）
    int assignedCount = 0;
    int unassignedCount = 0;
    for (size_t tetIdx = 0; tetIdx < numHighTets; tetIdx++) {
        if (!highResTetValid[tetIdx]) continue;
        if (skeletonBinding.tetToOBJSegmentId[tetIdx] > 0) {
            assignedCount++;
        } else {
            unassignedCount++;
        }
    }
    std::cout << "  Phase 1 (inside only): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(phase1Time - startTime).count()
              << " ms" << std::endl;
    std::cout << "    Assigned: " << assignedCount << ", Unassigned: " << unassignedCount << std::endl;

    // Phase 2: 頂点→四面体マップを構築
    std::vector<std::vector<int>> vertexToTets(numHighResVerts);
    for (size_t tetIdx = 0; tetIdx < numHighTets; tetIdx++) {
        if (!highResTetValid[tetIdx]) continue;
        for (int j = 0; j < 4; j++) {
            int vid = highResMeshData.tetIds[tetIdx * 4 + j];
            if (vid >= 0 && vid < static_cast<int>(numHighResVerts)) {
                vertexToTets[vid].push_back(static_cast<int>(tetIdx));
            }
        }
    }

    // Phase 3: 未割り当て四面体を隣接から伝播（複数パス）
    std::cout << "  Phase 3 (propagation)..." << std::endl;

    const int maxIterations = 10;
    int totalPropagated = 0;

    for (int iter = 0; iter < maxIterations; iter++) {
        int propagatedThisPass = 0;
        std::vector<int> newSegmentIds = skeletonBinding.tetToOBJSegmentId;

#pragma omp parallel for schedule(dynamic, 100) reduction(+:propagatedThisPass)
        for (int tetIdx = 0; tetIdx < static_cast<int>(numHighTets); tetIdx++) {
            if (!highResTetValid[tetIdx]) continue;

            // すでに割り当て済みならスキップ
            if (skeletonBinding.tetToOBJSegmentId[tetIdx] > 0) continue;

            // 隣接四面体のセグメントIDをカウント
            const int maxSegments = 16;
            std::array<int, maxSegments> neighborVotes = {};

            for (int j = 0; j < 4; j++) {
                int vid = highResMeshData.tetIds[tetIdx * 4 + j];
                for (int otherTet : vertexToTets[vid]) {
                    if (otherTet == tetIdx) continue;
                    int otherSegId = skeletonBinding.tetToOBJSegmentId[otherTet];
                    if (otherSegId > 0 && otherSegId < maxSegments) {
                        neighborVotes[otherSegId]++;
                    }
                }
            }

            // 最も投票が多いセグメントを採用
            int maxVotes = 0;
            int bestSegId = -1;
            for (int s = 1; s < maxSegments; s++) {
                if (neighborVotes[s] > maxVotes) {
                    maxVotes = neighborVotes[s];
                    bestSegId = s;
                }
            }

            if (bestSegId > 0 && maxVotes >= 2) {  // 最低2票必要
                newSegmentIds[tetIdx] = bestSegId;
                propagatedThisPass++;
            }
        }

        skeletonBinding.tetToOBJSegmentId = newSegmentIds;
        totalPropagated += propagatedThisPass;

        if (propagatedThisPass == 0) break;
    }

    std::cout << "    Propagated: " << totalPropagated << " tets" << std::endl;

    // Phase 4: 頂点にセグメントIDを伝播
    skeletonBinding.vertexToOBJSegmentId.resize(numHighResVerts, -1);

    const int maxSegments = 16;

#pragma omp parallel for schedule(static)
    for (int vid = 0; vid < static_cast<int>(numHighResVerts); vid++) {
        std::array<int, maxSegments> votes = {};
        for (int tetIdx : vertexToTets[vid]) {
            int segId = skeletonBinding.tetToOBJSegmentId[tetIdx];
            if (segId > 0 && segId < maxSegments) {
                votes[segId]++;
            }
        }

        int bestSegId = -1;
        int bestCount = 0;
        for (int s = 1; s < maxSegments; s++) {
            if (votes[s] > bestCount) {
                bestCount = votes[s];
                bestSegId = s;
            }
        }
        skeletonBinding.vertexToOBJSegmentId[vid] = bestSegId;
    }

    skeletonBinding.objSegmentsBound = true;

    auto endTime = std::chrono::high_resolution_clock::now();

    // 最終統計
    std::map<int, int> segmentTetCount;
    int finalUnassigned = 0;
    for (size_t tetIdx = 0; tetIdx < numHighTets; tetIdx++) {
        if (!highResTetValid[tetIdx]) continue;
        int segId = skeletonBinding.tetToOBJSegmentId[tetIdx];
        if (segId > 0) {
            segmentTetCount[segId]++;
        } else {
            finalUnassigned++;
        }
    }

    std::cout << "  Final results:" << std::endl;
    for (const auto& kv : segmentTetCount) {
        std::cout << "    S" << kv.first << ": " << kv.second << " tets" << std::endl;
    }
    std::cout << "    Unassigned: " << finalUnassigned << " tets" << std::endl;
    std::cout << "  Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() << " ms" << std::endl;
    std::cout << "=========================================\n" << std::endl;
}

//-----------------------------------------------------------------------------
// 単一四面体の体積を計算
//-----------------------------------------------------------------------------
float SoftBodyGPUDuo::calculateTetVolume(int tetIdx) const {
    if (tetIdx < 0 || tetIdx >= static_cast<int>(numHighTets)) {
        return 0.0f;
    }

    if (!highResTetValid[tetIdx]) {
        return 0.0f;
    }

    // 四面体の4頂点を取得
    int id0 = highResMeshData.tetIds[tetIdx * 4 + 0];
    int id1 = highResMeshData.tetIds[tetIdx * 4 + 1];
    int id2 = highResMeshData.tetIds[tetIdx * 4 + 2];
    int id3 = highResMeshData.tetIds[tetIdx * 4 + 3];

    glm::vec3 v0(highRes_positions[id0 * 3],
                 highRes_positions[id0 * 3 + 1],
                 highRes_positions[id0 * 3 + 2]);
    glm::vec3 v1(highRes_positions[id1 * 3],
                 highRes_positions[id1 * 3 + 1],
                 highRes_positions[id1 * 3 + 2]);
    glm::vec3 v2(highRes_positions[id2 * 3],
                 highRes_positions[id2 * 3 + 1],
                 highRes_positions[id2 * 3 + 2]);
    glm::vec3 v3(highRes_positions[id3 * 3],
                 highRes_positions[id3 * 3 + 1],
                 highRes_positions[id3 * 3 + 2]);

    // 体積 = |det(v1-v0, v2-v0, v3-v0)| / 6
    glm::vec3 e1 = v1 - v0;
    glm::vec3 e2 = v2 - v0;
    glm::vec3 e3 = v3 - v0;

    float det = glm::dot(e1, glm::cross(e2, e3));

    return std::abs(det) / 6.0f;
}

//-----------------------------------------------------------------------------
// 全体体積を計算
//-----------------------------------------------------------------------------
float SoftBodyGPUDuo::calculateTotalVolume() const {
    float totalVolume = 0.0f;

#pragma omp parallel for reduction(+:totalVolume)
    for (int tetIdx = 0; tetIdx < static_cast<int>(numHighTets); tetIdx++) {
        totalVolume += calculateTetVolume(tetIdx);
    }

    return totalVolume;
}

//-----------------------------------------------------------------------------
// 指定セグメントの体積を計算
//-----------------------------------------------------------------------------
float SoftBodyGPUDuo::calculateSegmentVolume(int segmentId, bool useOBJSegmentation) const {
    float segmentVolume = 0.0f;

    if (useOBJSegmentation) {
        // OBJセグメンテーションモード
        if (!skeletonBinding.objSegmentsBound) {
            return 0.0f;
        }

#pragma omp parallel for reduction(+:segmentVolume)
        for (int tetIdx = 0; tetIdx < static_cast<int>(numHighTets); tetIdx++) {
            if (skeletonBinding.tetToOBJSegmentId[tetIdx] == segmentId) {
                segmentVolume += calculateTetVolume(tetIdx);
            }
        }
    } else {
        // スケルトンセグメンテーションモード
        if (!skeletonBinding.isBound) {
            return 0.0f;
        }

#pragma omp parallel for reduction(+:segmentVolume)
        for (int tetIdx = 0; tetIdx < static_cast<int>(numHighTets); tetIdx++) {
            if (skeletonBinding.tetToSegmentId[tetIdx] == segmentId) {
                segmentVolume += calculateTetVolume(tetIdx);
            }
        }
    }

    return segmentVolume;
}

//-----------------------------------------------------------------------------
// 選択中のセグメントの体積を計算
//-----------------------------------------------------------------------------
float SoftBodyGPUDuo::calculateSelectedVolume(bool useOBJSegmentation) const {
    float selectedVolume = 0.0f;

    if (useOBJSegmentation) {
        // OBJセグメンテーションモードでは外部から選択IDを渡す必要がある
        // この関数はスケルトンモード用
        return 0.0f;
    }

    // スケルトンセグメンテーションモード
    if (!skeletonBinding.isBound || skeletonBinding.selectedSegments.empty()) {
        return 0.0f;
    }

#pragma omp parallel for reduction(+:selectedVolume)
    for (int tetIdx = 0; tetIdx < static_cast<int>(numHighTets); tetIdx++) {
        int segId = skeletonBinding.tetToSegmentId[tetIdx];
        if (skeletonBinding.selectedSegments.count(segId) > 0) {
            selectedVolume += calculateTetVolume(tetIdx);
        }
    }

    return selectedVolume;
}

//-----------------------------------------------------------------------------
// 全セグメントの体積マップを取得
//-----------------------------------------------------------------------------
std::map<int, float> SoftBodyGPUDuo::calculateAllSegmentVolumes(bool useOBJSegmentation) const {
    std::map<int, float> volumeMap;

    if (useOBJSegmentation) {
        if (!skeletonBinding.objSegmentsBound) {
            return volumeMap;
        }

        // まず全セグメントIDを収集
        std::set<int> segmentIds;
        for (size_t tetIdx = 0; tetIdx < numHighTets; tetIdx++) {
            int segId = skeletonBinding.tetToOBJSegmentId[tetIdx];
            if (segId > 0) {
                segmentIds.insert(segId);
            }
        }

        // 各セグメントの体積を計算
        for (int segId : segmentIds) {
            volumeMap[segId] = calculateSegmentVolume(segId, true);
        }
    } else {
        if (!skeletonBinding.isBound) {
            return volumeMap;
        }

        // まず全セグメントIDを収集
        std::set<int> segmentIds;
        for (size_t tetIdx = 0; tetIdx < numHighTets; tetIdx++) {
            int segId = skeletonBinding.tetToSegmentId[tetIdx];
            if (segId >= 0) {
                segmentIds.insert(segId);
            }
        }

        // 各セグメントの体積を計算
        for (int segId : segmentIds) {
            volumeMap[segId] = calculateSegmentVolume(segId, false);
        }
    }

    return volumeMap;
}

//-----------------------------------------------------------------------------
// 体積情報を出力
//-----------------------------------------------------------------------------
void SoftBodyGPUDuo::printVolumeInfo(bool useOBJSegmentation, int selectedOBJSegment) const {
    std::cout << "\n========================================" << std::endl;
    std::cout << "         Volume Calculation" << std::endl;
    std::cout << "========================================" << std::endl;

    // 全体体積
    float totalVolume = calculateTotalVolume();
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total Volume: " << totalVolume << " units³" << std::endl;
    std::cout << std::endl;

    if (useOBJSegmentation) {
        std::cout << "[OBJ Segmentation Mode]" << std::endl;

        // 選択中のセグメント
        if (selectedOBJSegment > 0) {
            float selectedVolume = calculateSegmentVolume(selectedOBJSegment, true);
            float ratio = (totalVolume > 0) ? (selectedVolume / totalVolume * 100.0f) : 0.0f;

            std::cout << "  Selected: S" << selectedOBJSegment << std::endl;
            std::cout << "    Volume: " << selectedVolume << " units³" << std::endl;
            std::cout << "    Ratio:  " << ratio << "%" << std::endl;
            std::cout << std::endl;
        }

        // 全セグメント一覧
        std::cout << "  All Segments:" << std::endl;
        auto volumeMap = calculateAllSegmentVolumes(true);

        float assignedTotal = 0.0f;
        for (const auto& kv : volumeMap) {
            float ratio = (totalVolume > 0) ? (kv.second / totalVolume * 100.0f) : 0.0f;
            std::cout << "    S" << kv.first << ": "
                      << std::setw(12) << kv.second << " units³ ("
                      << std::setw(5) << ratio << "%)" << std::endl;
            assignedTotal += kv.second;
        }

        // 未割り当て部分
        float unassignedVolume = totalVolume - assignedTotal;
        if (unassignedVolume > 0.01f) {
            float ratio = (totalVolume > 0) ? (unassignedVolume / totalVolume * 100.0f) : 0.0f;
            std::cout << "    Unassigned: "
                      << std::setw(8) << unassignedVolume << " units³ ("
                      << std::setw(5) << ratio << "%)" << std::endl;
        }

    } else {
        std::cout << "[Skeleton Segmentation Mode]" << std::endl;

        // 選択中のセグメント
        if (!skeletonBinding.selectedSegments.empty()) {
            float selectedVolume = calculateSelectedVolume(false);
            float ratio = (totalVolume > 0) ? (selectedVolume / totalVolume * 100.0f) : 0.0f;

            std::cout << "  Selected segments: ";
            for (int segId : skeletonBinding.selectedSegments) {
                std::cout << segId << " ";
            }
            std::cout << std::endl;
            std::cout << "    Volume: " << selectedVolume << " units³" << std::endl;
            std::cout << "    Ratio:  " << ratio << "%" << std::endl;
            std::cout << std::endl;
        }

        // セグメント数が多い場合は上位10件のみ表示
        auto volumeMap = calculateAllSegmentVolumes(false);

        // 体積でソート（降順）
        std::vector<std::pair<int, float>> sortedVolumes(volumeMap.begin(), volumeMap.end());
        std::sort(sortedVolumes.begin(), sortedVolumes.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        std::cout << "  Top segments by volume:" << std::endl;
        int displayCount = std::min(10, static_cast<int>(sortedVolumes.size()));
        for (int i = 0; i < displayCount; i++) {
            float ratio = (totalVolume > 0) ? (sortedVolumes[i].second / totalVolume * 100.0f) : 0.0f;
            std::cout << "    Seg " << std::setw(3) << sortedVolumes[i].first << ": "
                      << std::setw(12) << sortedVolumes[i].second << " units³ ("
                      << std::setw(5) << ratio << "%)" << std::endl;
        }

        if (sortedVolumes.size() > 10) {
            std::cout << "    ... and " << (sortedVolumes.size() - 10) << " more segments" << std::endl;
        }
    }

    std::cout << "========================================\n" << std::endl;
}

//=============================================================================
// 頂点IDからOBJセグメントIDを取得
//=============================================================================
int SoftBodyGPUDuo::getOBJSegmentIdAtVertex(int vertexId) const {
    if (!skeletonBinding.objSegmentsBound) {
        return -1;
    }

    if (vertexId < 0 || vertexId >= static_cast<int>(skeletonBinding.vertexToOBJSegmentId.size())) {
        return -1;
    }

    return skeletonBinding.vertexToOBJSegmentId[vertexId];
}

//=============================================================================
// レイキャストでOBJセグメントを取得
//=============================================================================
int SoftBodyGPUDuo::raycastOBJSegment(const glm::vec3& rayOrigin, const glm::vec3& rayDir,
                                glm::vec3& hitPoint) const {
    if (!skeletonBinding.objSegmentsBound) {
        return -1;
    }

    float minT = FLT_MAX;
    int hitVertexId = -1;

    // 表面三角形との交差判定
    const auto& triIds = highResMeshData.tetSurfaceTriIds;

    for (size_t i = 0; i < triIds.size(); i += 3) {
        int id0 = triIds[i];
        int id1 = triIds[i + 1];
        int id2 = triIds[i + 2];

        glm::vec3 v0(highRes_positions[id0 * 3],
                     highRes_positions[id0 * 3 + 1],
                     highRes_positions[id0 * 3 + 2]);
        glm::vec3 v1(highRes_positions[id1 * 3],
                     highRes_positions[id1 * 3 + 1],
                     highRes_positions[id1 * 3 + 2]);
        glm::vec3 v2(highRes_positions[id2 * 3],
                     highRes_positions[id2 * 3 + 1],
                     highRes_positions[id2 * 3 + 2]);

        // Möller–Trumbore
        glm::vec3 e1 = v1 - v0;
        glm::vec3 e2 = v2 - v0;
        glm::vec3 h = glm::cross(rayDir, e2);
        float a = glm::dot(e1, h);

        if (std::abs(a) < 1e-7f) continue;

        float f = 1.0f / a;
        glm::vec3 s = rayOrigin - v0;
        float u = f * glm::dot(s, h);

        if (u < 0.0f || u > 1.0f) continue;

        glm::vec3 q = glm::cross(s, e1);
        float v = f * glm::dot(rayDir, q);

        if (v < 0.0f || u + v > 1.0f) continue;

        float t = f * glm::dot(e2, q);

        if (t > 0.001f && t < minT) {
            minT = t;
            hitPoint = rayOrigin + rayDir * t;

            // 最も近い頂点を選択
            float d0 = glm::length(hitPoint - v0);
            float d1 = glm::length(hitPoint - v1);
            float d2 = glm::length(hitPoint - v2);

            if (d0 <= d1 && d0 <= d2) {
                hitVertexId = id0;
            } else if (d1 <= d0 && d1 <= d2) {
                hitVertexId = id1;
            } else {
                hitVertexId = id2;
            }
        }
    }

    if (hitVertexId >= 0) {
        return getOBJSegmentIdAtVertex(hitVertexId);
    }

    return -1;
}

// void SoftBodyGPUDuo::updateHighResPositions() {
//     if (!useHighResMesh) return;

//     for (size_t i = 0; i < numHighResVerts; i++) {
//         int tetIdx = static_cast<int>(skinningInfoLowToHigh[8 * i]);

//         if (tetIdx < 0 || tetIdx >= static_cast<int>(numLowTets)) {
//             // マッピングされていない頂点は元の位置を保持
//             highRes_positions[i * 3] = highResMeshData.verts[i * 3];
//             highRes_positions[i * 3 + 1] = highResMeshData.verts[i * 3 + 1];
//             highRes_positions[i * 3 + 2] = highResMeshData.verts[i * 3 + 2];
//             continue;
//         }

//         // 重心座標
//         float b0 = skinningInfoLowToHigh[8 * i + 1];
//         float b1 = skinningInfoLowToHigh[8 * i + 2];
//         float b2 = skinningInfoLowToHigh[8 * i + 3];
//         float b3 = skinningInfoLowToHigh[8 * i + 4];

//         // 四面体の現在の頂点位置
//         int id0 = lowRes_tetIds[4 * tetIdx];
//         int id1 = lowRes_tetIds[4 * tetIdx + 1];
//         int id2 = lowRes_tetIds[4 * tetIdx + 2];
//         int id3 = lowRes_tetIds[4 * tetIdx + 3];

//         // 重心座標による補間
//         highRes_positions[i * 3] = b0 * lowRes_positions[id0 * 3] +
//                                    b1 * lowRes_positions[id1 * 3] +
//                                    b2 * lowRes_positions[id2 * 3] +
//                                    b3 * lowRes_positions[id3 * 3];

//         highRes_positions[i * 3 + 1] = b0 * lowRes_positions[id0 * 3 + 1] +
//                                        b1 * lowRes_positions[id1 * 3 + 1] +
//                                        b2 * lowRes_positions[id2 * 3 + 1] +
//                                        b3 * lowRes_positions[id3 * 3 + 1];

//         highRes_positions[i * 3 + 2] = b0 * lowRes_positions[id0 * 3 + 2] +
//                                        b1 * lowRes_positions[id1 * 3 + 2] +
//                                        b2 * lowRes_positions[id2 * 3 + 2] +
//                                        b3 * lowRes_positions[id3 * 3 + 2];
//     }
// }

// ============================================================
// 静的メンバ変数の定義
// ============================================================
int SoftBodyGPUDuo::MAX_HANDLE_GROUPS = 5;

// ============================================================
// HandleGroup メンバ関数
// ============================================================

// HandleGroup: 相対位置を保存
void SoftBodyGPUDuo::HandleGroup::storeRelativePositions(const std::vector<float>& positions) {
    relativePositions.clear();

    // 中心頂点の実際の位置を使用
    centerPosition = glm::vec3(
        positions[centerVertex * 3],
        positions[centerVertex * 3 + 1],
        positions[centerVertex * 3 + 2]
        );

    // 中心頂点からの相対位置を計算
    for (int idx : vertices) {
        glm::vec3 vertPos(positions[idx * 3],
                          positions[idx * 3 + 1],
                          positions[idx * 3 + 2]);
        relativePositions.push_back(vertPos - centerPosition);
    }
}

// HandleGroup: 中心位置を更新
void SoftBodyGPUDuo::HandleGroup::updateCenterPosition(const std::vector<float>& positions) {
    centerPosition = glm::vec3(
        positions[centerVertex * 3],
        positions[centerVertex * 3 + 1],
        positions[centerVertex * 3 + 2]
        );
}

// ============================================================
// 頂点検索関数
// ============================================================

// 最近傍頂点を検索（全LowRes頂点から）
int SoftBodyGPUDuo::findClosestVertex(const glm::vec3& position) {
    float minD2 = std::numeric_limits<float>::max();
    int closestId = -1;

    for (size_t i = 0; i < numLowResParticles; i++) {
        glm::vec3 particlePos(lowRes_positions[i * 3],
                              lowRes_positions[i * 3 + 1],
                              lowRes_positions[i * 3 + 2]);
        glm::vec3 diff = particlePos - position;
        float d2 = glm::dot(diff, diff);

        if (d2 < minD2) {
            minD2 = d2;
            closestId = i;
        }
    }

    return closestId;
}

// 頂点位置を取得
glm::vec3 SoftBodyGPUDuo::getVertexPosition(int index) const {
    if (index >= 0 && index < static_cast<int>(numLowResParticles)) {
        return glm::vec3(lowRes_positions[index * 3],
                         lowRes_positions[index * 3 + 1],
                         lowRes_positions[index * 3 + 2]);
    }
    return glm::vec3(0.0f);
}

// ============================================================
// ハンドルグループ管理関数
// ============================================================

// 半径でハンドルグループを作成
bool SoftBodyGPUDuo::createHandleGroupByRadius(const glm::vec3& sphereCenter, float radius) {
    if (handleGroups.size() >= static_cast<size_t>(MAX_HANDLE_GROUPS)) {
        std::cout << "Max handle groups reached: " << MAX_HANDLE_GROUPS << std::endl;
        return false;
    }

    // 元の質量分布を保存（初回のみ）
    if (originalInvMasses.empty()) {
        originalInvMasses = lowRes_invMasses;
    }

    HandleGroup group;
    group.radius = radius;

    // 表面頂点から探す
    int centerIdx = findClosestSurfaceVertex(sphereCenter);
    if (centerIdx < 0) return false;

    group.centerVertex = centerIdx;

    // スフィア中心からの距離で頂点を収集
    for (size_t i = 0; i < numLowResParticles; i++) {
        glm::vec3 vertPos(lowRes_positions[i * 3],
                          lowRes_positions[i * 3 + 1],
                          lowRes_positions[i * 3 + 2]);

        float dist = glm::length(vertPos - sphereCenter);

        if (dist <= radius) {
            group.vertices.push_back(i);

            // 速度をゼロにする
            lowRes_velocities[i * 3] = 0.0f;
            lowRes_velocities[i * 3 + 1] = 0.0f;
            lowRes_velocities[i * 3 + 2] = 0.0f;

            // 前フレームの位置も現在位置に合わせる
            lowRes_prevPositions[i * 3] = lowRes_positions[i * 3];
            lowRes_prevPositions[i * 3 + 1] = lowRes_positions[i * 3 + 1];
            lowRes_prevPositions[i * 3 + 2] = lowRes_positions[i * 3 + 2];

            // 質量を固定
            lowRes_invMasses[i] = 0.0f;
        }
    }

    // 頂点が見つからなかった場合は作成しない
    if (group.vertices.empty()) {
        std::cout << "Warning: No vertices found within radius " << radius
                  << " at position (" << sphereCenter.x << ", "
                  << sphereCenter.y << ", " << sphereCenter.z << ")" << std::endl;
        return false;
    }

    group.storeRelativePositions(lowRes_positions);
    handleGroups.push_back(group);

    std::cout << "Handle group created with " << group.vertices.size() << " vertices" << std::endl;
    return true;
}

// 位置でハンドルグループを検索
int SoftBodyGPUDuo::findHandleGroupAtPosition(const glm::vec3& position, float threshold) {
    for (size_t g = 0; g < handleGroups.size(); g++) {
        glm::vec3 centerPos = handleGroups[g].centerPosition;
        float dist = glm::length(position - centerPos);

        if (dist <= threshold) {
            return g;
        }
    }
    return -1;
}

// ハンドルグループのグラブを試みる
bool SoftBodyGPUDuo::tryStartGrabHandleGroup(const glm::vec3& hitPosition, float threshold) {
    int groupIndex = findHandleGroupAtPosition(hitPosition, threshold);

    if (groupIndex >= 0) {
        activeHandleGroup = groupIndex;
        glm::vec3 centerPos = handleGroups[activeHandleGroup].centerPosition;
        grabOffset = hitPosition - centerPos;

        std::cout << "Grabbed handle group " << activeHandleGroup << std::endl;
        return true;
    }
    return false;
}

// インデックスで直接ハンドルグループをグラブ
void SoftBodyGPUDuo::startGrabHandleGroupByIndex(int groupIndex) {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(handleGroups.size())) {
        std::cout << "Invalid handle group index: " << groupIndex << std::endl;
        return;
    }

    activeHandleGroup = groupIndex;
    currentGrabMode = GRAB_HANDLE_GROUP;
    grabOffset = glm::vec3(0.0f);  // スフィア中心を直接グラブ

    std::cout << "Direct grab handle group " << groupIndex << " (grabOffset = 0)" << std::endl;
}

// グラブ中のハンドルグループを移動
void SoftBodyGPUDuo::moveGrabbedHandleGroup(const glm::vec3& newPosition, const glm::vec3& velocity) {
    if (activeHandleGroup < 0 || activeHandleGroup >= static_cast<int>(handleGroups.size())) return;

    HandleGroup& group = handleGroups[activeHandleGroup];
    glm::vec3 newCenterPos = newPosition - grabOffset;

    for (size_t i = 0; i < group.vertices.size(); i++) {
        int idx = group.vertices[i];
        glm::vec3 newVertPos = newCenterPos + group.relativePositions[i];

        lowRes_positions[idx * 3] = newVertPos.x;
        lowRes_positions[idx * 3 + 1] = newVertPos.y;
        lowRes_positions[idx * 3 + 2] = newVertPos.z;

        lowRes_velocities[idx * 3] = velocity.x;
        lowRes_velocities[idx * 3 + 1] = velocity.y;
        lowRes_velocities[idx * 3 + 2] = velocity.z;
    }

    group.centerPosition = newCenterPos;
}

// ハンドルグループのグラブを終了
void SoftBodyGPUDuo::endGrabHandleGroup(const glm::vec3& position, const glm::vec3& velocity) {
    if (activeHandleGroup >= 0) {
        for (int idx : handleGroups[activeHandleGroup].vertices) {
            lowRes_invMasses[idx] = 0.0f;  // 固定を維持
        }

        std::cout << "Released handle group " << activeHandleGroup << std::endl;
        activeHandleGroup = -1;
    }
}

// ハンドルグループを削除
void SoftBodyGPUDuo::removeHandleGroup(int groupIndex) {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(handleGroups.size())) {
        std::cout << "Invalid handle group index: " << groupIndex << std::endl;
        return;
    }

    std::cout << "Removing handle group " << groupIndex << "..." << std::endl;

    // 削除するグループの頂点を保存
    std::vector<int> removedVertices = handleGroups[groupIndex].vertices;

    // グループを削除
    handleGroups.erase(handleGroups.begin() + groupIndex);

    // アクティブなグループをリセット
    if (activeHandleGroup == groupIndex) {
        activeHandleGroup = -1;
        currentGrabMode = GRAB_NONE;
    } else if (activeHandleGroup > groupIndex) {
        activeHandleGroup--;
    }

    // 元の質量分布を復元
    lowRes_invMasses = originalInvMasses;

    // 残っているハンドルグループの頂点を固定
    for (const auto& group : handleGroups) {
        for (int idx : group.vertices) {
            lowRes_invMasses[idx] = 0.0f;
        }
    }

    // 削除されたグループの頂点だけprevPositionsを更新
    for (int idx : removedVertices) {
        lowRes_velocities[idx * 3] = 0.0f;
        lowRes_velocities[idx * 3 + 1] = 0.0f;
        lowRes_velocities[idx * 3 + 2] = 0.0f;

        lowRes_prevPositions[idx * 3] = lowRes_positions[idx * 3];
        lowRes_prevPositions[idx * 3 + 1] = lowRes_positions[idx * 3 + 1];
        lowRes_prevPositions[idx * 3 + 2] = lowRes_positions[idx * 3 + 2];
    }

    std::cout << "Handle group removed. Remaining groups: " << handleGroups.size() << std::endl;
}

// 全ハンドルグループをクリア
void SoftBodyGPUDuo::clearHandleGroups() {
    // 速度とprevPositionsをリセット
    for (const auto& group : handleGroups) {
        for (int idx : group.vertices) {
            if (idx >= 0 && idx < static_cast<int>(numLowResParticles)) {
                lowRes_velocities[idx * 3] = 0.0f;
                lowRes_velocities[idx * 3 + 1] = 0.0f;
                lowRes_velocities[idx * 3 + 2] = 0.0f;

                lowRes_prevPositions[idx * 3] = lowRes_positions[idx * 3];
                lowRes_prevPositions[idx * 3 + 1] = lowRes_positions[idx * 3 + 1];
                lowRes_prevPositions[idx * 3 + 2] = lowRes_positions[idx * 3 + 2];
            }
        }
    }

    currentGrabMode = GRAB_NONE;
    activeHandleGroup = -1;
    handleGroups.clear();

    // 元の質量分布を復元
    if (!originalInvMasses.empty()) {
        lowRes_invMasses = originalInvMasses;
    }

    std::cout << "Handle groups cleared (mass restored)" << std::endl;
}

// ハンドルグループの頂点位置を取得
std::vector<glm::vec3> SoftBodyGPUDuo::getHandleGroupPositions(int groupIndex) const {
    std::vector<glm::vec3> positions_out;
    if (groupIndex >= 0 && groupIndex < static_cast<int>(handleGroups.size())) {
        for (int idx : handleGroups[groupIndex].vertices) {
            positions_out.push_back(glm::vec3(
                lowRes_positions[idx * 3],
                lowRes_positions[idx * 3 + 1],
                lowRes_positions[idx * 3 + 2]
                ));
        }
    }
    return positions_out;
}

// ============================================================
// スマートグラブ関数（自動判定）
// ============================================================
// スマートグラブ（自動判定）
void SoftBodyGPUDuo::smartGrab(const glm::vec3& hitPosition, float handleThreshold) {
    // まずハンドルグループのグラブを試みる
    if (tryStartGrabHandleGroup(hitPosition, handleThreshold)) {
        currentGrabMode = GRAB_HANDLE_GROUP;
    } else {
        // ハンドルグループがヒットしなければ通常のグラブ
        startLowResGrab(hitPosition);
        currentGrabMode = GRAB_NORMAL;
        std::cout << "Normal grab at position: " << hitPosition.x
                  << ", " << hitPosition.y << ", " << hitPosition.z << std::endl;
    }
}



// スマート移動
void SoftBodyGPUDuo::smartMove(const glm::vec3& newPosition, const glm::vec3& velocity) {
    switch (currentGrabMode) {
    case GRAB_HANDLE_GROUP:
        moveGrabbedHandleGroup(newPosition, velocity);
        break;
    case GRAB_NORMAL:
        moveLowResGrabbed(newPosition, velocity);
        break;
    default:
        break;
    }
}

// スマートグラブ終了
void SoftBodyGPUDuo::smartEndGrab(const glm::vec3& position, const glm::vec3& velocity) {
    switch (currentGrabMode) {
    case GRAB_HANDLE_GROUP:
        endGrabHandleGroup(position, velocity);
        break;
    case GRAB_NORMAL:
        endLowResGrab(position, velocity);
        break;
    default:
        break;
    }
    currentGrabMode = GRAB_NONE;
}

// ============================================================
// エッジ有効性更新
// ============================================================

// LowResエッジの有効性を更新
// void SoftBodyGPUDuo::updateLowResEdgeValidity() {
//     size_t numEdges = lowRes_edgeIds.size() / 2;
//     lowRes_edgeValid.resize(numEdges);
//     std::fill(lowRes_edgeValid.begin(), lowRes_edgeValid.end(), false);

//     for (size_t i = 0; i < numEdges; i++) {
//         int id0 = lowRes_edgeIds[2 * i];
//         int id1 = lowRes_edgeIds[2 * i + 1];

//         // このエッジが有効な四面体に属しているかチェック
//         for (size_t t = 0; t < numLowTets; t++) {
//             if (!lowRes_tetValid[t]) continue;

//             bool hasId0 = false, hasId1 = false;
//             for (int j = 0; j < 4; j++) {
//                 int vid = lowRes_tetIds[t * 4 + j];
//                 if (vid == id0) hasId0 = true;
//                 if (vid == id1) hasId1 = true;
//             }

//             if (hasId0 && hasId1) {
//                 lowRes_edgeValid[i] = true;
//                 break;
//             }
//         }
//     }

//     // 有効なエッジ数をカウント
//     int validEdgeCount = 0;
//     for (bool valid : lowRes_edgeValid) {
//         if (valid) validEdgeCount++;
//     }
//     //std::cout << "  Valid edges: " << validEdgeCount << " / " << numEdges << std::endl;
// }

// スムージングキャッシュを構築
void SoftBodyGPUDuo::buildSmoothingCache() {
    if (smoothingCache.isValid) return;

    std::vector<std::set<int>> tempNeighbors(numHighResVerts);
    std::set<int> tempSurfaceVerts;

    // 隣接関係構築
    for (size_t i = 0; i < smoothSurfaceTriIds.size(); i += 3) {
        int v0 = smoothSurfaceTriIds[i];
        int v1 = smoothSurfaceTriIds[i + 1];
        int v2 = smoothSurfaceTriIds[i + 2];

        tempNeighbors[v0].insert(v1);
        tempNeighbors[v0].insert(v2);
        tempNeighbors[v1].insert(v0);
        tempNeighbors[v1].insert(v2);
        tempNeighbors[v2].insert(v0);
        tempNeighbors[v2].insert(v1);

        tempSurfaceVerts.insert(v0);
        tempSurfaceVerts.insert(v1);
        tempSurfaceVerts.insert(v2);
    }

    // set → vector に変換（高速アクセス用）
    smoothingCache.neighbors.resize(numHighResVerts);
    for (size_t i = 0; i < numHighResVerts; i++) {
        smoothingCache.neighbors[i].assign(
            tempNeighbors[i].begin(),
            tempNeighbors[i].end()
            );
    }

    smoothingCache.surfaceVertexList.assign(
        tempSurfaceVerts.begin(),
        tempSurfaceVerts.end()
        );

    smoothingCache.isValid = true;

    std::cout << "Smoothing cache built: "
              << smoothingCache.surfaceVertexList.size()
              << " surface vertices" << std::endl;
}

// SoftBodyGPUDuo.cpp に追加

void SoftBodyGPUDuo::updateSegmentColorsByMode(const VoxelSkeleton::VesselSegmentation& skeleton) {
    using namespace VoxelSkeleton;

    SegmentationMode mode = skeleton.getSegmentationMode();

    std::cout << "Updating colors for mode: " << getSegmentationModeName(mode) << std::endl;

    // 頂点色バッファのサイズを確保
    size_t numVerts = highRes_positions.size() / 3;
    if (objSegmentVertexColors_.size() != numVerts * 4) {
        objSegmentVertexColors_.resize(numVerts * 4);
    }

    switch (mode) {
    case SegmentationMode::OBJ:
        // 既存のOBJ色更新を使用
        updateOBJSegmentColors(skeleton);
        useOBJSegmentColors_ = true;
        break;

    case SegmentationMode::SkeletonDistance:
        // スケルトンベースの色（既存のセグメント色）
        useOBJSegmentColors_ = false;
        forceUpdateSegmentColors();
        break;

    case SegmentationMode::Voronoi3D:
        // Voronoi3Dブランチの色
        updateVoronoi3DColors(skeleton);
        useOBJSegmentColors_ = true;  // 同じバッファを使用
        break;
    }
}


//=============================================================================
// Voronoi3D バインド（OBJベースと同じ構造）
//=============================================================================
void SoftBodyGPUDuo::bindVoronoi3D(const VoxelSkeleton::VesselSegmentation& skeleton) {
    if (!skeleton.hasVoronoi3D()) {
        std::cout << "No Voronoi3D to bind" << std::endl;
        return;
    }

    const auto* voronoi = skeleton.getVoronoiSegmenter();
    if (!voronoi) {
        std::cout << "VoronoiSegmenter is null" << std::endl;
        return;
    }

    std::cout << "\n=== Binding Voronoi3D to Vertices ===" << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();

    // 頂点→ブランチIDリスト、頂点→基本色のマッピングを構築
    skeletonBinding.vertexToVoronoi3DBranchIds.resize(numHighResVerts);
    skeletonBinding.vertexToVoronoi3DColor.resize(numHighResVerts);

    // 休息状態の座標を使用
    const std::vector<float>* restPositions = &highRes_positions;
    if (!original_highRes_positions.empty() && original_highRes_positions.size() == highRes_positions.size()) {
        restPositions = &original_highRes_positions;
    }

#pragma omp parallel for schedule(dynamic, 1000)
    for (int vid = 0; vid < static_cast<int>(numHighResVerts); vid++) {
        glm::vec3 pos(
            (*restPositions)[vid * 3],
            (*restPositions)[vid * 3 + 1],
            (*restPositions)[vid * 3 + 2]
            );

        // ブランチIDリストと基本色を取得
        skeletonBinding.vertexToVoronoi3DBranchIds[vid] = voronoi->getBranchesAtPosition(pos);
        skeletonBinding.vertexToVoronoi3DColor[vid] = voronoi->getColorAtPosition(pos);
    }

    skeletonBinding.voronoi3DBound = true;
    // 四面体レベルのキャッシュも構築
    bindVoronoi3DTets(skeleton);
    auto endTime = std::chrono::high_resolution_clock::now();
    std::cout << "  Voronoi3D bound to " << numHighResVerts << " vertices in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count()
              << " ms" << std::endl;
}

//=============================================================================
// Voronoi3D 色更新（OBJベースと同じ構造）
//=============================================================================
void SoftBodyGPUDuo::updateVoronoi3DColors(const VoxelSkeleton::VesselSegmentation& skeleton) {
    // まだバインドされていない場合はバインドを実行
    if (!skeletonBinding.voronoi3DBound) {
        bindVoronoi3D(skeleton);
    }

    if (!skeleton.hasVoronoi3D()) {
        useOBJSegmentColors_ = false;
        return;
    }

    // 頂点色配列を初期化
    objSegmentVertexColors_.resize(numHighResVerts * 3, 0.5f);

    // バインド済みのマッピングから色を設定
    for (size_t vid = 0; vid < numHighResVerts; vid++) {
        const glm::vec3& color = skeletonBinding.vertexToVoronoi3DColor[vid];

        objSegmentVertexColors_[vid * 3] = color.r;
        objSegmentVertexColors_[vid * 3 + 1] = color.g;
        objSegmentVertexColors_[vid * 3 + 2] = color.b;
    }

    useOBJSegmentColors_ = true;
    std::cout << "Voronoi3D colors updated from bound data" << std::endl;
}

//=============================================================================
// Voronoi3D 選択ハイライト（int版）
//=============================================================================
void SoftBodyGPUDuo::updateVoronoi3DColorsWithSelection(
    const VoxelSkeleton::VesselSegmentation& skeleton,
    int selectedBranchId)
{
    std::vector<int> branchIds = {selectedBranchId};
    updateVoronoi3DColorsWithSelection(skeleton, branchIds);
}

//=============================================================================
// Voronoi3D 選択ハイライト（vector版）- OBJベースと同じ構造
//=============================================================================
void SoftBodyGPUDuo::updateVoronoi3DColorsWithSelection(
    const VoxelSkeleton::VesselSegmentation& skeleton,
    const std::vector<int>& selectedBranchIds)
{
    // まだバインドされていない場合はバインドを実行
    if (!skeletonBinding.voronoi3DBound) {
        bindVoronoi3D(skeleton);
    }

    if (!skeleton.hasVoronoi3D()) {
        useOBJSegmentColors_ = false;
        return;
    }

    // 選択されたブランチIDをsetに変換（高速検索用）
    std::set<int> selectedSet(selectedBranchIds.begin(), selectedBranchIds.end());

    objSegmentVertexColors_.resize(numHighResVerts * 3, 0.2f);

    int highlightedCount = 0;

    for (size_t vid = 0; vid < numHighResVerts; vid++) {
        const std::vector<int>& branchIds = skeletonBinding.vertexToVoronoi3DBranchIds[vid];
        const glm::vec3& baseColor = skeletonBinding.vertexToVoronoi3DColor[vid];

        glm::vec3 color;

        // 頂点のブランチIDが選択ブランチの「部分集合」かチェック
        bool isSubset = !branchIds.empty();
        for (int bid : branchIds) {
            if (selectedSet.count(bid) == 0) {
                isSubset = false;
                break;
            }
        }

        if (isSubset) {
            // 選択部分：元の色をそのまま使用
            color = baseColor;
            highlightedCount++;
        } else {
            // 非選択部分：元の色を暗くする
            color = baseColor * 0.3f;
        }

        objSegmentVertexColors_[vid * 3] = color.r;
        objSegmentVertexColors_[vid * 3 + 1] = color.g;
        objSegmentVertexColors_[vid * 3 + 2] = color.b;
    }

    useOBJSegmentColors_ = true;
    std::cout << "  Highlighted " << highlightedCount << " / " << numHighResVerts << " vertices" << std::endl;
}


// ========================================
// カット境界頂点の管理関数
// ========================================

void SoftBodyGPUDuo::setCutBoundaryVertices(const std::set<int>& boundaryVertices) {
    highResCutBoundaryVertices = boundaryVertices;
    std::cout << "SoftBody: Set " << boundaryVertices.size()
              << " boundary vertices for enhanced smoothing" << std::endl;
}

const std::set<int>& SoftBodyGPUDuo::getCutBoundaryVertices() const {
    return highResCutBoundaryVertices;
}

size_t SoftBodyGPUDuo::getBoundaryVertexCount() const {
    return highResCutBoundaryVertices.size();
}

void SoftBodyGPUDuo::setBoundaryInfluence(float influence) {
    boundaryInfluence = std::max(0.1f, std::min(1.0f, influence));
    std::cout << "SoftBody: Set boundary influence to "
              << boundaryInfluence << std::endl;
}

float SoftBodyGPUDuo::getBoundaryInfluence() const {
    return boundaryInfluence;
}

// ==================================================================================
// 2. 既存の applySmoothingToSurface 関数を以下に置き換え（約 line 5078〜5126）:
// ==================================================================================

void SoftBodyGPUDuo::applySmoothingToSurface() {
    if (!smoothDisplayMode) return;

    // キャッシュ構築（初回のみ）
    if (!smoothingCache.isValid) {
        buildSmoothingCache();
    }

    const auto& neighbors = smoothingCache.neighbors;
    const auto& surfaceVertexList = smoothingCache.surfaceVertexList;
    const int numSurfaceVerts = static_cast<int>(surfaceVertexList.size());
    const int numVerts = static_cast<int>(numHighResVerts);

    if (numSurfaceVerts == 0) return;

    // ★★★ BoundingBox計算（並列リダクション）★★★
    glm::vec3 originalMin(FLT_MAX), originalMax(-FLT_MAX);

    if (enableSizeAdjustment) {
#pragma omp parallel
        {
            glm::vec3 localMin(FLT_MAX), localMax(-FLT_MAX);

#pragma omp for nowait
            for (int idx = 0; idx < numSurfaceVerts; idx++) {
                int vid = surfaceVertexList[idx];
                float x = smoothedVertices[vid * 3];
                float y = smoothedVertices[vid * 3 + 1];
                float z = smoothedVertices[vid * 3 + 2];

                localMin.x = std::min(localMin.x, x);
                localMin.y = std::min(localMin.y, y);
                localMin.z = std::min(localMin.z, z);
                localMax.x = std::max(localMax.x, x);
                localMax.y = std::max(localMax.y, y);
                localMax.z = std::max(localMax.z, z);
            }

#pragma omp critical
            {
                originalMin.x = std::min(originalMin.x, localMin.x);
                originalMin.y = std::min(originalMin.y, localMin.y);
                originalMin.z = std::min(originalMin.z, localMin.z);
                originalMax.x = std::max(originalMax.x, localMax.x);
                originalMax.y = std::max(originalMax.y, localMax.y);
                originalMax.z = std::max(originalMax.z, localMax.z);
            }
        }
    }

    // ★★★ 初期位置を保存（並列化）★★★
    std::vector<glm::vec3> initialPositions(numVerts);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < numVerts; i++) {
        initialPositions[i] = glm::vec3(
            smoothedVertices[i * 3],
            smoothedVertices[i * 3 + 1],
            smoothedVertices[i * 3 + 2]
            );
    }

    // ★★★ スムージング反復（並列化）★★★
    for (int iter = 0; iter < smoothingIterations; iter++) {
        std::vector<float> newVertices = smoothedVertices;

#pragma omp parallel for schedule(static)
        for (int idx = 0; idx < numSurfaceVerts; idx++) {
            int vid = surfaceVertexList[idx];
            const auto& neighList = neighbors[vid];

            if (neighList.empty()) continue;

            glm::vec3 oldPos(smoothedVertices[vid * 3],
                             smoothedVertices[vid * 3 + 1],
                             smoothedVertices[vid * 3 + 2]);

            // 隣接頂点の平均位置
            glm::vec3 avgPos(0.0f);
            for (int n : neighList) {
                avgPos.x += smoothedVertices[n * 3];
                avgPos.y += smoothedVertices[n * 3 + 1];
                avgPos.z += smoothedVertices[n * 3 + 2];
            }
            avgPos /= float(neighList.size());

            // ★★★ 境界頂点かどうかでスムージング係数を変える ★★★
            float localFactor;
            if (highResCutBoundaryVertices.count(vid) > 0) {
                // カット境界頂点：boundaryInfluenceを使う（強いスムージング）
                localFactor = boundaryInfluence;
            } else {
                // 普通の頂点：smoothingFactorを使う（弱いスムージング）
                localFactor = smoothingFactor;
            }

            // ラプラシアンスムージング
            glm::vec3 newPos = (1.0f - localFactor) * oldPos + localFactor * avgPos;

            // 移動距離制限
            const float maxMoveDistance = 2.0f;
            glm::vec3 moveVector = newPos - initialPositions[vid];
            float moveDistance = glm::length(moveVector);

            if (moveDistance > maxMoveDistance) {
                moveVector = glm::normalize(moveVector) * maxMoveDistance;
                newPos = initialPositions[vid] + moveVector;
            }

            newVertices[vid * 3]     = newPos.x;
            newVertices[vid * 3 + 1] = newPos.y;
            newVertices[vid * 3 + 2] = newPos.z;
        }

        smoothedVertices = newVertices;
    }

    // ★★★ サイズ調整（並列化）★★★
    if (enableSizeAdjustment) {
        // スムージング後のBoundingBox（並列リダクション）
        glm::vec3 smoothedMin(FLT_MAX), smoothedMax(-FLT_MAX);

#pragma omp parallel
        {
            glm::vec3 localMin(FLT_MAX), localMax(-FLT_MAX);

#pragma omp for nowait
            for (int idx = 0; idx < numSurfaceVerts; idx++) {
                int vid = surfaceVertexList[idx];
                float x = smoothedVertices[vid * 3];
                float y = smoothedVertices[vid * 3 + 1];
                float z = smoothedVertices[vid * 3 + 2];

                localMin.x = std::min(localMin.x, x);
                localMin.y = std::min(localMin.y, y);
                localMin.z = std::min(localMin.z, z);
                localMax.x = std::max(localMax.x, x);
                localMax.y = std::max(localMax.y, y);
                localMax.z = std::max(localMax.z, z);
            }

#pragma omp critical
            {
                smoothedMin.x = std::min(smoothedMin.x, localMin.x);
                smoothedMin.y = std::min(smoothedMin.y, localMin.y);
                smoothedMin.z = std::min(smoothedMin.z, localMin.z);
                smoothedMax.x = std::max(smoothedMax.x, localMax.x);
                smoothedMax.y = std::max(smoothedMax.y, localMax.y);
                smoothedMax.z = std::max(smoothedMax.z, localMax.z);
            }
        }

        // スケーリング係数計算
        glm::vec3 originalSize = originalMax - originalMin;
        glm::vec3 smoothedSize = smoothedMax - smoothedMin;
        glm::vec3 originalCenter = (originalMin + originalMax) * 0.5f;
        glm::vec3 smoothedCenter = (smoothedMin + smoothedMax) * 0.5f;

        glm::vec3 scale(1.0f);
        if (smoothedSize.x > 1e-6f) scale.x = originalSize.x / smoothedSize.x;
        if (smoothedSize.y > 1e-6f) scale.y = originalSize.y / smoothedSize.y;
        if (smoothedSize.z > 1e-6f) scale.z = originalSize.z / smoothedSize.z;

        // スケーリング方法の選択
        float uniformScale = 1.0f;
        if (scalingMethod == 0) {
            uniformScale = std::min({scale.x, scale.y, scale.z});
        } else if (scalingMethod == 1) {
            uniformScale = std::max({scale.x, scale.y, scale.z});
        } else {
            uniformScale = (scale.x + scale.y + scale.z) / 3.0f;
        }

// 頂点位置の調整（並列化）
#pragma omp parallel for schedule(static)
        for (int idx = 0; idx < numSurfaceVerts; idx++) {
            int vid = surfaceVertexList[idx];

            glm::vec3 pos(smoothedVertices[vid * 3],
                          smoothedVertices[vid * 3 + 1],
                          smoothedVertices[vid * 3 + 2]);

            glm::vec3 relPos = pos - smoothedCenter;
            relPos *= uniformScale;
            glm::vec3 newPos = originalCenter + relPos;

            smoothedVertices[vid * 3]     = newPos.x;
            smoothedVertices[vid * 3 + 1] = newPos.y;
            smoothedVertices[vid * 3 + 2] = newPos.z;
        }
    }
}


void SoftBodyGPUDuo::computeSkinningInfoLowToHigh(const std::vector<float>& highResVerts) {
    std::cout << "Computing relaxed tetrahedral skinning (highly optimized)..." << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();

    skinningInfoLowToHigh.clear();
    skinningInfoLowToHigh.resize(8 * numHighResVerts, -1.0f);

    int numHighResTetsInt = static_cast<int>(highResMeshData.tetIds.size() / 4);
    int numLowTetsInt = static_cast<int>(numLowTets);
    const float tolerance = 0.9f;

    // ========================================
    // ステップ1: LowRes四面体のAABB + 逆行列を事前計算
    // ========================================
    struct LowTetCache {
        glm::vec3 min, max;       // AABB
        glm::vec3 v0, v1, v2, v3; // 頂点座標
        glm::mat3 T_inv;          // 逆行列（事前計算）
        bool valid;               // 逆行列が有効か
    };

    std::vector<LowTetCache> lowTetCache(numLowTetsInt);

    // ★ globalMin/Max を並列で計算（reductionなしで）
    glm::vec3 globalMin(std::numeric_limits<float>::max());
    glm::vec3 globalMax(std::numeric_limits<float>::lowest());

    // まず逆行列とAABBを並列計算
    #pragma omp parallel for
    for (int i = 0; i < numLowTetsInt; i++) {
        int lv0 = lowRes_tetIds[i * 4 + 0];
        int lv1 = lowRes_tetIds[i * 4 + 1];
        int lv2 = lowRes_tetIds[i * 4 + 2];
        int lv3 = lowRes_tetIds[i * 4 + 3];

        LowTetCache& cache = lowTetCache[i];
        cache.v0 = glm::vec3(lowResMeshData.verts[lv0 * 3], lowResMeshData.verts[lv0 * 3 + 1], lowResMeshData.verts[lv0 * 3 + 2]);
        cache.v1 = glm::vec3(lowResMeshData.verts[lv1 * 3], lowResMeshData.verts[lv1 * 3 + 1], lowResMeshData.verts[lv1 * 3 + 2]);
        cache.v2 = glm::vec3(lowResMeshData.verts[lv2 * 3], lowResMeshData.verts[lv2 * 3 + 1], lowResMeshData.verts[lv2 * 3 + 2]);
        cache.v3 = glm::vec3(lowResMeshData.verts[lv3 * 3], lowResMeshData.verts[lv3 * 3 + 1], lowResMeshData.verts[lv3 * 3 + 2]);

        cache.min = glm::min(glm::min(cache.v0, cache.v1), glm::min(cache.v2, cache.v3));
        cache.max = glm::max(glm::max(cache.v0, cache.v1), glm::max(cache.v2, cache.v3));

        // tolerance分拡張
        float expand = tolerance * 0.5f;
        cache.min -= glm::vec3(expand);
        cache.max += glm::vec3(expand);

        // 逆行列を事前計算
        glm::mat3 T;
        T[0] = cache.v0 - cache.v3;
        T[1] = cache.v1 - cache.v3;
        T[2] = cache.v2 - cache.v3;

        float det = glm::determinant(T);
        if (std::abs(det) > 1e-10f) {
            cache.T_inv = glm::inverse(T);
            cache.valid = true;
        } else {
            cache.valid = false;
        }
    }

    // ★ globalMin/Max はシーケンシャルに計算（軽い処理なので問題なし）
    for (int i = 0; i < numLowTetsInt; i++) {
        globalMin = glm::min(globalMin, lowTetCache[i].min);
        globalMax = glm::max(globalMax, lowTetCache[i].max);
    }

    // ========================================
    // ステップ2: 空間グリッド構築
    // ========================================
    const int GRID_RES = 32;
    glm::vec3 gridSize = globalMax - globalMin;
    glm::vec3 cellSize = gridSize / float(GRID_RES);

    // ゼロ除算防止
    for (int i = 0; i < 3; i++) {
        if (cellSize[i] < 1e-6f) cellSize[i] = 1.0f;
    }

    std::vector<std::vector<int>> grid(GRID_RES * GRID_RES * GRID_RES);

    auto posToCell = [&](const glm::vec3& pos) -> glm::ivec3 {
        glm::vec3 rel = pos - globalMin;
        return glm::clamp(glm::ivec3(rel / cellSize), glm::ivec3(0), glm::ivec3(GRID_RES - 1));
    };

    for (int i = 0; i < numLowTetsInt; i++) {
        glm::ivec3 minCell = posToCell(lowTetCache[i].min);
        glm::ivec3 maxCell = posToCell(lowTetCache[i].max);

        for (int z = minCell.z; z <= maxCell.z; z++) {
            for (int y = minCell.y; y <= maxCell.y; y++) {
                for (int x = minCell.x; x <= maxCell.x; x++) {
                    int cellIdx = z * GRID_RES * GRID_RES + y * GRID_RES + x;
                    grid[cellIdx].push_back(i);
                }
            }
        }
    }

    std::cout << "  Grid built: " << GRID_RES << "^3 cells" << std::endl;
    std::cout << "  LowTet inverse matrices pre-computed" << std::endl;

    // ========================================
    // ステップ3: 並列処理でHighRes四面体をマッピング
    // ========================================
    std::atomic<int> successfullyMapped(0);
    std::atomic<int> unmapped(0);

    #pragma omp parallel
    {
        // スレッドローカルな候補リスト（毎回newしない）
        std::vector<int> candidateLowTets;
        candidateLowTets.reserve(256);

        // スレッドローカルな一時変数
        glm::vec4 tempBaryCoords[4];
        glm::vec4 bestBaryCoords[4];

        #pragma omp for schedule(dynamic, 256)
        for (int highTetIdx = 0; highTetIdx < numHighResTetsInt; highTetIdx++) {
            int hv0 = highResMeshData.tetIds[highTetIdx * 4 + 0];
            int hv1 = highResMeshData.tetIds[highTetIdx * 4 + 1];
            int hv2 = highResMeshData.tetIds[highTetIdx * 4 + 2];
            int hv3 = highResMeshData.tetIds[highTetIdx * 4 + 3];

            int highVertices[4] = {hv0, hv1, hv2, hv3};

            glm::vec3 highVertPos[4];
            glm::vec3 highTetMin(std::numeric_limits<float>::max());
            glm::vec3 highTetMax(std::numeric_limits<float>::lowest());

            for (int i = 0; i < 4; i++) {
                int vid = highVertices[i];
                highVertPos[i] = glm::vec3(
                    highResVerts[vid * 3],
                    highResVerts[vid * 3 + 1],
                    highResVerts[vid * 3 + 2]
                );
                highTetMin = glm::min(highTetMin, highVertPos[i]);
                highTetMax = glm::max(highTetMax, highVertPos[i]);
            }

            // グリッドから候補を取得（std::setではなくvectorを再利用）
            candidateLowTets.clear();
            glm::ivec3 minCell = posToCell(highTetMin);
            glm::ivec3 maxCell = posToCell(highTetMax);

            for (int z = minCell.z; z <= maxCell.z; z++) {
                for (int y = minCell.y; y <= maxCell.y; y++) {
                    for (int x = minCell.x; x <= maxCell.x; x++) {
                        int cellIdx = z * GRID_RES * GRID_RES + y * GRID_RES + x;
                        const auto& cell = grid[cellIdx];
                        candidateLowTets.insert(candidateLowTets.end(), cell.begin(), cell.end());
                    }
                }
            }

            // 重複除去（ソートして連続する重複を削除）
            std::sort(candidateLowTets.begin(), candidateLowTets.end());
            candidateLowTets.erase(std::unique(candidateLowTets.begin(), candidateLowTets.end()), candidateLowTets.end());

            int bestLowTet = -1;
            float bestScore = std::numeric_limits<float>::max();

            for (int lowTetIdx : candidateLowTets) {
                const LowTetCache& cache = lowTetCache[lowTetIdx];
                if (!cache.valid) continue;

                bool allNearlyInside = true;
                float maxOutsideDist = 0.0f;

                for (int i = 0; i < 4; i++) {
                    // 事前計算済みの逆行列を使用
                    glm::vec3 localPos = highVertPos[i] - cache.v3;
                    glm::vec3 baryCoords = cache.T_inv * localPos;

                    float b0 = baryCoords.x;
                    float b1 = baryCoords.y;
                    float b2 = baryCoords.z;
                    float b3 = 1.0f - b0 - b1 - b2;

                    tempBaryCoords[i] = glm::vec4(b0, b1, b2, b3);

                    float outsideDist = std::max({-b0, -b1, -b2, -b3, 0.0f});

                    if (outsideDist > tolerance) {
                        allNearlyInside = false;
                        break;
                    }

                    maxOutsideDist = std::max(maxOutsideDist, outsideDist);
                }

                if (allNearlyInside && maxOutsideDist < bestScore) {
                    bestScore = maxOutsideDist;
                    bestLowTet = lowTetIdx;
                    for (int i = 0; i < 4; i++) {
                        bestBaryCoords[i] = tempBaryCoords[i];
                    }

                    // 完全に内部なら早期終了
                    if (bestScore < 1e-6f) break;
                }
            }

            if (bestLowTet >= 0) {
                for (int i = 0; i < 4; i++) {
                    int vid = highVertices[i];
                    skinningInfoLowToHigh[8 * vid + 0] = static_cast<float>(bestLowTet);
                    skinningInfoLowToHigh[8 * vid + 1] = bestBaryCoords[i].x;
                    skinningInfoLowToHigh[8 * vid + 2] = bestBaryCoords[i].y;
                    skinningInfoLowToHigh[8 * vid + 3] = bestBaryCoords[i].z;
                    skinningInfoLowToHigh[8 * vid + 4] = bestBaryCoords[i].w;
                }
                successfullyMapped++;
            } else {
                unmapped++;
            }
        }
    }

    // 統計情報
    int mappedVerts = 0;
    for (size_t i = 0; i < numHighResVerts; i++) {
        if (skinningInfoLowToHigh[8 * i] >= 0.0f) {
            mappedVerts++;
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    std::cout << "\n=== Relaxed Skinning Results (highly optimized) ===" << std::endl;
    std::cout << "  Time: " << elapsed << " ms" << std::endl;
    std::cout << "  Total high-res tets: " << numHighResTetsInt << std::endl;
    std::cout << "  Successfully mapped: " << successfullyMapped.load()
              << " (" << (100.0f * successfullyMapped.load() / numHighResTetsInt) << "%)" << std::endl;
    std::cout << "  Mapped vertices: " << mappedVerts << " / " << numHighResVerts
              << " (" << (100.0f * mappedVerts / numHighResVerts) << "%)" << std::endl;
    std::cout << "==========================================\n" << std::endl;
}



//==============================================================================
// プリセット・設定・統計ユーティリティ
//==============================================================================

//------------------------------------------------------------------------------
// ファクトリメソッド
//------------------------------------------------------------------------------
SoftBodyGPUDuo* SoftBodyGPUDuo::createFromPaths(
    const std::string& lowResPath,
    const std::string& highResPath,
    float edgeCompliance,
    float volCompliance,
    MeshPreset preset,
    SoftBodyGPUDuo* parent)
{
    MeshData lowResMesh = loadTetMesh(lowResPath);
    MeshData highResMesh = loadTetMesh(highResPath);

    SoftBodyGPUDuo* body = new SoftBodyGPUDuo(lowResMesh, highResMesh,
                                              edgeCompliance, volCompliance, preset);

    // VESSELで親が指定されている場合、Follow設定
    if (preset == MeshPreset::VESSEL && parent != nullptr) {
        body->setParentSoftBody(parent, getDefaultFollowParams());

        std::cout << "\n=== Follow Mode Enabled ===" << std::endl;
        std::cout << "  Vessel anchored to Parent" << std::endl;
        std::cout << "  Anchored vertices: " << body->getNumAnchoredVertices()
                  << " / " << body->numLowResParticles << std::endl;
    }

    return body;
}

//------------------------------------------------------------------------------
// プリセット取得
//------------------------------------------------------------------------------
SoftBodyGPUDuo::SimulationConfig SoftBodyGPUDuo::getPresetLiver() {
    SimulationConfig config;
    config.smoothFactor = 0.9f;
    config.motionDampingFactor = 0.95f;
    config.boundaryDampingFactor = 0.80f;
    return config;
}

SoftBodyGPUDuo::SimulationConfig SoftBodyGPUDuo::getPresetVessel() {
    SimulationConfig config;
    config.smoothFactor = 0.5f;
    config.motionDampingFactor = 0.0f;
    config.boundaryDampingFactor = 0.0f;
    return config;
}

SoftBodyGPUDuo::FollowParams SoftBodyGPUDuo::getDefaultFollowParams() {
    FollowParams params;
    params.barycentricEpsilon = 0.01f;
    params.maxAcceptableDist = 0.1f;
    params.lowQualityFactor = 2.0f;
    params.border = 0.05f;
    return params;
}

//------------------------------------------------------------------------------
// 設定適用
//------------------------------------------------------------------------------
void SoftBodyGPUDuo::applyConfig(const SimulationConfig& config) {
    // スムージング
    enableSmoothDisplay(config.smoothEnabled);
    setSmoothingParameters(config.smoothIterations, config.smoothFactor,
                          config.boundaryPreserve, config.boundaryIterations);

    // ダンピング
    lowRes_useBoundaryDamping = config.useBoundaryDamping;
    lowRes_motionDampingFactor = config.motionDampingFactor;
    lowRes_boundaryDampingFactor = config.boundaryDampingFactor;

    // 表示
    showLowHighTetMesh = config.showLowHighTetMesh;

    // ストレインリミット
    LowRes_enableStrainLimiting = config.enableStrainLimiting;
    edgeStrainSoftLimit = config.edgeStrainSoft;
    edgeStrainHardLimit = config.edgeStrainHard;
    edgeStrainMaxLimit = config.edgeStrainMax;
    volStrainSoftLimit = config.volStrainSoft;
    volStrainHardLimit = config.volStrainHard;
    volStrainMaxLimit = config.volStrainMax;

    // スキニング調整
    skinningAdjustParams.enabled = config.skinningEnabled;
    skinningAdjustParams.blendFactor = config.skinningBlendFactor;
    skinningAdjustParams.maxIterations = config.skinningMaxIterations;
}

//------------------------------------------------------------------------------
// 統計出力
//------------------------------------------------------------------------------
void SoftBodyGPUDuo::printCreationStats(const std::string& name) const {
    std::string title = name.empty() ? "SoftBody" : name;
    std::cout << "\n=== SoftBody (" << title << ") Created ===" << std::endl;
    std::cout << "  LowRes particles: " << numLowResParticles << std::endl;
    std::cout << "  LowRes tets: " << numLowTets << std::endl;
    std::cout << "  HighRes vertices: " << numHighResVerts << std::endl;
    std::cout << "  HighRes tets: " << numHighTets << std::endl;
}

void SoftBodyGPUDuo::printSkinningStats(const std::string& name) const {
    int mappedVerts = 0;
    int unmappedVerts = 0;
    int externalVerts = 0;

    for (size_t i = 0; i < numHighResVerts; i++) {
        int tetIdx = static_cast<int>(skinningInfoLowToHigh[8 * i]);
        if (tetIdx < 0) {
            unmappedVerts++;
        } else {
            mappedVerts++;
            float b0 = skinningInfoLowToHigh[8 * i + 1];
            float b1 = skinningInfoLowToHigh[8 * i + 2];
            float b2 = skinningInfoLowToHigh[8 * i + 3];
            float b3 = skinningInfoLowToHigh[8 * i + 4];
            if (b0 < -0.01f || b1 < -0.01f || b2 < -0.01f || b3 < -0.01f) {
                externalVerts++;
            }
        }
    }

    std::string title = name.empty() ? "Skinning Info" : name;
    std::cout << "\n=== Skinning Info (" << title << ") ===" << std::endl;
    std::cout << "  Mapped vertices: " << mappedVerts << " / " << numHighResVerts
              << " (" << (100.0f * mappedVerts / numHighResVerts) << "%)" << std::endl;
    std::cout << "  Unmapped vertices: " << unmappedVerts << std::endl;
    std::cout << "  External vertices (outside tet): " << externalVerts << std::endl;

    if (unmappedVerts > 0) {
        std::cout << "  WARNING: Some vertices have no skinning info!" << std::endl;
    }
    if (externalVerts > numHighResVerts * 0.1f) {
        std::cout << "  WARNING: Many vertices are outside their assigned tet!" << std::endl;
    }
}

void SoftBodyGPUDuo::printInvMassStats(const std::string& name) const {
    int zeroInvMass = 0;
    int nonZeroInvMass = 0;
    float minNonZero = FLT_MAX;
    float maxInvMass = 0.0f;

    for (size_t i = 0; i < numLowResParticles; i++) {
        float invMass = lowRes_invMasses[i];
        if (invMass == 0.0f) {
            zeroInvMass++;
        } else {
            nonZeroInvMass++;
            if (invMass < minNonZero) minNonZero = invMass;
            if (invMass > maxInvMass) maxInvMass = invMass;
        }
    }

    std::string title = name.empty() ? "InvMass Stats" : name;
    std::cout << "\n=== " << title << " ===" << std::endl;
    std::cout << "  Zero invMass (fixed): " << zeroInvMass << " / " << numLowResParticles
              << " (" << (100.0f * zeroInvMass / numLowResParticles) << "%)" << std::endl;
    std::cout << "  Non-zero invMass (movable): " << nonZeroInvMass << std::endl;

    if (nonZeroInvMass > 0) {
        std::cout << "  InvMass range: [" << minNonZero << ", " << maxInvMass << "]" << std::endl;
    }

    if (zeroInvMass > numLowResParticles * 0.5f) {
        std::cout << "  *** WARNING: More than 50% of particles are fixed! ***" << std::endl;
    }
}

int SoftBodyGPUDuo::computeEdgeValidity(const std::string& name) {
    size_t numEdges = lowRes_edgeIds.size() / 2;
    edgeValid.resize(numEdges);
    std::fill(edgeValid.begin(), edgeValid.end(), false);

    int validEdgeCount = 0;
    for (size_t i = 0; i < numEdges; i++) {
        int id0 = lowRes_edgeIds[2 * i];
        int id1 = lowRes_edgeIds[2 * i + 1];

        for (size_t t = 0; t < numLowTets; t++) {
            if (!lowRes_tetValid[t]) continue;

            bool hasId0 = false, hasId1 = false;
            for (int j = 0; j < 4; j++) {
                int vid = lowRes_tetIds[t * 4 + j];
                if (vid == id0) hasId0 = true;
                if (vid == id1) hasId1 = true;
            }

            if (hasId0 && hasId1) {
                edgeValid[i] = true;
                validEdgeCount++;
                break;
            }
        }
    }

    std::string title = name.empty() ? "EdgeValid Computed" : "EdgeValid Computed for " + name;
    std::cout << "\n=== " << title << " ===" << std::endl;
    std::cout << "  Valid edges: " << validEdgeCount << " / " << numEdges
              << " (" << (100.0f * validEdgeCount / numEdges) << "%)" << std::endl;

    return validEdgeCount;
}

// ============================================================================
// rebuildLowResMassesAndConstraints - LowRes四面体復元後の質量・拘束再構築
// UNDO操作後に呼び出す
// ============================================================================
// ============================================================================
// rebuildLowResMassesAndConstraints - LowRes四面体復元後の質量・拘束再構築
// UNDO操作後に呼び出す
// ============================================================================
void SoftBodyGPUDuo::rebuildLowResMassesAndConstraints() {
    std::cout << "\n[UNDO] === REBUILDING LOW-RES MASSES AND CONSTRAINTS ===" << std::endl;

    // 1. 現在の固定頂点を保存
    std::vector<int> currentPinnedVertices = lowRes_pinnedVertices;
    std::cout << "  Saved pinned vertices: " << currentPinnedVertices.size() << std::endl;

    // 2. 質量を有効な四面体から再計算
    std::fill(lowRes_invMasses.begin(), lowRes_invMasses.end(), 0.0f);

    int validTetCount = 0;
    for (size_t i = 0; i < numLowTets; i++) {
        if (!lowRes_tetValid[i]) continue;
        validTetCount++;

        float vol = lowResGetTetVolume(i);
        float pInvMass = vol > 0.0f ? 1.0f / (vol / 4000000.0f) : 1000000.0f;

        for (int j = 0; j < 4; j++) {
            lowRes_invMasses[lowRes_tetIds[4 * i + j]] += pInvMass;
        }
    }
    std::cout << "  Valid tets after rebuild: " << validTetCount << std::endl;

    // 3. 固定頂点の質量を0に戻す
    // 注意: ハンドルグループで解除された固定は復元しない（仕様）
    for (int pinnedId : currentPinnedVertices) {
        if (pinnedId >= 0 && static_cast<size_t>(pinnedId) < lowRes_invMasses.size()) {
            lowRes_invMasses[pinnedId] = 0.0f;
        }
    }

    // 4. 現在のハンドルグループの頂点も固定を維持
    int handleGroupFixedCount = 0;
    for (const auto& group : handleGroups) {
        for (int vertexId : group.vertices) {
            if (vertexId >= 0 && static_cast<size_t>(vertexId) < lowRes_invMasses.size()) {
                lowRes_invMasses[vertexId] = 0.0f;
                handleGroupFixedCount++;
            }
        }
    }
    std::cout << "  HandleGroup vertices kept fixed: " << handleGroupFixedCount << std::endl;

    // 5. 拘束（休止長さ・体積）を再設定
    restoreLowResInitialConstraints();

    // 6. エッジ有効性を更新
    updateLowResEdgeValidity();

    // 7. 統計出力
    int fixedCount = 0;
    for (size_t i = 0; i < lowRes_invMasses.size(); i++) {
        if (lowRes_invMasses[i] == 0.0f) fixedCount++;
    }
    std::cout << "  Fixed vertices after rebuild: " << fixedCount << std::endl;
    std::cout << "[UNDO] === REBUILD COMPLETED ===" << std::endl;
}

//=============================================================================
// Voronoi3D 四面体キャッシュのバインド
//=============================================================================
void SoftBodyGPUDuo::bindVoronoi3DTets(const VoxelSkeleton::VesselSegmentation& skeleton) {
    if (!skeleton.hasVoronoi3D()) {
        std::cout << "No Voronoi3D to bind (tets)" << std::endl;
        return;
    }

    const auto* voronoi = skeleton.getVoronoiSegmenter();
    if (!voronoi) {
        std::cout << "VoronoiSegmenter is null" << std::endl;
        return;
    }

    std::cout << "\n=== Binding Voronoi3D to Tetrahedra ===" << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();

    // 四面体→ブランチIDリストのマッピングを構築
    skeletonBinding.tetToVoronoi3DBranchIds.resize(numHighTets);

    // 休息状態の座標を使用
    const std::vector<float>* restPositions = &highRes_positions;
    if (!original_highRes_positions.empty() && original_highRes_positions.size() == highRes_positions.size()) {
        restPositions = &original_highRes_positions;
    }

#pragma omp parallel for schedule(dynamic, 1000)
    for (int tetIdx = 0; tetIdx < static_cast<int>(numHighTets); tetIdx++) {
        // 四面体の重心を計算
        glm::vec3 centroid(0.0f);
        for (int j = 0; j < 4; j++) {
            int vid = highResMeshData.tetIds[tetIdx * 4 + j];
            centroid.x += (*restPositions)[vid * 3];
            centroid.y += (*restPositions)[vid * 3 + 1];
            centroid.z += (*restPositions)[vid * 3 + 2];
        }
        centroid /= 4.0f;

        // ブランチIDリストを取得
        skeletonBinding.tetToVoronoi3DBranchIds[tetIdx] = voronoi->getBranchesAtPosition(centroid);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::cout << "  Voronoi3D bound to " << numHighTets << " tetrahedra in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count()
              << " ms" << std::endl;
}

//=============================================================================
// カット対象の四面体を取得（3モード共通）
//=============================================================================
std::vector<int> SoftBodyGPUDuo::getSelectedTetsForCut(
    const VoxelSkeleton::VesselSegmentation& skeleton) const
{
    std::vector<int> segmentTets;

    VoxelSkeleton::SegmentationMode mode = skeleton.getSegmentationMode();

    if (mode == VoxelSkeleton::SegmentationMode::OBJ) {
        //=================================================================
        // OBJモード（キャッシュ使用）
        //=================================================================
        int selectedOBJSeg = skeleton.getSelectedOBJSegment();
        if (selectedOBJSeg > 0) {
            std::cout << "Collecting tets for OBJ Segment: S" << selectedOBJSeg << std::endl;

            for (size_t i = 0; i < numHighTets; i++) {
                if (!highResTetValid[i]) continue;

                if (i < skeletonBinding.tetToOBJSegmentId.size()) {
                    int segId = skeletonBinding.tetToOBJSegmentId[i];
                    if (segId == selectedOBJSeg) {
                        segmentTets.push_back(static_cast<int>(i));
                    }
                }
            }
        }

    } else if (mode == VoxelSkeleton::SegmentationMode::Voronoi3D) {
        //=================================================================
        // Voronoi3Dモード（キャッシュ使用）
        //=================================================================
        const std::set<int>& selectedBranches = skeleton.getSelectedBranches();
        if (!selectedBranches.empty()) {
            std::cout << "Collecting tets for Voronoi3D Branches: ";
            for (int bid : selectedBranches) {
                std::cout << bid << " ";
            }
            std::cout << "(" << selectedBranches.size() << " branches selected)" << std::endl;

            // キャッシュがない場合は警告
            if (skeletonBinding.tetToVoronoi3DBranchIds.empty()) {
                std::cerr << "[Warning] Voronoi3D tet cache not built. Call bindVoronoi3DTets() first." << std::endl;
                return segmentTets;
            }

            int totalTets = 0;
            int sharedSkipped = 0;
            int exclusiveCut = 0;
            int noMatchCount = 0;

            std::map<int, int> cutBranchDistribution;
            std::map<int, int> skippedBranchDistribution;

            for (size_t i = 0; i < numHighTets; i++) {
                if (!highResTetValid[i]) continue;
                totalTets++;

                // ★★★ キャッシュから取得（毎回計算しない）★★★
                const std::vector<int>& tetBranchIds = skeletonBinding.tetToVoronoi3DBranchIds[i];

                if (totalTets <= 10) {
                    std::cout << "  [DEBUG] Tet " << i << ": branches = {";
                    for (size_t b = 0; b < tetBranchIds.size(); b++) {
                        std::cout << tetBranchIds[b];
                        if (b < tetBranchIds.size() - 1) std::cout << ", ";
                    }
                    std::cout << "} (count=" << tetBranchIds.size() << ")" << std::endl;
                }

                if (tetBranchIds.empty()) {
                    noMatchCount++;
                    continue;
                }

                // 全てのブランチが選択されているか確認
                bool isSubset = true;
                for (int bid : tetBranchIds) {
                    if (selectedBranches.count(bid) == 0) {
                        isSubset = false;
                        break;
                    }
                }

                if (isSubset) {
                    segmentTets.push_back(static_cast<int>(i));
                    exclusiveCut++;
                    cutBranchDistribution[tetBranchIds.size()]++;

                    if (exclusiveCut <= 5) {
                        std::cout << "  [CUT] Tet " << i << ": branches = {";
                        for (size_t b = 0; b < tetBranchIds.size(); b++) {
                            std::cout << tetBranchIds[b];
                            if (b < tetBranchIds.size() - 1) std::cout << ", ";
                        }
                        std::cout << "}" << std::endl;
                    }
                } else {
                    bool anySelected = false;
                    for (int bid : tetBranchIds) {
                        if (selectedBranches.count(bid) > 0) {
                            anySelected = true;
                            break;
                        }
                    }
                    if (anySelected) {
                        sharedSkipped++;
                        skippedBranchDistribution[tetBranchIds.size()]++;

                        if (sharedSkipped <= 5) {
                            std::cout << "  [SKIP] Tet " << i << ": branches = {";
                            for (size_t b = 0; b < tetBranchIds.size(); b++) {
                                std::cout << tetBranchIds[b];
                                if (b < tetBranchIds.size() - 1) std::cout << ", ";
                            }
                            std::cout << "} (not subset of selected)" << std::endl;
                        }
                    }
                }
            }

            std::cout << "\n=== Voronoi3D Cut Summary ===" << std::endl;
            std::cout << "  Total valid tets: " << totalTets << std::endl;
            std::cout << "  No branch match: " << noMatchCount << std::endl;
            std::cout << "  Exclusive to selected (will cut): " << exclusiveCut << std::endl;
            std::cout << "  Shared (skipped): " << sharedSkipped << std::endl;

            std::cout << "\n  Cut tet branch count distribution:" << std::endl;
            for (const auto& pair : cutBranchDistribution) {
                std::cout << "    " << pair.first << " branch(es): " << pair.second << " tets" << std::endl;
            }

            if (!skippedBranchDistribution.empty()) {
                std::cout << "\n  Skipped tet branch count distribution:" << std::endl;
                for (const auto& pair : skippedBranchDistribution) {
                    std::cout << "    " << pair.first << " branch(es): " << pair.second << " tets" << std::endl;
                }
            }
            std::cout << "============================\n" << std::endl;
        } else {
            std::cout << "No branches selected" << std::endl;
        }

    } else {
        //=================================================================
        // スケルトン距離モード（キャッシュ使用）
        //=================================================================
        const auto& selectedSegments = getSelectedSegments();
        if (!selectedSegments.empty()) {
            std::cout << "Collecting tets for " << selectedSegments.size() << " skeleton segments" << std::endl;

            for (size_t i = 0; i < numHighTets; i++) {
                if (!highResTetValid[i]) continue;

                if (i < skeletonBinding.tetToSegmentId.size()) {
                    int segId = skeletonBinding.tetToSegmentId[i];
                    if (selectedSegments.count(segId) > 0) {
                        segmentTets.push_back(static_cast<int>(i));
                    }
                }
            }
        }
    }

    std::cout << "Found " << segmentTets.size() << " tetrahedra to cut" << std::endl;
    return segmentTets;
}



//=============================================================================
// 低解像度四面体を有効化（復元）
//=============================================================================
void SoftBodyGPUDuo::validateLowResTetrahedra(const std::vector<int>& tetIndices) {
    // グラブ状態の完全保存
    bool grabWasActive = (lowRes_grabId >= 0);
    int savedGrabId = lowRes_grabId;
    glm::vec3 savedGrabPosition;
    glm::vec3 savedGrabOffset = lowRes_grabOffset;
    std::vector<int> savedActiveParticles = lowRes_activeParticles;

    // 固定されている頂点を記録（invMass == 0の頂点）
    std::set<int> fixedVertices;
    for (size_t i = 0; i < numLowResParticles; i++) {
        if (lowRes_invMasses[i] == 0.0f) {
            fixedVertices.insert(i);
        }
    }

    if (grabWasActive) {
        savedGrabPosition = glm::vec3(
            lowRes_positions[lowRes_grabId * 3],
            lowRes_positions[lowRes_grabId * 3 + 1],
            lowRes_positions[lowRes_grabId * 3 + 2]
            );
    }

    // 四面体を有効化
    for (int idx : tetIndices) {
        if (idx >= 0 && idx < static_cast<int>(lowRes_tetValid.size())) {
            lowRes_tetValid[idx] = true;
        }
    }

    // 質量の再計算（有効な四面体のみ）
    std::fill(lowRes_invMasses.begin(), lowRes_invMasses.end(), 0.0f);
    for (size_t i = 0; i < numLowTets; i++) {
        if (!lowRes_tetValid[i]) continue;

        float vol = lowResGetTetVolume(i);
        float pInvMass = vol > 0.0f ? 1.0f / (vol / 4000000.0f) : 1000000.0f;

        for (int j = 0; j < 4; j++) {
            int vid = lowRes_tetIds[4 * i + j];
            // 固定頂点でなければ質量を追加
            if (fixedVertices.find(vid) == fixedVertices.end()) {
                lowRes_invMasses[vid] += pInvMass;
            }
        }
    }

    // 固定頂点の質量を0に戻す（重要！）
    for (int vid : fixedVertices) {
        lowRes_invMasses[vid] = 0.0f;
    }

    // 初期制約を復元
    restoreLowResInitialConstraints();

    // グラブ状態の完全復元
    if (grabWasActive) {
        lowRes_grabId = savedGrabId;
        lowRes_grabOffset = savedGrabOffset;
        lowRes_activeParticles = savedActiveParticles;
    }
    // ★ これを追加
    invalidateSurfaceCache();
    // ★★★ XPBD隣接キャッシュを無効化（Undo後に再構築される）★★★
    lowResNeighborsCacheBuilt_ = false;

    std::cout << "Validated " << tetIndices.size() << " low-res tetrahedra" << std::endl;
}

//=============================================================================
// 全ての低解像度四面体を有効化
//=============================================================================
void SoftBodyGPUDuo::validateAllLowResTetrahedra() {
    std::vector<int> allIndices;
    for (size_t i = 0; i < numLowTets; i++) {
        allIndices.push_back(static_cast<int>(i));
    }
    validateLowResTetrahedra(allIndices);
}

//=============================================================================
// 低解像度四面体を有効化（復元）- 質量情報付き
//=============================================================================
void SoftBodyGPUDuo::validateLowResTetrahedraWithMasses(
    const std::vector<int>& tetIndices,
    const std::vector<float>& originalInvMasses)
{
    // グラブ状態の保存
    bool grabWasActive = (lowRes_grabId >= 0);
    int savedGrabId = lowRes_grabId;
    glm::vec3 savedGrabOffset = lowRes_grabOffset;
    std::vector<int> savedActiveParticles = lowRes_activeParticles;

    // 四面体を有効化
    for (int idx : tetIndices) {
        if (idx >= 0 && idx < static_cast<int>(lowRes_tetValid.size())) {
            lowRes_tetValid[idx] = true;
        }
    }

    // ★★★ 保存しておいた質量を直接復元 ★★★
    if (originalInvMasses.size() == lowRes_invMasses.size()) {
        lowRes_invMasses = originalInvMasses;
        std::cout << "  Restored original inverse masses" << std::endl;
    } else {
        // フォールバック：再計算（ただし正確ではない可能性あり）
        std::cerr << "  Warning: Mass backup size mismatch, recalculating..." << std::endl;

        // 初期化時に固定されていた頂点を特定するため
        // 有効な四面体に一度も属さない頂点を探す
        std::vector<bool> vertexUsed(numLowResParticles, false);
        for (size_t i = 0; i < numLowTets; i++) {
            if (!lowRes_tetValid[i]) continue;
            for (int j = 0; j < 4; j++) {
                int vid = lowRes_tetIds[4 * i + j];
                vertexUsed[vid] = true;
            }
        }

        // 質量を再計算
        std::fill(lowRes_invMasses.begin(), lowRes_invMasses.end(), 0.0f);
        for (size_t i = 0; i < numLowTets; i++) {
            if (!lowRes_tetValid[i]) continue;

            float vol = lowResGetTetVolume(i);
            float pInvMass = vol > 0.0f ? 1.0f / (vol / 4000000.0f) : 1000000.0f;

            for (int j = 0; j < 4; j++) {
                int vid = lowRes_tetIds[4 * i + j];
                lowRes_invMasses[vid] += pInvMass;
            }
        }
    }

    // 初期制約を復元
    restoreLowResInitialConstraints();

    // グラブ状態の復元
    if (grabWasActive) {
        lowRes_grabId = savedGrabId;
        lowRes_grabOffset = savedGrabOffset;
        lowRes_activeParticles = savedActiveParticles;
    }
    invalidateSurfaceCache();  // ★ 追加
    // ★★★ XPBD隣接キャッシュを無効化（Undo後に再構築される）★★★
    lowResNeighborsCacheBuilt_ = false;
    std::cout << "Validated " << tetIndices.size() << " low-res tetrahedra with masses" << std::endl;
}

//=============================================================================
// CutSegmentMode用メソッド - カッター位置でのセグメント2色オーバーレイ
//=============================================================================

void SoftBodyGPUDuo::applyCutSegmentOverlaySkeleton(
    const std::set<int>& selectedSegments,
    const VoxelSkeleton::VesselSegmentation& skeleton,
    const glm::vec4& baseColor,
    const glm::vec4& highlightColor)
{
    // スケルトンがバインドされていなければバインド
    if (!skeletonBinding.isBound) {
        std::cout << "[CutSegmentOverlay] Warning: Skeleton not bound, attempting to bind..." << std::endl;
        return;
    }

    // 色配列をリサイズ
    objSegmentVertexColors_.resize(numHighResVerts * 3, 0.5f);

    int highlightedCount = 0;

    for (size_t vid = 0; vid < numHighResVerts; vid++) {
        // この頂点のセグメントIDを取得
        int segId = -1;
        if (vid < skeletonBinding.vertexToSegmentId.size()) {
            segId = skeletonBinding.vertexToSegmentId[vid];
        }

        glm::vec3 color;

        // 選択されたセグメントに含まれているかチェック
        if (segId >= 0 && selectedSegments.count(segId) > 0) {
            // 選択領域：黄色（ハイライト）
            color = glm::vec3(highlightColor.r, highlightColor.g, highlightColor.b);
            highlightedCount++;
        } else {
            // 非選択領域：赤っぽい色（ベース）
            color = glm::vec3(baseColor.r, baseColor.g, baseColor.b);
        }

        objSegmentVertexColors_[vid * 3] = color.r;
        objSegmentVertexColors_[vid * 3 + 1] = color.g;
        objSegmentVertexColors_[vid * 3 + 2] = color.b;
    }

    useOBJSegmentColors_ = true;
    std::cout << "[CutSegmentOverlay-Skeleton] Highlighted " << highlightedCount
              << " / " << numHighResVerts << " vertices in "
              << selectedSegments.size() << " segments" << std::endl;
}

void SoftBodyGPUDuo::applyCutSegmentOverlayVoronoi(
    const std::vector<int>& selectedBranches,
    const VoxelSkeleton::VesselSegmentation& skeleton,
    const glm::vec4& baseColor,
    const glm::vec4& highlightColor)
{
    // Voronoi3Dがバインドされていなければバインド
    if (!skeletonBinding.voronoi3DBound) {
        std::cout << "[CutSegmentOverlay] Warning: Voronoi3D not bound, attempting to bind..." << std::endl;
        bindVoronoi3D(skeleton);
    }

    if (!skeleton.hasVoronoi3D()) {
        std::cout << "[CutSegmentOverlay] Warning: Voronoi3D not available" << std::endl;
        return;
    }

    // 選択されたブランチIDをsetに変換（高速検索用）
    std::set<int> selectedSet(selectedBranches.begin(), selectedBranches.end());

    // 色配列をリサイズ
    objSegmentVertexColors_.resize(numHighResVerts * 3, 0.5f);

    int highlightedCount = 0;

    for (size_t vid = 0; vid < numHighResVerts; vid++) {
        const std::vector<int>& branchIds = skeletonBinding.vertexToVoronoi3DBranchIds[vid];

        glm::vec3 color;

        // 頂点のブランチIDが選択ブランチの「部分集合」かチェック
        bool isSubset = !branchIds.empty();
        for (int bid : branchIds) {
            if (selectedSet.count(bid) == 0) {
                isSubset = false;
                break;
            }
        }

        if (isSubset) {
            // 選択領域：黄色（ハイライト）
            color = glm::vec3(highlightColor.r, highlightColor.g, highlightColor.b);
            highlightedCount++;
        } else {
            // 非選択領域：赤っぽい色（ベース）
            color = glm::vec3(baseColor.r, baseColor.g, baseColor.b);
        }

        objSegmentVertexColors_[vid * 3] = color.r;
        objSegmentVertexColors_[vid * 3 + 1] = color.g;
        objSegmentVertexColors_[vid * 3 + 2] = color.b;
    }

    useOBJSegmentColors_ = true;
    std::cout << "[CutSegmentOverlay-Voronoi] Highlighted " << highlightedCount
              << " / " << numHighResVerts << " vertices in "
              << selectedBranches.size() << " branches" << std::endl;
}

void SoftBodyGPUDuo::resetToDefaultColors() {
    // OBJセグメント色の使用を無効化
    useOBJSegmentColors_ = false;

    // 色配列をクリア
    objSegmentVertexColors_.clear();

    std::cout << "[CutSegmentOverlay] Reset to default colors" << std::endl;
}


void SoftBodyGPUDuo::captureOriginalVolumes() {
    originalTotalVolume_ = calculateTotalVolume();
    originalSegmentVolumes_ = calculateAllSegmentVolumes(false);  // Skeleton

    std::cout << "[SoftBody] Original volumes captured: "
              << (originalTotalVolume_ / 1000.0f) << " cm³, "
              << originalSegmentVolumes_.size() << " segments" << std::endl;
}

float SoftBodyGPUDuo::getOriginalSegmentVolume(int segId) const {
    auto it = originalSegmentVolumes_.find(segId);
    return (it != originalSegmentVolumes_.end()) ? it->second : 0.0f;
}



//==============================================================================
// SoftBodyGPUDuo.cpp に追加するコード
// 【追加場所】ファイルの末尾
//
// ★★★ 重要 ★★★
// 既存のVisual-only関連のコード（以下の関数）があれば全て削除してから追加：
// - SoftBodyGPUDuo::SoftBodyGPUDuo(const MeshData& visMesh, ...)
// - SoftBodyGPUDuo::createVisualOnly(...)
// - SoftBodyGPUDuo::computeSkinningToParentLowRes(...)
// - SoftBodyGPUDuo::setupVisualOnlyMesh()
// - SoftBodyGPUDuo::updateVisualOnlyFromParent()
// - SoftBodyGPUDuo::computeVisualOnlyNormals()
// - SoftBodyGPUDuo::drawVisualOnlyMesh(...)
// - SoftBodyGPUDuo::buildAdjacencyList(...)
// - SoftBodyGPUDuo::applySmoothingForVertex(...)
// - SoftBodyGPUDuo::correctAbnormalVertices(...)
// - SoftBodyGPUDuo::applyVisualMeshCorrection()
// - SoftBodyGPUDuo::applyDisplacementPropagation(...)
//==============================================================================

//=============================================================================
// VISUAL-ONLY MODE IMPLEMENTATION
//=============================================================================

SoftBodyGPUDuo::SoftBodyGPUDuo(const MeshData& visMesh,
                               SoftBodyGPUDuo* parent,
                               const FollowParams& params)
    : isVisualOnlyMode(true)
    , parentSoftBody(parent)
    , followParams(params)
    , followMode(FOLLOW_PARENT_SOFTBODY)
    , numLowResParticles(0)
    , numLowTets(0)
    , numHighTets(0)
    , numHighResVerts(0)
    , numVisVerts(0)
    , visVAO(0), visVBO(0), visEBO(0), visNormalVBO(0)
{
    std::cout << "\n=== SoftBodyGPUDuo (Visual Only Mode) ===" << std::endl;

    if (!parent) {
        std::cerr << "Error: Parent required for Visual Only mode!" << std::endl;
        return;
    }

    numVisVerts = visMesh.verts.size() / 3;
    vis_positions = visMesh.verts;
    original_vis_positions = visMesh.verts;
    visSurfaceTriIds = visMesh.tetSurfaceTriIds;
    vis_normals.resize(numVisVerts * 3, 0.0f);

    std::cout << "Vertices: " << numVisVerts
              << ", Triangles: " << visSurfaceTriIds.size() / 3 << std::endl;

    skinningToParentLowRes.resize(4 * numVisVerts, -1.0f);
    computeSkinningToParentLowRes(visMesh.verts);

    buildAdjacencyList();
    setupVisualOnlyMesh();
    updateVisualOnlyFromParent();

    std::cout << "=== Visual Only Complete ===" << std::endl;
}

SoftBodyGPUDuo* SoftBodyGPUDuo::createVisualOnly(const std::string& visObjPath,
                                                  SoftBodyGPUDuo* parent,
                                                  const FollowParams& params)
{
    if (!parent) {
        std::cerr << "Error: createVisualOnly requires parent" << std::endl;
        return nullptr;
    }

    std::cout << "Loading visual mesh: " << visObjPath << std::endl;
    MeshData visMesh = ReadVertexAndFace(visObjPath);

    if (visMesh.verts.empty()) {
        std::cerr << "Error: Failed to load " << visObjPath << std::endl;
        return nullptr;
    }

    return new SoftBodyGPUDuo(visMesh, parent, params);
}

void SoftBodyGPUDuo::computeSkinningToParentLowRes(const std::vector<float>& visVerts) {
    if (!parentSoftBody) return;

    const std::vector<float>& parentPos = parentSoftBody->getLowResPositions();
    const std::vector<int>& parentTetIds = parentSoftBody->lowRes_tetIds;
    size_t parentNumTets = parentSoftBody->numLowTets;

    glm::vec3 pMin(FLT_MAX), pMax(-FLT_MAX);
    for (size_t i = 0; i < parentPos.size(); i += 3) {
        pMin = glm::min(pMin, glm::vec3(parentPos[i], parentPos[i+1], parentPos[i+2]));
        pMax = glm::max(pMax, glm::vec3(parentPos[i], parentPos[i+1], parentPos[i+2]));
    }
    float spacing = glm::length(pMax - pMin);

    Hash hash(spacing, numVisVerts);
    hash.create(visVerts);

    std::vector<int> quality(numVisVerts, 0);
    std::vector<float> minDist(numVisVerts, FLT_MAX);
    std::vector<float> tetCenter(3), mat(9), bary(4);

    for (size_t t = 0; t < parentNumTets; t++) {
        std::fill(tetCenter.begin(), tetCenter.end(), 0.0f);
        for (int j = 0; j < 4; j++) {
            int vid = parentTetIds[4*t + j];
            tetCenter[0] += parentPos[vid*3]     * 0.25f;
            tetCenter[1] += parentPos[vid*3 + 1] * 0.25f;
            tetCenter[2] += parentPos[vid*3 + 2] * 0.25f;
        }

        float rMax = 0.0f;
        for (int j = 0; j < 4; j++) {
            int vid = parentTetIds[4*t + j];
            float dx = parentPos[vid*3]     - tetCenter[0];
            float dy = parentPos[vid*3 + 1] - tetCenter[1];
            float dz = parentPos[vid*3 + 2] - tetCenter[2];
            rMax = std::max(rMax, std::sqrt(dx*dx + dy*dy + dz*dz));
        }
        rMax += followParams.border;

        hash.query(tetCenter, 0, rMax);
        if (hash.querySize == 0) continue;

        int id0 = parentTetIds[4*t], id1 = parentTetIds[4*t+1];
        int id2 = parentTetIds[4*t+2], id3 = parentTetIds[4*t+3];

        VectorMath::vecSetDiff(mat, 0, parentPos, id0, parentPos, id3);
        VectorMath::vecSetDiff(mat, 1, parentPos, id1, parentPos, id3);
        VectorMath::vecSetDiff(mat, 2, parentPos, id2, parentPos, id3);
        VectorMath::matSetInverse(mat);

        for (int j = 0; j < hash.querySize; j++) {
            int id = hash.queryIds[j];
            if (quality[id] == 3) continue;

            VectorMath::vecSetDiff(bary, 0, visVerts, id, parentPos, id3);
            VectorMath::matSetMult(mat, bary, 0, bary, 0);
            bary[3] = 1.0f - bary[0] - bary[1] - bary[2];

            bool allPos = true;
            float minNeg = 0.0f;
            for (int k = 0; k < 4; k++) {
                if (bary[k] < -followParams.barycentricEpsilon) {
                    allPos = false;
                    minNeg = std::min(minNeg, bary[k]);
                }
            }

            float dist = -minNeg;
            int q = allPos ? 3 : (dist < followParams.maxAcceptableDist ? 2 :
                    (dist < followParams.maxAcceptableDist * followParams.lowQualityFactor ? 1 : 0));

            if (q > quality[id] || (q == quality[id] && dist < minDist[id])) {
                quality[id] = q;
                minDist[id] = dist;
                skinningToParentLowRes[4*id]   = static_cast<float>(t);
                skinningToParentLowRes[4*id+1] = bary[0];
                skinningToParentLowRes[4*id+2] = bary[1];
                skinningToParentLowRes[4*id+3] = bary[2];
            }
        }
    }

    int cnt[4] = {0};
    for (size_t i = 0; i < numVisVerts; i++) cnt[quality[i]]++;
    std::cout << "Skinning: High=" << cnt[3] << " Med=" << cnt[2]
              << " Low=" << cnt[1] << " None=" << cnt[0] << std::endl;
}

void SoftBodyGPUDuo::setupVisualOnlyMesh() {
    glGenVertexArrays(1, &visVAO);
    glBindVertexArray(visVAO);

    glGenBuffers(1, &visVBO);
    glBindBuffer(GL_ARRAY_BUFFER, visVBO);
    glBufferData(GL_ARRAY_BUFFER, vis_positions.size() * sizeof(float),
                 vis_positions.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &visNormalVBO);
    glBindBuffer(GL_ARRAY_BUFFER, visNormalVBO);
    glBufferData(GL_ARRAY_BUFFER, vis_normals.size() * sizeof(float),
                 vis_normals.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(1);

    glGenBuffers(1, &visEBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, visEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, visSurfaceTriIds.size() * sizeof(int),
                 visSurfaceTriIds.data(), GL_STATIC_DRAW);

    glBindVertexArray(0);
}

void SoftBodyGPUDuo::buildAdjacencyList() {
    cachedAdjacencyList.clear();
    cachedAdjacencyList.resize(numVisVerts);

    for (size_t i = 0; i < visSurfaceTriIds.size(); i += 3) {
        int id0 = visSurfaceTriIds[i];
        int id1 = visSurfaceTriIds[i+1];
        int id2 = visSurfaceTriIds[i+2];

        if (id0 >= 0 && id0 < (int)numVisVerts &&
            id1 >= 0 && id1 < (int)numVisVerts &&
            id2 >= 0 && id2 < (int)numVisVerts) {
            cachedAdjacencyList[id0].push_back(id1);
            cachedAdjacencyList[id0].push_back(id2);
            cachedAdjacencyList[id1].push_back(id0);
            cachedAdjacencyList[id1].push_back(id2);
            cachedAdjacencyList[id2].push_back(id0);
            cachedAdjacencyList[id2].push_back(id1);
        }
    }

    for (auto& neighbors : cachedAdjacencyList) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }
}

void SoftBodyGPUDuo::drawVisualOnlyMesh(ShaderProgram& shader) {
    if (!isVisualOnlyMode || visVAO == 0) return;

    shader.use();
    glBindVertexArray(visVAO);
    glDrawElements(GL_TRIANGLES, visSurfaceTriIds.size(), GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
}

void SoftBodyGPUDuo::drawVisualOnlyMesh(ShaderProgram& shader, const glm::vec4& color) {
    if (!isVisualOnlyMode || visVAO == 0) return;

    shader.use();
    shader.setUniform("objectColor", glm::vec3(color.r, color.g, color.b));  // ★vec3
    shader.setUniform("objectAlpha", color.a);  // ★float
    shader.setUniform("useVertexColor", false); // ★頂点カラーを使わない

    glBindVertexArray(visVAO);
    glDrawElements(GL_TRIANGLES, visSurfaceTriIds.size(), GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
}

//==============================================================================
// 追加終了
//==============================================================================
//=============================================================================
// 並列化版 Visual-Only 更新関数
// 以下の3つの関数を既存のものと置き換えてください
//=============================================================================

void SoftBodyGPUDuo::updateVisualOnlyFromParent() {
    if (!isVisualOnlyMode || !parentSoftBody) return;

    const std::vector<float>& parentPos = parentSoftBody->getLowResPositions();
    const std::vector<int>& parentTetIds = parentSoftBody->lowRes_tetIds;
    const std::vector<bool>& parentTetValid = parentSoftBody->lowRes_tetValid;

    std::vector<bool> needsCorrection(numVisVerts, false);

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(numVisVerts); i++) {
        int tetIdx = static_cast<int>(skinningToParentLowRes[4*i]);

        if (tetIdx < 0 ||
            tetIdx >= static_cast<int>(parentTetValid.size()) ||
            !parentTetValid[tetIdx]) {
            needsCorrection[i] = true;
            continue;
        }

        float b0 = skinningToParentLowRes[4*i + 1];
        float b1 = skinningToParentLowRes[4*i + 2];
        float b2 = skinningToParentLowRes[4*i + 3];
        float b3 = 1.0f - b0 - b1 - b2;

        int id0 = parentTetIds[4*tetIdx];
        int id1 = parentTetIds[4*tetIdx + 1];
        int id2 = parentTetIds[4*tetIdx + 2];
        int id3 = parentTetIds[4*tetIdx + 3];

        vis_positions[3*i]   = b0*parentPos[id0*3]   + b1*parentPos[id1*3]   +
                               b2*parentPos[id2*3]   + b3*parentPos[id3*3];
        vis_positions[3*i+1] = b0*parentPos[id0*3+1] + b1*parentPos[id1*3+1] +
                               b2*parentPos[id2*3+1] + b3*parentPos[id3*3+1];
        vis_positions[3*i+2] = b0*parentPos[id0*3+2] + b1*parentPos[id1*3+2] +
                               b2*parentPos[id2*3+2] + b3*parentPos[id3*3+2];
    }

    applySmoothingCorrection(needsCorrection);

    glBindBuffer(GL_ARRAY_BUFFER, visVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, vis_positions.size() * sizeof(float),
                    vis_positions.data());

    computeVisualOnlyNormals();
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void SoftBodyGPUDuo::computeVisualOnlyNormals() {
    vis_normals.assign(numVisVerts * 3, 0.0f);

    // 三角形ループ（atomic加算で並列化）
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(visSurfaceTriIds.size()); i += 3) {
        int id0 = visSurfaceTriIds[i];
        int id1 = visSurfaceTriIds[i+1];
        int id2 = visSurfaceTriIds[i+2];

        if (id0 < 0 || id0 >= static_cast<int>(numVisVerts) ||
            id1 < 0 || id1 >= static_cast<int>(numVisVerts) ||
            id2 < 0 || id2 >= static_cast<int>(numVisVerts)) continue;

        glm::vec3 v0(vis_positions[3*id0], vis_positions[3*id0+1], vis_positions[3*id0+2]);
        glm::vec3 v1(vis_positions[3*id1], vis_positions[3*id1+1], vis_positions[3*id1+2]);
        glm::vec3 v2(vis_positions[3*id2], vis_positions[3*id2+1], vis_positions[3*id2+2]);

        glm::vec3 n = glm::cross(v1 - v0, v2 - v0);

        #pragma omp atomic
        vis_normals[3*id0]   += n.x;
        #pragma omp atomic
        vis_normals[3*id0+1] += n.y;
        #pragma omp atomic
        vis_normals[3*id0+2] += n.z;

        #pragma omp atomic
        vis_normals[3*id1]   += n.x;
        #pragma omp atomic
        vis_normals[3*id1+1] += n.y;
        #pragma omp atomic
        vis_normals[3*id1+2] += n.z;

        #pragma omp atomic
        vis_normals[3*id2]   += n.x;
        #pragma omp atomic
        vis_normals[3*id2+1] += n.y;
        #pragma omp atomic
        vis_normals[3*id2+2] += n.z;
    }

    // 正規化（完全並列）
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(numVisVerts); i++) {
        glm::vec3 n(vis_normals[3*i], vis_normals[3*i+1], vis_normals[3*i+2]);
        float len = glm::length(n);
        if (len > 0.0001f) {
            n /= len;
            vis_normals[3*i]   = n.x;
            vis_normals[3*i+1] = n.y;
            vis_normals[3*i+2] = n.z;
        }
    }

    glBindBuffer(GL_ARRAY_BUFFER, visNormalVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, vis_normals.size() * sizeof(float),
                    vis_normals.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void SoftBodyGPUDuo::applySmoothingCorrection(const std::vector<bool>& needsCorrection) {
    // 有効な頂点の変位を計算
    std::vector<glm::vec3> displacement(numVisVerts, glm::vec3(0.0f));
    std::vector<int> validIndices;

    for (size_t i = 0; i < numVisVerts; i++) {
        if (!needsCorrection[i]) {
            displacement[i] = glm::vec3(
                vis_positions[3*i]   - original_vis_positions[3*i],
                vis_positions[3*i+1] - original_vis_positions[3*i+1],
                vis_positions[3*i+2] - original_vis_positions[3*i+2]
            );
            validIndices.push_back(i);
        }
    }

    if (validIndices.empty()) return;

    // 補正対象頂点に変位を伝播（全有効頂点から距離ベース）
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(numVisVerts); i++) {
        if (!needsCorrection[i]) continue;

        glm::vec3 myOrigPos(original_vis_positions[3*i],
                           original_vis_positions[3*i+1],
                           original_vis_positions[3*i+2]);

        glm::vec3 avgDisp(0.0f);
        float totalWeight = 0.0f;

        // 全有効頂点からの距離加重平均
        for (int j : validIndices) {
            glm::vec3 otherOrigPos(original_vis_positions[3*j],
                                   original_vis_positions[3*j+1],
                                   original_vis_positions[3*j+2]);

            float dist = glm::distance(myOrigPos, otherOrigPos);
            float weight = 1.0f / (dist * dist + 0.0001f);

            avgDisp += weight * displacement[j];
            totalWeight += weight;
        }

        if (totalWeight > 0.0f) {
            avgDisp /= totalWeight;
            vis_positions[3*i]   = original_vis_positions[3*i]   + avgDisp.x;
            vis_positions[3*i+1] = original_vis_positions[3*i+1] + avgDisp.y;
            vis_positions[3*i+2] = original_vis_positions[3*i+2] + avgDisp.z;
        }
    }
}


// ============================================================
// 有効サーフェスキャッシュの再構築
// ============================================================
void SoftBodyGPUDuo::rebuildValidSurfaceCache() {
    if (!surfaceCacheDirty_) return;

    validSurfaceTriIds_.clear();
    validSurfaceVertices_.clear();

    // 頂点→テトラのマッピングを構築
    std::vector<std::vector<int>> vertexToTets(numLowResParticles);
    for (size_t t = 0; t < numLowTets; t++) {
        for (int j = 0; j < 4; j++) {
            int vIdx = lowRes_tetIds[t * 4 + j];
            if (vIdx >= 0 && vIdx < static_cast<int>(numLowResParticles)) {
                vertexToTets[vIdx].push_back(static_cast<int>(t));
            }
        }
    }

    // 有効な表面三角形と頂点を収集
    std::set<int> validVertSet;
    const auto& surfaceTriIds = lowResMeshData.tetSurfaceTriIds;

    for (size_t i = 0; i + 2 < surfaceTriIds.size(); i += 3) {
        int idx0 = surfaceTriIds[i];
        int idx1 = surfaceTriIds[i + 1];
        int idx2 = surfaceTriIds[i + 2];

        // 3頂点すべてが少なくとも1つの有効テトラに属するかチェック
        bool allValid = true;
        for (int vIdx : {idx0, idx1, idx2}) {
            if (vIdx < 0 || vIdx >= static_cast<int>(numLowResParticles)) {
                allValid = false;
                break;
            }

            bool hasValidTet = false;
            for (int tetIdx : vertexToTets[vIdx]) {
                if (tetIdx >= 0 && tetIdx < static_cast<int>(lowRes_tetValid.size())
                    && lowRes_tetValid[tetIdx]) {
                    hasValidTet = true;
                    break;
                }
            }
            if (!hasValidTet) {
                allValid = false;
                break;
            }
        }

        if (allValid) {
            validSurfaceTriIds_.push_back(idx0);
            validSurfaceTriIds_.push_back(idx1);
            validSurfaceTriIds_.push_back(idx2);
            validVertSet.insert(idx0);
            validVertSet.insert(idx1);
            validVertSet.insert(idx2);
        }
    }

    validSurfaceVertices_.assign(validVertSet.begin(), validVertSet.end());
    surfaceCacheDirty_ = false;

    std::cout << "[SurfaceCache] Rebuilt: "
              << validSurfaceTriIds_.size() / 3 << " valid triangles / "
              << surfaceTriIds.size() / 3 << " total, "
              << validSurfaceVertices_.size() << " valid vertices" << std::endl;
}
const std::vector<int>& SoftBodyGPUDuo::getValidSurfaceTriIds() {
    if (surfaceCacheDirty_) {
        rebuildValidSurfaceCache();
    }
    return validSurfaceTriIds_;
}

const std::vector<int>& SoftBodyGPUDuo::getValidSurfaceVertices() {
    if (surfaceCacheDirty_) {
        rebuildValidSurfaceCache();
    }
    return validSurfaceVertices_;
}
// 最近傍表面頂点を検索（有効なテトラに属する頂点のみ）
int SoftBodyGPUDuo::findClosestSurfaceVertex(const glm::vec3& position) {
    // キャッシュが古ければ更新
    if (surfaceCacheDirty_) {
        rebuildValidSurfaceCache();
    }

    float minD2 = std::numeric_limits<float>::max();
    int closestId = -1;

    // 有効な表面頂点のみから探索
    for (int idx : validSurfaceVertices_) {
        if (idx < 0 || idx >= static_cast<int>(numLowResParticles)) continue;

        glm::vec3 particlePos(lowRes_positions[idx * 3],
                              lowRes_positions[idx * 3 + 1],
                              lowRes_positions[idx * 3 + 2]);
        glm::vec3 diff = particlePos - position;
        float d2 = glm::dot(diff, diff);

        if (d2 < minD2) {
            minD2 = d2;
            closestId = idx;
        }
    }

    return closestId;
}






// =====================================================
// SoftBodyGPUDuo.cpp に追加
// =====================================================

#include <set>
#include <algorithm>
#include <iomanip>

// Get VALID high-res tets only (filtered by highResTetValid)
std::set<int> SoftBodyGPUDuo::getValidHighResTetsFromLowResTet(int lowResTetIdx) {
    std::set<int> allHighTets = getHighResTetsFromLowResTet(lowResTetIdx);
    std::set<int> validHighTets;

    for (int highTetIdx : allHighTets) {
        // Check if this high-res tet is valid
        if (highResTetValid.empty() || highResTetValid[highTetIdx]) {
            validHighTets.insert(highTetIdx);
        }
    }

    return validHighTets;
}

std::vector<SoftBodyGPUDuo::UnstableTetInfo> SoftBodyGPUDuo::detectUnstableTetrahedra(float velocityThreshold) const {
    std::vector<UnstableTetInfo> unstableTets;

    for (size_t tetIdx = 0; tetIdx < numLowTets; tetIdx++) {
        // Get 4 vertex indices of the tetrahedron
        int v0 = lowRes_tetIds[tetIdx * 4 + 0];
        int v1 = lowRes_tetIds[tetIdx * 4 + 1];
        int v2 = lowRes_tetIds[tetIdx * 4 + 2];
        int v3 = lowRes_tetIds[tetIdx * 4 + 3];

        // Get velocity magnitude of each vertex
        auto getMag = [&](int v) {
            return std::sqrt(
                lowRes_velocities[v*3] * lowRes_velocities[v*3] +
                lowRes_velocities[v*3+1] * lowRes_velocities[v*3+1] +
                lowRes_velocities[v*3+2] * lowRes_velocities[v*3+2]);
        };

        float mag0 = getMag(v0);
        float mag1 = getMag(v1);
        float mag2 = getMag(v2);
        float mag3 = getMag(v3);

        float avgMag = (mag0 + mag1 + mag2 + mag3) / 4.0f;
        float maxMag = std::max({mag0, mag1, mag2, mag3});

        // If exceeds threshold, mark as unstable
        if (maxMag > velocityThreshold) {
            UnstableTetInfo info;
            info.lowResTetIdx = static_cast<int>(tetIdx);
            info.avgVelocityMag = avgMag;
            info.maxVelocityMag = maxMag;
            info.isValid = lowRes_tetValid.empty() || lowRes_tetValid[tetIdx];

            // Get ALL high-res tets from mapping (including invalid ones)
            auto allHighResTets = const_cast<SoftBodyGPUDuo*>(this)->getHighResTetsFromLowResTet(static_cast<int>(tetIdx));
            info.skinnedHighResTetCount = static_cast<int>(allHighResTets.size());

            // Get only VALID high-res tets
            auto validHighResTets = const_cast<SoftBodyGPUDuo*>(this)->getValidHighResTetsFromLowResTet(static_cast<int>(tetIdx));
            info.validHighResTetCount = static_cast<int>(validHighResTets.size());

            unstableTets.push_back(info);
        }
    }

    // Sort by max velocity (descending)
    std::sort(unstableTets.begin(), unstableTets.end(),
              [](const UnstableTetInfo& a, const UnstableTetInfo& b) {
                  return a.maxVelocityMag > b.maxVelocityMag;
              });

    return unstableTets;
}

void SoftBodyGPUDuo::printUnstableTetrahedraDebugInfo(float velocityThreshold) {
    auto unstableTets = detectUnstableTetrahedra(velocityThreshold);

    // Count statistics
    int validCount = 0;
    int invalidCount = 0;
    int noValidHighResCount = 0;      // LowRes tets with no VALID HighRes tets
    int invalidWithValidHighRes = 0;  // Problem: LowRes invalid but has valid HighRes
    int validWithNoValidHighRes = 0;  // Problem: LowRes valid but no valid HighRes

    for (const auto& info : unstableTets) {
        if (info.isValid) validCount++;
        else invalidCount++;

        if (info.validHighResTetCount == 0) noValidHighResCount++;

        // Check for problematic cases
        if (!info.isValid && info.validHighResTetCount > 0) invalidWithValidHighRes++;
        if (info.isValid && info.validHighResTetCount == 0) validWithNoValidHighRes++;
    }

    std::cout << "\n========== Unstable Tetrahedra Debug Info ==========" << std::endl;
    std::cout << "Velocity Threshold: " << std::fixed << std::setprecision(2) << velocityThreshold << std::endl;
    std::cout << "Total LowRes Tets: " << numLowTets << std::endl;
    std::cout << "Unstable Tets: " << unstableTets.size() << std::endl;
    std::cout << "  - LowRes Valid (Y): " << validCount << std::endl;
    std::cout << "  - LowRes Invalid (N): " << invalidCount << std::endl;
    std::cout << "  - No VALID HighRes (validHighRes=0): " << noValidHighResCount << std::endl;

    if (invalidWithValidHighRes > 0) {
        std::cout << "  - *** PROBLEM: LowRes Invalid but has VALID HighRes: " << invalidWithValidHighRes << " ***" << std::endl;
    }
    if (validWithNoValidHighRes > 0) {
        std::cout << "  - *** WARNING: LowRes Valid but no VALID HighRes: " << validWithNoValidHighRes << " ***" << std::endl;
    }

    if (unstableTets.empty()) {
        std::cout << "-> No unstable tetrahedra detected" << std::endl;
    } else {
        std::cout << "\n[LowTetIdx] | LowValid | AvgVel  | MaxVel  | AllHi | ValidHi" << std::endl;
        std::cout << "---------------------------------------------------------------" << std::endl;

        int displayCount = std::min(static_cast<int>(unstableTets.size()), 20);
        for (int i = 0; i < displayCount; i++) {
            const auto& info = unstableTets[i];
            std::cout << "  [" << std::setw(5) << info.lowResTetIdx << "]  |    "
                      << (info.isValid ? "Y" : "N") << "   | "
                      << std::fixed << std::setprecision(2) << std::setw(7) << info.avgVelocityMag << " | "
                      << std::setw(7) << info.maxVelocityMag << " | "
                      << std::setw(5) << info.skinnedHighResTetCount << " | "
                      << std::setw(5) << info.validHighResTetCount;

            // Highlight problematic cases
            if (!info.isValid && info.validHighResTetCount > 0) {
                std::cout << " <- PROBLEM: Invalid LowRes has VALID HighRes!";
            } else if (info.isValid && info.validHighResTetCount == 0) {
                std::cout << " <- WARNING: Valid LowRes has no VALID HighRes";
            }
            std::cout << std::endl;
        }

        if (unstableTets.size() > 20) {
            std::cout << "... and " << (unstableTets.size() - 20) << " more unstable tets" << std::endl;
        }
    }
    std::cout << "====================================================\n" << std::endl;
}

int SoftBodyGPUDuo::resetUnstableTetVelocities(float velocityThreshold) {
    std::set<int> verticesToReset;
    int resetTetCount = 0;

    for (size_t tetIdx = 0; tetIdx < numLowTets; tetIdx++) {
        int v0 = lowRes_tetIds[tetIdx * 4 + 0];
        int v1 = lowRes_tetIds[tetIdx * 4 + 1];
        int v2 = lowRes_tetIds[tetIdx * 4 + 2];
        int v3 = lowRes_tetIds[tetIdx * 4 + 3];

        auto getMag = [&](int v) {
            return std::sqrt(
                lowRes_velocities[v*3] * lowRes_velocities[v*3] +
                lowRes_velocities[v*3+1] * lowRes_velocities[v*3+1] +
                lowRes_velocities[v*3+2] * lowRes_velocities[v*3+2]);
        };

        float maxMag = std::max({getMag(v0), getMag(v1), getMag(v2), getMag(v3)});

        if (maxMag > velocityThreshold) {
            verticesToReset.insert(v0);
            verticesToReset.insert(v1);
            verticesToReset.insert(v2);
            verticesToReset.insert(v3);
            resetTetCount++;
        }
    }

    // Reset velocities AND prevPositions
    for (int vertIdx : verticesToReset) {
        lowRes_velocities[vertIdx * 3 + 0] = 0.0f;
        lowRes_velocities[vertIdx * 3 + 1] = 0.0f;
        lowRes_velocities[vertIdx * 3 + 2] = 0.0f;

        if (!lowRes_prevPositions.empty()) {
            lowRes_prevPositions[vertIdx * 3 + 0] = lowRes_positions[vertIdx * 3 + 0];
            lowRes_prevPositions[vertIdx * 3 + 1] = lowRes_positions[vertIdx * 3 + 1];
            lowRes_prevPositions[vertIdx * 3 + 2] = lowRes_positions[vertIdx * 3 + 2];
        }
    }

    std::cout << "[resetUnstableTetVelocities] Reset " << verticesToReset.size()
              << " vertices from " << resetTetCount << " unstable tets (threshold: "
              << velocityThreshold << ")" << std::endl;

    return static_cast<int>(verticesToReset.size());
}

void SoftBodyGPUDuo::diagnoseVibrationCause(float velocityThreshold) {
    std::cout << "\n========== Vibration Diagnosis ==========" << std::endl;

    // 1. 最も速い頂点を見つける
    int fastestVertex = -1;
    float maxVel = 0.0f;

    for (size_t i = 0; i < numLowResParticles; i++) {
        float vx = lowRes_velocities[i * 3];
        float vy = lowRes_velocities[i * 3 + 1];
        float vz = lowRes_velocities[i * 3 + 2];
        float vel = std::sqrt(vx*vx + vy*vy + vz*vz);

        if (vel > maxVel) {
            maxVel = vel;
            fastestVertex = i;
        }
    }

    if (fastestVertex < 0 || maxVel < velocityThreshold) {
        std::cout << "No significant velocity detected (max: " << maxVel << ")" << std::endl;
        return;
    }

    std::cout << "Fastest vertex: " << fastestVertex << " (vel: " << maxVel << ")" << std::endl;
    std::cout << "  invMass: " << lowRes_invMasses[fastestVertex] << std::endl;
    std::cout << "  Position: ("
              << lowRes_positions[fastestVertex * 3] << ", "
              << lowRes_positions[fastestVertex * 3 + 1] << ", "
              << lowRes_positions[fastestVertex * 3 + 2] << ")" << std::endl;
    std::cout << "  Velocity: ("
              << lowRes_velocities[fastestVertex * 3] << ", "
              << lowRes_velocities[fastestVertex * 3 + 1] << ", "
              << lowRes_velocities[fastestVertex * 3 + 2] << ")" << std::endl;

    // 2. この頂点が属する四面体を見つける
    std::cout << "\nTetrahedra containing vertex " << fastestVertex << ":" << std::endl;

    for (size_t t = 0; t < numLowTets; t++) {
        bool containsVertex = false;
        for (int j = 0; j < 4; j++) {
            if (lowRes_tetIds[t * 4 + j] == fastestVertex) {
                containsVertex = true;
                break;
            }
        }

        if (containsVertex) {
            bool isValid = lowRes_tetValid.empty() || lowRes_tetValid[t];
            float currentVol = lowResGetTetVolume(t);
            float restVol = lowRes_restVols[t];
            float volRatio = (restVol > 0.0f) ? currentVol / restVol : 0.0f;

            std::cout << "  Tet " << t << ": " << (isValid ? "VALID" : "INVALID")
                      << " | Vol: " << currentVol << " / " << restVol
                      << " (ratio: " << volRatio << ")";

            // 大きな体積変化があれば警告
            if (isValid && (volRatio < 0.5f || volRatio > 2.0f)) {
                std::cout << " <- LARGE VOLUME CHANGE!";
            }
            std::cout << std::endl;
        }
    }

    // 3. この頂点を含むエッジを見つける
    std::cout << "\nEdges containing vertex " << fastestVertex << ":" << std::endl;
    int edgeCount = 0;

    for (size_t e = 0; e < lowRes_edgeLengths.size() && edgeCount < 10; e++) {
        int id0 = lowRes_edgeIds[e * 2];
        int id1 = lowRes_edgeIds[e * 2 + 1];

        if (id0 == fastestVertex || id1 == fastestVertex) {
            float currentLen = std::sqrt(VectorMath::vecDistSquared(lowRes_positions, id0, lowRes_positions, id1));
            float restLen = lowRes_edgeLengths[e];
            float lenRatio = (restLen > 0.0f) ? currentLen / restLen : 0.0f;

            std::cout << "  Edge " << e << " (" << id0 << "-" << id1 << "): "
                      << " Len: " << currentLen << " / " << restLen
                      << " (ratio: " << lenRatio << ")";

            // 大きな長さ変化があれば警告
            if (restLen > 0.0f && (lenRatio < 0.5f || lenRatio > 2.0f)) {
                std::cout << " <- LARGE LENGTH CHANGE!";
            }
            std::cout << std::endl;
            edgeCount++;
        }
    }

    // 4. 無効な四面体に隣接しているか確認
    std::cout << "\nChecking for invalid neighbors..." << std::endl;
    int invalidNeighborCount = 0;

    for (size_t t = 0; t < numLowTets; t++) {
        if (lowRes_tetValid.empty() || lowRes_tetValid[t]) continue;  // 有効はスキップ

        // 無効な四面体がfastestVertexを含むか
        for (int j = 0; j < 4; j++) {
            if (lowRes_tetIds[t * 4 + j] == fastestVertex) {
                invalidNeighborCount++;
                std::cout << "  Vertex " << fastestVertex << " belongs to INVALID tet " << t << std::endl;
                break;
            }
        }
    }

    if (invalidNeighborCount > 0) {
        std::cout << "*** WARNING: Fastest vertex belongs to " << invalidNeighborCount
                  << " INVALID tetrahedra! ***" << std::endl;
        std::cout << "This may cause instability at the cut boundary." << std::endl;
    } else {
        std::cout << "Fastest vertex only belongs to valid tetrahedra." << std::endl;
    }

    std::cout << "==========================================\n" << std::endl;
}

// =====================================================
// 最適化版: 隣接リストを使った高速スムージング
// 全頂点の距離計算は重いので、エッジ接続を利用
// =====================================================

// =====================================================
// SoftBodyGPUDuo.h に追加（publicセクション）
// =====================================================

// 振動している低解像度四面体を無効化
int invalidateUnstableLowResTets(float velocityThreshold = 0.5f);



// =====================================================
// 修正1: SoftBodyGPUDuo.cpp - updateHighResPositions
// lowRes_tetValid をチェックしてスムージングを適用
// =====================================================

// =====================================================
// 修正2: SoftBodyGPUDuo.cpp - invalidateUnstableLowResTets
// 高解像度四面体は無効化しない
// =====================================================

int SoftBodyGPUDuo::invalidateUnstableLowResTets(float velocityThreshold) {
    std::cout << "\n=== Invalidating Unstable Low-Res Tets ===" << std::endl;

    int invalidatedCount = 0;
    std::vector<int> tetsToInvalidate;

    for (size_t tetIdx = 0; tetIdx < numLowTets; tetIdx++) {
        // 既に無効ならスキップ
        if (!lowRes_tetValid.empty() && !lowRes_tetValid[tetIdx]) continue;

        // 4頂点のインデックスを取得
        int v0 = lowRes_tetIds[tetIdx * 4 + 0];
        int v1 = lowRes_tetIds[tetIdx * 4 + 1];
        int v2 = lowRes_tetIds[tetIdx * 4 + 2];
        int v3 = lowRes_tetIds[tetIdx * 4 + 3];

        // 速度の大きさを計算
        auto getMag = [&](int v) {
            return std::sqrt(
                lowRes_velocities[v*3] * lowRes_velocities[v*3] +
                lowRes_velocities[v*3+1] * lowRes_velocities[v*3+1] +
                lowRes_velocities[v*3+2] * lowRes_velocities[v*3+2]);
        };

        float maxMag = std::max({getMag(v0), getMag(v1), getMag(v2), getMag(v3)});

        // 閾値を超えていれば無効化リストに追加
        if (maxMag > velocityThreshold) {
            tetsToInvalidate.push_back(static_cast<int>(tetIdx));
        }
    }

    // 低解像度四面体のみ無効化
    for (int tetIdx : tetsToInvalidate) {
        lowRes_tetValid[tetIdx] = false;
        invalidatedCount++;
    }

    std::cout << "Invalidated " << invalidatedCount << " low-res tets (threshold: "
              << velocityThreshold << ")" << std::endl;

    // ★★★ 高解像度四面体は無効化しない ★★★
    // 高解像度メッシュの位置は updateHighResPositions() で
    // applySmoothingCorrectionHighRes() によりスムージングで計算される

    if (invalidatedCount > 0) {
        // メッシュ表示更新（低解像度のみ）
        setupLowResTetMesh();
        updateLowResMesh();

        // サーフェスキャッシュを無効化
        invalidateSurfaceCache();
    }

    std::cout << "==========================================\n" << std::endl;

    return invalidatedCount;
}


// =====================================================
// 修正3: SoftBodyGPUDuo.cpp - applySmoothingCorrectionHighRes
// (ヘッダーに宣言が必要: void applySmoothingCorrectionHighRes(const std::vector<bool>& needsCorrection);)
// =====================================================

void SoftBodyGPUDuo::applySmoothingCorrectionHighRes(const std::vector<bool>& needsCorrection) {
    // 有効な頂点の変位を計算
    std::vector<glm::vec3> displacement(numHighResVerts, glm::vec3(0.0f));
    std::vector<int> validIndices;

    for (size_t i = 0; i < numHighResVerts; i++) {
        if (!needsCorrection[i]) {
            // オリジナル位置からの変位
            displacement[i] = glm::vec3(
                highRes_positions[3*i]   - highResMeshData.verts[3*i],
                highRes_positions[3*i+1] - highResMeshData.verts[3*i+1],
                highRes_positions[3*i+2] - highResMeshData.verts[3*i+2]
            );
            validIndices.push_back(static_cast<int>(i));
        }
    }

    if (validIndices.empty()) {
        // 有効な頂点がない場合、元の位置を保持
        return;
    }

    // 補正対象頂点に変位を伝播（距離ベースの加重平均）
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(numHighResVerts); i++) {
        if (!needsCorrection[i]) continue;

        glm::vec3 myOrigPos(highResMeshData.verts[3*i],
                           highResMeshData.verts[3*i+1],
                           highResMeshData.verts[3*i+2]);

        glm::vec3 avgDisp(0.0f);
        float totalWeight = 0.0f;

        // 全有効頂点からの距離加重平均
        for (int j : validIndices) {
            glm::vec3 otherOrigPos(highResMeshData.verts[3*j],
                                   highResMeshData.verts[3*j+1],
                                   highResMeshData.verts[3*j+2]);

            float dist = glm::distance(myOrigPos, otherOrigPos);
            float weight = 1.0f / (dist * dist + 0.0001f);

            avgDisp += weight * displacement[j];
            totalWeight += weight;
        }

        if (totalWeight > 0.0f) {
            avgDisp /= totalWeight;
            highRes_positions[3*i]   = highResMeshData.verts[3*i]   + avgDisp.x;
            highRes_positions[3*i+1] = highResMeshData.verts[3*i+1] + avgDisp.y;
            highRes_positions[3*i+2] = highResMeshData.verts[3*i+2] + avgDisp.z;
        }
    }
}



std::string SoftBodyGPUDuo::cycleSmoothingPreset() {
    // 現在のプリセットを判定して次に切り替え
    static int currentPreset = 0;
    currentPreset = (currentPreset + 1) % 5;

    switch (currentPreset) {
        case 0: // Balanced（バランス）
            smoothingParams_.maxIterations = 3;
            smoothingParams_.laplacianFactor = 0.5f;
            smoothingParams_.edgeFactor = 0.3f;
            smoothingParams_.presetName = "Balanced";
            break;

        case 1: // Smooth（滑らか優先）
            smoothingParams_.maxIterations = 9;
            smoothingParams_.laplacianFactor = 0.9f;
            smoothingParams_.edgeFactor = 0.1f;
            smoothingParams_.presetName = "Smooth";
            break;

        case 2: // Shape Preserve（形状保持優先）
            smoothingParams_.maxIterations = 3;
            smoothingParams_.laplacianFactor = 0.3f;
            smoothingParams_.edgeFactor = 0.5f;
            smoothingParams_.presetName = "Shape Preserve";
            break;

        case 3: // Fast（高速）
            smoothingParams_.maxIterations = 1;
            smoothingParams_.laplacianFactor = 0.6f;
            smoothingParams_.edgeFactor = 0.4f;
            smoothingParams_.presetName = "Fast";
            break;

        case 4: // Precise（精密）
            smoothingParams_.maxIterations = 5;
            smoothingParams_.laplacianFactor = 0.4f;
            smoothingParams_.edgeFactor = 0.4f;
            smoothingParams_.presetName = "Precise";
            break;
    }

    std::cout << "[Smoothing] Preset changed to: " << smoothingParams_.presetName
              << " (iter=" << smoothingParams_.maxIterations
              << ", laplacian=" << smoothingParams_.laplacianFactor
              << ", edge=" << smoothingParams_.edgeFactor << ")" << std::endl;

    return smoothingParams_.presetName;
}



// =====================================================
// 2. applySmoothingCorrectionHighResFast にデバッグ追加
// =====================================================

void SoftBodyGPUDuo::applySmoothingCorrectionHighResFast(const std::vector<bool>& needsCorrection) {

    // ========================================
    // デバッグ: 補正対象頂点数をカウント
    // ========================================
    int correctionCount = 0;
    for (size_t i = 0; i < needsCorrection.size(); i++) {
        if (needsCorrection[i]) correctionCount++;
    }

    lastSmoothingDebug_.totalVertices = static_cast<int>(numHighResVerts);
    lastSmoothingDebug_.needsCorrectionCount = correctionCount;
    lastSmoothingDebug_.actuallySmoothedCount = 0;
    lastSmoothingDebug_.maxDisplacement = 0.0f;
    lastSmoothingDebug_.avgDisplacement = 0.0f;

    if (smoothingDebugMode_) {
        std::cout << "[Smoothing DEBUG] Total vertices: " << numHighResVerts
                  << ", needsCorrection: " << correctionCount
                  << " (" << (100.0f * correctionCount / numHighResVerts) << "%)" << std::endl;
    }

    // 補正対象がなければ終了
    if (correctionCount == 0) {
        if (smoothingDebugMode_) {
            std::cout << "[Smoothing DEBUG] No vertices need correction - skipping" << std::endl;
        }
        return;
    }

    // ========================================
    // キャッシュ構築（初回のみ）
    // ========================================
    if (!highResNeighborsCacheBuilt_) {
        highResNeighborsCache_.clear();
        highResNeighborsCache_.resize(numHighResVerts);

        for (size_t e = 0; e < highResEdgeIds.size() / 2; e++) {
            int id0 = highResEdgeIds[e * 2];
            int id1 = highResEdgeIds[e * 2 + 1];
            if (id0 >= 0 && id0 < static_cast<int>(numHighResVerts) &&
                id1 >= 0 && id1 < static_cast<int>(numHighResVerts)) {
                highResNeighborsCache_[id0].push_back(id1);
                highResNeighborsCache_[id1].push_back(id0);
            }
        }
        highResNeighborsCacheBuilt_ = true;

        std::cout << "[Smoothing] Neighbor cache built: "
                  << numHighResVerts << " vertices" << std::endl;
    }

    if (!highResEdgeRestLengthsBuilt_) {
        size_t numEdges = highResEdgeIds.size() / 2;
        highResEdgeRestLengths_.resize(numEdges);

        for (size_t e = 0; e < numEdges; e++) {
            int id0 = highResEdgeIds[e * 2];
            int id1 = highResEdgeIds[e * 2 + 1];

            if (id0 >= 0 && id0 < static_cast<int>(numHighResVerts) &&
                id1 >= 0 && id1 < static_cast<int>(numHighResVerts)) {

                glm::vec3 orig0(highResMeshData.verts[id0 * 3],
                                highResMeshData.verts[id0 * 3 + 1],
                                highResMeshData.verts[id0 * 3 + 2]);
                glm::vec3 orig1(highResMeshData.verts[id1 * 3],
                                highResMeshData.verts[id1 * 3 + 1],
                                highResMeshData.verts[id1 * 3 + 2]);

                highResEdgeRestLengths_[e] = glm::distance(orig0, orig1);
            } else {
                highResEdgeRestLengths_[e] = 0.0f;
            }
        }
        highResEdgeRestLengthsBuilt_ = true;

        std::cout << "[Smoothing] Edge rest lengths cached: "
                  << numEdges << " edges" << std::endl;
    }

    // ========================================
    // パラメータ取得
    // ========================================
    const int maxIterations = smoothingParams_.maxIterations;
    const float laplacianFactor = smoothingParams_.laplacianFactor;
    const float edgeFactor = smoothingParams_.edgeFactor;

    // ========================================
    // デバッグ: 変位前の位置を保存
    // ========================================
    std::vector<float> positionsBefore;
    if (smoothingDebugMode_) {
        positionsBefore = highRes_positions;
    }

    // ========================================
    // メインループ
    // ========================================
    int actuallySmoothed = 0;

    for (int iter = 0; iter < maxIterations; iter++) {

        // Phase 1: Laplacianスムージング
        std::vector<float> newPositions = highRes_positions;

        #pragma omp parallel for reduction(+:actuallySmoothed)
        for (int i = 0; i < static_cast<int>(numHighResVerts); i++) {
            if (!needsCorrection[i]) continue;

            const auto& neighbors = highResNeighborsCache_[i];
            if (neighbors.empty()) continue;

            glm::vec3 avgPos(0.0f);
            int validCount = 0;

            for (int neighborId : neighbors) {
                if (!needsCorrection[neighborId]) {
                    avgPos.x += highRes_positions[neighborId * 3];
                    avgPos.y += highRes_positions[neighborId * 3 + 1];
                    avgPos.z += highRes_positions[neighborId * 3 + 2];
                    validCount++;
                }
            }

            if (validCount > 0) {
                avgPos /= static_cast<float>(validCount);

                glm::vec3 oldPos(highRes_positions[i * 3],
                                 highRes_positions[i * 3 + 1],
                                 highRes_positions[i * 3 + 2]);

                glm::vec3 newPos = oldPos * (1.0f - laplacianFactor) + avgPos * laplacianFactor;

                newPositions[i * 3]     = newPos.x;
                newPositions[i * 3 + 1] = newPos.y;
                newPositions[i * 3 + 2] = newPos.z;

                if (iter == 0) actuallySmoothed++;
            }
        }

        highRes_positions = newPositions;

        // Phase 2: エッジ長制約
        size_t numEdges = highResEdgeIds.size() / 2;

        for (size_t e = 0; e < numEdges; e++) {
            int id0 = highResEdgeIds[e * 2];
            int id1 = highResEdgeIds[e * 2 + 1];

            if (id0 < 0 || id0 >= static_cast<int>(numHighResVerts) ||
                id1 < 0 || id1 >= static_cast<int>(numHighResVerts)) {
                continue;
            }

            if (!needsCorrection[id0] && !needsCorrection[id1]) continue;

            glm::vec3 p0(highRes_positions[id0 * 3],
                         highRes_positions[id0 * 3 + 1],
                         highRes_positions[id0 * 3 + 2]);
            glm::vec3 p1(highRes_positions[id1 * 3],
                         highRes_positions[id1 * 3 + 1],
                         highRes_positions[id1 * 3 + 2]);

            float restLength = highResEdgeRestLengths_[e];
            float currentLength = glm::distance(p0, p1);

            if (currentLength < 0.0001f || restLength < 0.0001f) continue;

            glm::vec3 dir = (p1 - p0) / currentLength;
            float diff = currentLength - restLength;
            glm::vec3 correction = dir * diff * edgeFactor * 0.5f;

            if (needsCorrection[id0] && needsCorrection[id1]) {
                highRes_positions[id0 * 3]     += correction.x;
                highRes_positions[id0 * 3 + 1] += correction.y;
                highRes_positions[id0 * 3 + 2] += correction.z;

                highRes_positions[id1 * 3]     -= correction.x;
                highRes_positions[id1 * 3 + 1] -= correction.y;
                highRes_positions[id1 * 3 + 2] -= correction.z;
            } else if (needsCorrection[id0]) {
                highRes_positions[id0 * 3]     += correction.x * 2.0f;
                highRes_positions[id0 * 3 + 1] += correction.y * 2.0f;
                highRes_positions[id0 * 3 + 2] += correction.z * 2.0f;
            } else if (needsCorrection[id1]) {
                highRes_positions[id1 * 3]     -= correction.x * 2.0f;
                highRes_positions[id1 * 3 + 1] -= correction.y * 2.0f;
                highRes_positions[id1 * 3 + 2] -= correction.z * 2.0f;
            }
        }
    }

    // ========================================
    // デバッグ: 変位量を計算
    // ========================================
    lastSmoothingDebug_.actuallySmoothedCount = actuallySmoothed;

    if (smoothingDebugMode_) {
        float maxDisp = 0.0f;
        float totalDisp = 0.0f;
        int dispCount = 0;

        for (size_t i = 0; i < numHighResVerts; i++) {
            if (needsCorrection[i]) {
                glm::vec3 before(positionsBefore[i * 3],
                                 positionsBefore[i * 3 + 1],
                                 positionsBefore[i * 3 + 2]);
                glm::vec3 after(highRes_positions[i * 3],
                                highRes_positions[i * 3 + 1],
                                highRes_positions[i * 3 + 2]);
                float disp = glm::distance(before, after);
                maxDisp = std::max(maxDisp, disp);
                totalDisp += disp;
                dispCount++;
            }
        }

        lastSmoothingDebug_.maxDisplacement = maxDisp;
        lastSmoothingDebug_.avgDisplacement = (dispCount > 0) ? totalDisp / dispCount : 0.0f;

        std::cout << "[Smoothing DEBUG] Actually smoothed: " << actuallySmoothed
                  << ", Max displacement: " << maxDisp
                  << ", Avg displacement: " << lastSmoothingDebug_.avgDisplacement << std::endl;
    }
}



void SoftBodyGPUDuo::updateHighResPositions() {
    if (!useHighResMesh) return;

    std::vector<bool> needsCorrection(numHighResVerts, false);

    int validSkinningCount = 0;
    int invalidTetCount = 0;
    int outOfRangeCount = 0;

    for (size_t i = 0; i < numHighResVerts; i++) {
        int tetIdx = static_cast<int>(skinningInfoLowToHigh[8 * i]);

        // 範囲チェック
        if (tetIdx < 0 || tetIdx >= static_cast<int>(numLowTets)) {
            needsCorrection[i] = true;
            outOfRangeCount++;
            continue;
        }

        // 子lowRes四面体が無効 → 補正対象
        if (!lowRes_tetValid.empty() && !lowRes_tetValid[tetIdx]) {
            needsCorrection[i] = true;
            invalidTetCount++;
            continue;
        }

        // 有効なスキニング
        validSkinningCount++;

        float b0 = skinningInfoLowToHigh[8 * i + 1];
        float b1 = skinningInfoLowToHigh[8 * i + 2];
        float b2 = skinningInfoLowToHigh[8 * i + 3];
        float b3 = skinningInfoLowToHigh[8 * i + 4];

        int id0 = lowRes_tetIds[4 * tetIdx];
        int id1 = lowRes_tetIds[4 * tetIdx + 1];
        int id2 = lowRes_tetIds[4 * tetIdx + 2];
        int id3 = lowRes_tetIds[4 * tetIdx + 3];

        highRes_positions[i * 3]     = b0 * lowRes_positions[id0 * 3] +
                                       b1 * lowRes_positions[id1 * 3] +
                                       b2 * lowRes_positions[id2 * 3] +
                                       b3 * lowRes_positions[id3 * 3];

        highRes_positions[i * 3 + 1] = b0 * lowRes_positions[id0 * 3 + 1] +
                                       b1 * lowRes_positions[id1 * 3 + 1] +
                                       b2 * lowRes_positions[id2 * 3 + 1] +
                                       b3 * lowRes_positions[id3 * 3 + 1];

        highRes_positions[i * 3 + 2] = b0 * lowRes_positions[id0 * 3 + 2] +
                                       b1 * lowRes_positions[id1 * 3 + 2] +
                                       b2 * lowRes_positions[id2 * 3 + 2] +
                                       b3 * lowRes_positions[id3 * 3 + 2];
    }

    if (smoothingDebugMode_) {
        std::cout << "[updateHighResPositions XPBD] "
                  << "Valid: " << validSkinningCount
                  << ", Invalid SELF: " << invalidTetCount
                  << ", OutOfRange: " << outOfRangeCount << std::endl;
    }

    bool hasCorrection = false;
    for (size_t i = 0; i < numHighResVerts && !hasCorrection; i++) {
        if (needsCorrection[i]) hasCorrection = true;
    }

    if (hasCorrection) {
        applySmoothingCorrectionHighResFast(needsCorrection);
    }
}


// =====================================================
// 3. updateHighResPositionsOnlySkinning() - 新規
//    子lowResは有効だが、親lowResが無効な場合も補正
// =====================================================

void SoftBodyGPUDuo::updateHighResPositionsOnlySkinning() {
    if (!useHighResMesh) return;

    std::vector<bool> needsCorrection(numHighResVerts, false);

    int validSkinningCount = 0;
    int invalidSelfTetCount = 0;
    int invalidParentTetCount = 0;
    int outOfRangeCount = 0;

    // ===== 親の四面体有効性を事前キャッシュ（子lowRes頂点ごと）=====
    std::vector<bool> parentTetInvalidForLowResVert;
    if (parentSoftBody != nullptr && !skinningToParent.empty()) {
        parentTetInvalidForLowResVert.resize(numLowResParticles, false);

        for (size_t i = 0; i < numLowResParticles; i++) {
            int parentTetIdx = static_cast<int>(skinningToParent[4 * i + 0]);

            if (parentTetIdx >= 0 &&
                parentTetIdx < static_cast<int>(parentSoftBody->lowRes_tetValid.size())) {
                if (!parentSoftBody->lowRes_tetValid[parentTetIdx]) {
                    parentTetInvalidForLowResVert[i] = true;
                }
            }
        }
    }

    for (size_t i = 0; i < numHighResVerts; i++) {
        int tetIdx = static_cast<int>(skinningInfoLowToHigh[8 * i]);

        // 範囲チェック
        if (tetIdx < 0 || tetIdx >= static_cast<int>(numLowTets)) {
            needsCorrection[i] = true;
            outOfRangeCount++;
            continue;
        }

        // 子lowRes四面体が無効 → 補正対象
        if (!lowRes_tetValid.empty() && !lowRes_tetValid[tetIdx]) {
            needsCorrection[i] = true;
            invalidSelfTetCount++;
            continue;
        }

        // ★新規: 親lowRes四面体が無効 → 補正対象
        if (!parentTetInvalidForLowResVert.empty()) {
            bool anyParentInvalid = false;

            // この子lowRes四面体の4頂点のうち、親の無効四面体にスキニングされているものがあるか
            for (int v = 0; v < 4; v++) {
                int lowResVertIdx = lowRes_tetIds[4 * tetIdx + v];

                if (lowResVertIdx >= 0 &&
                    lowResVertIdx < static_cast<int>(parentTetInvalidForLowResVert.size())) {
                    if (parentTetInvalidForLowResVert[lowResVertIdx]) {
                        anyParentInvalid = true;
                        break;
                    }
                }
            }

            if (anyParentInvalid) {
                needsCorrection[i] = true;
                invalidParentTetCount++;
                continue;
            }
        }

        // 有効なスキニング
        validSkinningCount++;

        float b0 = skinningInfoLowToHigh[8 * i + 1];
        float b1 = skinningInfoLowToHigh[8 * i + 2];
        float b2 = skinningInfoLowToHigh[8 * i + 3];
        float b3 = skinningInfoLowToHigh[8 * i + 4];

        int id0 = lowRes_tetIds[4 * tetIdx];
        int id1 = lowRes_tetIds[4 * tetIdx + 1];
        int id2 = lowRes_tetIds[4 * tetIdx + 2];
        int id3 = lowRes_tetIds[4 * tetIdx + 3];

        highRes_positions[i * 3]     = b0 * lowRes_positions[id0 * 3] +
                                       b1 * lowRes_positions[id1 * 3] +
                                       b2 * lowRes_positions[id2 * 3] +
                                       b3 * lowRes_positions[id3 * 3];

        highRes_positions[i * 3 + 1] = b0 * lowRes_positions[id0 * 3 + 1] +
                                       b1 * lowRes_positions[id1 * 3 + 1] +
                                       b2 * lowRes_positions[id2 * 3 + 1] +
                                       b3 * lowRes_positions[id3 * 3 + 1];

        highRes_positions[i * 3 + 2] = b0 * lowRes_positions[id0 * 3 + 2] +
                                       b1 * lowRes_positions[id1 * 3 + 2] +
                                       b2 * lowRes_positions[id2 * 3 + 2] +
                                       b3 * lowRes_positions[id3 * 3 + 2];
    }

    if (smoothingDebugMode_) {
        std::cout << "[updateHighResPositions SKINNING] "
                  << "Valid: " << validSkinningCount
                  << ", Invalid SELF: " << invalidSelfTetCount
                  << ", Invalid PARENT: " << invalidParentTetCount
                  << ", OutOfRange: " << outOfRangeCount << std::endl;
    }

    bool hasCorrection = false;
    for (size_t i = 0; i < numHighResVerts && !hasCorrection; i++) {
        if (needsCorrection[i]) hasCorrection = true;
    }

    if (hasCorrection) {
        applySmoothingCorrectionHighResFast(needsCorrection);
    }
}


// 符号付き体積を計算するヘルパー関数
inline float computeSignedVolume(const glm::vec3& p0, const glm::vec3& p1,
                                  const glm::vec3& p2, const glm::vec3& p3) {
    glm::vec3 d1 = p1 - p0;
    glm::vec3 d2 = p2 - p0;
    glm::vec3 d3 = p3 - p0;
    return glm::dot(glm::cross(d1, d2), d3) / 6.0f;
}

// =====================================================
// 3. updateHighResPositionsOnlySkinning() を簡略化
//    子lowRes → 子highRes の単純スキニングのみ
// =====================================================
void SoftBodyGPUDuo::updateHighResPositionsSimpleSkinning() {
    if (!useHighResMesh) return;

    int validCount = 0;
    int skippedCount = 0;

    for (size_t i = 0; i < numHighResVerts; i++) {
        int tetIdx = static_cast<int>(skinningInfoLowToHigh[8 * i]);

        // マッピングがない場合はスキップ（位置保持）
        if (tetIdx < 0 || tetIdx >= static_cast<int>(numLowTets)) {
            skippedCount++;
            continue;
        }

        // 四面体が無効でもスキニングは実行
        // （子lowResは solveFreeVerticesWithConstraints で更新済み）

        float b0 = skinningInfoLowToHigh[8 * i + 1];
        float b1 = skinningInfoLowToHigh[8 * i + 2];
        float b2 = skinningInfoLowToHigh[8 * i + 3];
        float b3 = skinningInfoLowToHigh[8 * i + 4];

        int id0 = lowRes_tetIds[4 * tetIdx];
        int id1 = lowRes_tetIds[4 * tetIdx + 1];
        int id2 = lowRes_tetIds[4 * tetIdx + 2];
        int id3 = lowRes_tetIds[4 * tetIdx + 3];

        highRes_positions[i * 3]     = b0 * lowRes_positions[id0 * 3] +
                                       b1 * lowRes_positions[id1 * 3] +
                                       b2 * lowRes_positions[id2 * 3] +
                                       b3 * lowRes_positions[id3 * 3];

        highRes_positions[i * 3 + 1] = b0 * lowRes_positions[id0 * 3 + 1] +
                                       b1 * lowRes_positions[id1 * 3 + 1] +
                                       b2 * lowRes_positions[id2 * 3 + 1] +
                                       b3 * lowRes_positions[id3 * 3 + 1];

        highRes_positions[i * 3 + 2] = b0 * lowRes_positions[id0 * 3 + 2] +
                                       b1 * lowRes_positions[id1 * 3 + 2] +
                                       b2 * lowRes_positions[id2 * 3 + 2] +
                                       b3 * lowRes_positions[id3 * 3 + 2];

        validCount++;
    }

    if (smoothingDebugMode_) {
        std::cout << "[SimpleSkinning] Valid: " << validCount
                  << ", Skipped: " << skippedCount << std::endl;
    }
}

// =============================================================================
// 方法1: updateFreeVerticesWithSmoothing() - IDW
// 振動: なし ✓  形状保持: 弱い
// =============================================================================

void SoftBodyGPUDuo::updateFreeVerticesWithSmoothing() {
    if (parentSoftBody == nullptr) return;

    // Phase 1: アンカーされている頂点の変位を収集
    std::vector<glm::vec3> displacement(numLowResParticles, glm::vec3(0.0f));
    std::vector<int> validIndices;

    for (size_t i = 0; i < numLowResParticles; i++) {
        if (isAnchoredToParent[i]) {
            displacement[i] = glm::vec3(
                lowRes_positions[3*i]   - lowResMeshData.verts[3*i],
                lowRes_positions[3*i+1] - lowResMeshData.verts[3*i+1],
                lowRes_positions[3*i+2] - lowResMeshData.verts[3*i+2]
            );
            validIndices.push_back(static_cast<int>(i));
        }
    }

    if (validIndices.empty()) {
        return;
    }

    // Phase 2: 自由頂点に変位を伝播（距離ベース加重平均）
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(numLowResParticles); i++) {
        if (isAnchoredToParent[i]) continue;

        glm::vec3 myOrigPos(
            lowResMeshData.verts[3*i],
            lowResMeshData.verts[3*i+1],
            lowResMeshData.verts[3*i+2]
        );

        glm::vec3 avgDisp(0.0f);
        float totalWeight = 0.0f;

        for (int j : validIndices) {
            glm::vec3 otherOrigPos(
                lowResMeshData.verts[3*j],
                lowResMeshData.verts[3*j+1],
                lowResMeshData.verts[3*j+2]
            );

            float dist = glm::distance(myOrigPos, otherOrigPos);
            float weight = 1.0f / (dist * dist + 0.0001f);

            avgDisp += weight * displacement[j];
            totalWeight += weight;
        }

        if (totalWeight > 0.0f) {
            avgDisp /= totalWeight;

            lowRes_positions[3*i]   = lowResMeshData.verts[3*i]   + avgDisp.x;
            lowRes_positions[3*i+1] = lowResMeshData.verts[3*i+1] + avgDisp.y;
            lowRes_positions[3*i+2] = lowResMeshData.verts[3*i+2] + avgDisp.z;

            lowRes_prevPositions[3*i]   = lowRes_positions[3*i];
            lowRes_prevPositions[3*i+1] = lowRes_positions[3*i+1];
            lowRes_prevPositions[3*i+2] = lowRes_positions[3*i+2];

            lowRes_velocities[3*i]   = 0.0f;
            lowRes_velocities[3*i+1] = 0.0f;
            lowRes_velocities[3*i+2] = 0.0f;
        }
    }
}


// =============================================================================
// 方法2: solveFreeVerticesStable() - IDW + ブレンド
// 振動: なし ✓  形状保持: 弱い
// 特徴: 目標位置へゆっくりブレンドするため、より滑らかな動き
// =============================================================================

void SoftBodyGPUDuo::solveFreeVerticesStable() {
    if (parentSoftBody == nullptr) return;

    // アンカー頂点の変位を収集
    std::vector<glm::vec3> displacement(numLowResParticles, glm::vec3(0.0f));
    std::vector<int> validIndices;

    for (size_t i = 0; i < numLowResParticles; i++) {
        if (isAnchoredToParent[i]) {
            displacement[i] = glm::vec3(
                lowRes_positions[3*i]   - lowResMeshData.verts[3*i],
                lowRes_positions[3*i+1] - lowResMeshData.verts[3*i+1],
                lowRes_positions[3*i+2] - lowResMeshData.verts[3*i+2]
            );
            validIndices.push_back(static_cast<int>(i));
        }
    }

    if (validIndices.empty()) return;

    // パラメータ
    const float blendFactor = 0.3f;  // 目標位置へのブレンド率（0.1〜0.5）

    // 自由頂点の目標位置を計算してブレンド
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(numLowResParticles); i++) {
        if (isAnchoredToParent[i]) continue;

        glm::vec3 origPos(
            lowResMeshData.verts[3*i],
            lowResMeshData.verts[3*i+1],
            lowResMeshData.verts[3*i+2]
        );

        glm::vec3 currentPos(
            lowRes_positions[3*i],
            lowRes_positions[3*i+1],
            lowRes_positions[3*i+2]
        );

        // IDWで目標変位を計算
        glm::vec3 targetDisp(0.0f);
        float totalWeight = 0.0f;

        for (int j : validIndices) {
            glm::vec3 otherOrigPos(
                lowResMeshData.verts[3*j],
                lowResMeshData.verts[3*j+1],
                lowResMeshData.verts[3*j+2]
            );

            float dist = glm::distance(origPos, otherOrigPos);
            float weight = 1.0f / (dist * dist + 0.0001f);

            targetDisp += weight * displacement[j];
            totalWeight += weight;
        }

        if (totalWeight > 0.0f) {
            targetDisp /= totalWeight;

            // 目標位置
            glm::vec3 targetPos = origPos + targetDisp;

            // 現在位置から目標位置へゆっくりブレンド
            glm::vec3 newPos = currentPos + (targetPos - currentPos) * blendFactor;

            lowRes_positions[3*i]   = newPos.x;
            lowRes_positions[3*i+1] = newPos.y;
            lowRes_positions[3*i+2] = newPos.z;

            lowRes_prevPositions[3*i]   = newPos.x;
            lowRes_prevPositions[3*i+1] = newPos.y;
            lowRes_prevPositions[3*i+2] = newPos.z;

            lowRes_velocities[3*i]   = 0.0f;
            lowRes_velocities[3*i+1] = 0.0f;
            lowRes_velocities[3*i+2] = 0.0f;
        }
    }
}


// =============================================================================
// 方法3: solveFreeVerticesWithDamping() - Laplacian + ダンピング
// 振動: 少し  形状保持: 中程度
// 特徴: 隣接頂点ベースで形状を保持しつつ、ダンピングで振動を抑制
// =============================================================================

void SoftBodyGPUDuo::solveFreeVerticesWithDamping() {
    if (parentSoftBody == nullptr) return;

    // 隣接リストを構築（初回のみ）
    if (!lowResNeighborsCacheBuilt_) {
        lowResNeighborsCache_.clear();
        lowResNeighborsCache_.resize(numLowResParticles);

        for (size_t e = 0; e < lowRes_edgeIds.size() / 2; e++) {
            int id0 = lowRes_edgeIds[e * 2];
            int id1 = lowRes_edgeIds[e * 2 + 1];
            if (id0 >= 0 && id0 < static_cast<int>(numLowResParticles) &&
                id1 >= 0 && id1 < static_cast<int>(numLowResParticles)) {
                lowResNeighborsCache_[id0].push_back(id1);
                lowResNeighborsCache_[id1].push_back(id0);
            }
        }
        lowResNeighborsCacheBuilt_ = true;
    }

    // パラメータ
    const float laplacianFactor = 0.5f;
    const float dampingFactor = 0.8f;  // 速度の減衰率

    // 前フレームの位置を保存
    std::vector<float> oldPositions = lowRes_positions;

    // Laplacianスムージング
    for (size_t i = 0; i < numLowResParticles; i++) {
        if (isAnchoredToParent[i]) continue;

        glm::vec3 avgPos(0.0f);
        int validCount = 0;

        // アンカーされた隣接頂点の平均位置
        for (int neighborId : lowResNeighborsCache_[i]) {
            if (isAnchoredToParent[neighborId]) {
                avgPos.x += lowRes_positions[neighborId * 3];
                avgPos.y += lowRes_positions[neighborId * 3 + 1];
                avgPos.z += lowRes_positions[neighborId * 3 + 2];
                validCount++;
            }
        }

        // アンカーされた隣接がない場合は、全隣接頂点を使用
        if (validCount == 0) {
            for (int neighborId : lowResNeighborsCache_[i]) {
                avgPos.x += lowRes_positions[neighborId * 3];
                avgPos.y += lowRes_positions[neighborId * 3 + 1];
                avgPos.z += lowRes_positions[neighborId * 3 + 2];
                validCount++;
            }
        }

        if (validCount > 0) {
            avgPos /= static_cast<float>(validCount);

            glm::vec3 currentPos(
                lowRes_positions[i * 3],
                lowRes_positions[i * 3 + 1],
                lowRes_positions[i * 3 + 2]
            );

            glm::vec3 prevPos(
                lowRes_prevPositions[i * 3],
                lowRes_prevPositions[i * 3 + 1],
                lowRes_prevPositions[i * 3 + 2]
            );

            // 速度（前フレームからの変位）
            glm::vec3 velocity = currentPos - prevPos;

            // Laplacianで目標位置を計算
            glm::vec3 laplacianTarget = currentPos * (1.0f - laplacianFactor) + avgPos * laplacianFactor;

            // 新しい位置 = 目標位置 + ダンピングされた速度
            glm::vec3 newPos = laplacianTarget + velocity * dampingFactor;

            lowRes_positions[i * 3]     = newPos.x;
            lowRes_positions[i * 3 + 1] = newPos.y;
            lowRes_positions[i * 3 + 2] = newPos.z;
        }
    }

    // prevPositionsを更新
    for (size_t i = 0; i < numLowResParticles; i++) {
        if (!isAnchoredToParent[i]) {
            lowRes_prevPositions[i * 3]     = oldPositions[i * 3];
            lowRes_prevPositions[i * 3 + 1] = oldPositions[i * 3 + 1];
            lowRes_prevPositions[i * 3 + 2] = oldPositions[i * 3 + 2];

            lowRes_velocities[i * 3]   = 0.0f;
            lowRes_velocities[i * 3 + 1] = 0.0f;
            lowRes_velocities[i * 3 + 2] = 0.0f;
        }
    }
}


// =============================================================================
// 方法4: solveFreeVerticesWithConstraints() - XPBD風
// 振動: あり  形状保持: 強い
// 特徴: エッジ制約と体積制約で元の形状を強く保持
// =============================================================================

void SoftBodyGPUDuo::solveFreeVerticesWithConstraints() {
    if (parentSoftBody == nullptr) return;

    // ========================================
    // キャッシュ構築（初回のみ）
    // ========================================
    if (!lowResConstraintsCacheBuilt_) {
        // 隣接リスト
        lowResNeighborsCache_.clear();
        lowResNeighborsCache_.resize(numLowResParticles);

        for (size_t e = 0; e < lowRes_edgeIds.size() / 2; e++) {
            int id0 = lowRes_edgeIds[e * 2];
            int id1 = lowRes_edgeIds[e * 2 + 1];
            if (id0 >= 0 && id0 < static_cast<int>(numLowResParticles) &&
                id1 >= 0 && id1 < static_cast<int>(numLowResParticles)) {
                lowResNeighborsCache_[id0].push_back(id1);
                lowResNeighborsCache_[id1].push_back(id0);
            }
        }
        lowResNeighborsCacheBuilt_ = true;

        // エッジのレスト長
        size_t numEdges = lowRes_edgeIds.size() / 2;
        lowResEdgeRestLengths_.resize(numEdges);

        for (size_t e = 0; e < numEdges; e++) {
            int id0 = lowRes_edgeIds[e * 2];
            int id1 = lowRes_edgeIds[e * 2 + 1];

            if (id0 >= 0 && id0 < static_cast<int>(numLowResParticles) &&
                id1 >= 0 && id1 < static_cast<int>(numLowResParticles)) {

                glm::vec3 p0(lowResMeshData.verts[id0 * 3],
                             lowResMeshData.verts[id0 * 3 + 1],
                             lowResMeshData.verts[id0 * 3 + 2]);
                glm::vec3 p1(lowResMeshData.verts[id1 * 3],
                             lowResMeshData.verts[id1 * 3 + 1],
                             lowResMeshData.verts[id1 * 3 + 2]);

                lowResEdgeRestLengths_[e] = glm::distance(p0, p1);
            }
        }

        // 四面体のレスト体積
        lowResTetRestVolumes_.resize(numLowTets);

        for (size_t t = 0; t < numLowTets; t++) {
            int id0 = lowRes_tetIds[t * 4 + 0];
            int id1 = lowRes_tetIds[t * 4 + 1];
            int id2 = lowRes_tetIds[t * 4 + 2];
            int id3 = lowRes_tetIds[t * 4 + 3];

            glm::vec3 p0(lowResMeshData.verts[id0 * 3],
                         lowResMeshData.verts[id0 * 3 + 1],
                         lowResMeshData.verts[id0 * 3 + 2]);
            glm::vec3 p1(lowResMeshData.verts[id1 * 3],
                         lowResMeshData.verts[id1 * 3 + 1],
                         lowResMeshData.verts[id1 * 3 + 2]);
            glm::vec3 p2(lowResMeshData.verts[id2 * 3],
                         lowResMeshData.verts[id2 * 3 + 1],
                         lowResMeshData.verts[id2 * 3 + 2]);
            glm::vec3 p3(lowResMeshData.verts[id3 * 3],
                         lowResMeshData.verts[id3 * 3 + 1],
                         lowResMeshData.verts[id3 * 3 + 2]);

            lowResTetRestVolumes_[t] = computeSignedVolume(p0, p1, p2, p3);
        }

        lowResConstraintsCacheBuilt_ = true;
    }

    // ========================================
    // 自由頂点を判定
    // ========================================
    std::vector<bool> isFreeVertex(numLowResParticles, false);
    int freeCount = 0;

    for (size_t i = 0; i < numLowResParticles; i++) {
        if (!isAnchoredToParent[i]) {
            isFreeVertex[i] = true;
            freeCount++;
        }
    }

    if (freeCount == 0) return;

    // ========================================
    // パラメータ
    // ========================================
    const int maxIterations = 3;
    const float laplacianFactor = 0.5f;
    const float edgeFactor = 0.3f;
    const float volumeFactor = 0.5f;

    // ========================================
    // メインループ
    // ========================================
    for (int iter = 0; iter < maxIterations; iter++) {

        // ==========================================
        // Phase 1: Laplacianスムージング
        // ==========================================
        std::vector<float> newPositions = lowRes_positions;

        for (size_t i = 0; i < numLowResParticles; i++) {
            if (!isFreeVertex[i]) continue;

            glm::vec3 avgPos(0.0f);
            int validCount = 0;

            // アンカーされた隣接頂点の平均
            for (int neighborId : lowResNeighborsCache_[i]) {
                if (!isFreeVertex[neighborId]) {
                    avgPos.x += lowRes_positions[neighborId * 3];
                    avgPos.y += lowRes_positions[neighborId * 3 + 1];
                    avgPos.z += lowRes_positions[neighborId * 3 + 2];
                    validCount++;
                }
            }

            if (validCount > 0) {
                avgPos /= static_cast<float>(validCount);

                glm::vec3 oldPos(lowRes_positions[i * 3],
                                 lowRes_positions[i * 3 + 1],
                                 lowRes_positions[i * 3 + 2]);

                glm::vec3 newPos = oldPos * (1.0f - laplacianFactor) + avgPos * laplacianFactor;

                newPositions[i * 3]     = newPos.x;
                newPositions[i * 3 + 1] = newPos.y;
                newPositions[i * 3 + 2] = newPos.z;
            }
        }

        lowRes_positions = newPositions;

        // ==========================================
        // Phase 2: エッジ長制約
        // ==========================================
        size_t numEdges = lowRes_edgeIds.size() / 2;

        for (size_t e = 0; e < numEdges; e++) {
            int id0 = lowRes_edgeIds[e * 2];
            int id1 = lowRes_edgeIds[e * 2 + 1];

            if (id0 < 0 || id0 >= static_cast<int>(numLowResParticles) ||
                id1 < 0 || id1 >= static_cast<int>(numLowResParticles)) {
                continue;
            }

            if (!isFreeVertex[id0] && !isFreeVertex[id1]) continue;

            glm::vec3 p0(lowRes_positions[id0 * 3],
                         lowRes_positions[id0 * 3 + 1],
                         lowRes_positions[id0 * 3 + 2]);
            glm::vec3 p1(lowRes_positions[id1 * 3],
                         lowRes_positions[id1 * 3 + 1],
                         lowRes_positions[id1 * 3 + 2]);

            float restLength = lowResEdgeRestLengths_[e];
            float currentLength = glm::distance(p0, p1);

            if (currentLength < 0.0001f || restLength < 0.0001f) continue;

            glm::vec3 dir = (p1 - p0) / currentLength;
            float diff = currentLength - restLength;
            glm::vec3 correction = dir * diff * edgeFactor * 0.5f;

            if (isFreeVertex[id0] && isFreeVertex[id1]) {
                lowRes_positions[id0 * 3]     += correction.x;
                lowRes_positions[id0 * 3 + 1] += correction.y;
                lowRes_positions[id0 * 3 + 2] += correction.z;

                lowRes_positions[id1 * 3]     -= correction.x;
                lowRes_positions[id1 * 3 + 1] -= correction.y;
                lowRes_positions[id1 * 3 + 2] -= correction.z;
            } else if (isFreeVertex[id0]) {
                lowRes_positions[id0 * 3]     += correction.x * 2.0f;
                lowRes_positions[id0 * 3 + 1] += correction.y * 2.0f;
                lowRes_positions[id0 * 3 + 2] += correction.z * 2.0f;
            } else if (isFreeVertex[id1]) {
                lowRes_positions[id1 * 3]     -= correction.x * 2.0f;
                lowRes_positions[id1 * 3 + 1] -= correction.y * 2.0f;
                lowRes_positions[id1 * 3 + 2] -= correction.z * 2.0f;
            }
        }

        // ==========================================
        // Phase 3: 体積制約（裏返り防止）
        // ==========================================
        for (size_t t = 0; t < numLowTets; t++) {
            if (!lowRes_tetValid.empty() && !lowRes_tetValid[t]) continue;

            int id0 = lowRes_tetIds[t * 4 + 0];
            int id1 = lowRes_tetIds[t * 4 + 1];
            int id2 = lowRes_tetIds[t * 4 + 2];
            int id3 = lowRes_tetIds[t * 4 + 3];

            bool hasFree = isFreeVertex[id0] || isFreeVertex[id1] ||
                           isFreeVertex[id2] || isFreeVertex[id3];
            if (!hasFree) continue;

            glm::vec3 p0(lowRes_positions[id0 * 3],
                         lowRes_positions[id0 * 3 + 1],
                         lowRes_positions[id0 * 3 + 2]);
            glm::vec3 p1(lowRes_positions[id1 * 3],
                         lowRes_positions[id1 * 3 + 1],
                         lowRes_positions[id1 * 3 + 2]);
            glm::vec3 p2(lowRes_positions[id2 * 3],
                         lowRes_positions[id2 * 3 + 1],
                         lowRes_positions[id2 * 3 + 2]);
            glm::vec3 p3(lowRes_positions[id3 * 3],
                         lowRes_positions[id3 * 3 + 1],
                         lowRes_positions[id3 * 3 + 2]);

            float currentVolume = computeSignedVolume(p0, p1, p2, p3);
            float restVolume = lowResTetRestVolumes_[t];

            float volumeDiff = currentVolume - restVolume;

            if (currentVolume < 0.0001f || std::abs(volumeDiff) > std::abs(restVolume) * 0.5f) {
                glm::vec3 center = (p0 + p1 + p2 + p3) * 0.25f;

                float scale = (restVolume > 0) ?
                              std::cbrt(std::abs(restVolume) / std::max(std::abs(currentVolume), 0.0001f)) : 1.0f;
                scale = glm::clamp(scale, 0.8f, 1.2f);

                auto applyVolumeCorrection = [&](int id, const glm::vec3& pos) {
                    if (!isFreeVertex[id]) return;

                    glm::vec3 toCenter = center - pos;
                    glm::vec3 newPos = center - toCenter * scale;
                    glm::vec3 correction = (newPos - pos) * volumeFactor;

                    lowRes_positions[id * 3]     += correction.x;
                    lowRes_positions[id * 3 + 1] += correction.y;
                    lowRes_positions[id * 3 + 2] += correction.z;
                };

                applyVolumeCorrection(id0, p0);
                applyVolumeCorrection(id1, p1);
                applyVolumeCorrection(id2, p2);
                applyVolumeCorrection(id3, p3);
            }
        }
    }

    // ========================================
    // 速度とprevPositionsを更新
    // ========================================
    for (size_t i = 0; i < numLowResParticles; i++) {
        if (isFreeVertex[i]) {
            lowRes_prevPositions[i * 3]     = lowRes_positions[i * 3];
            lowRes_prevPositions[i * 3 + 1] = lowRes_positions[i * 3 + 1];
            lowRes_prevPositions[i * 3 + 2] = lowRes_positions[i * 3 + 2];

            lowRes_velocities[i * 3]     = 0.0f;
            lowRes_velocities[i * 3 + 1] = 0.0f;
            lowRes_velocities[i * 3 + 2] = 0.0f;
        }
    }
}


void SoftBodyGPUDuo::solveFreeVerticesWithConstraintsStable() {
    if (parentSoftBody == nullptr) return;

    // ========================================
    // キャッシュ構築（省略 - 前と同じ）
    // ========================================
    if (!lowResConstraintsCacheBuilt_) {
        // ... 前と同じ ...
        lowResNeighborsCache_.clear();
        lowResNeighborsCache_.resize(numLowResParticles);

        for (size_t e = 0; e < lowRes_edgeIds.size() / 2; e++) {
            int id0 = lowRes_edgeIds[e * 2];
            int id1 = lowRes_edgeIds[e * 2 + 1];
            if (id0 >= 0 && id0 < static_cast<int>(numLowResParticles) &&
                id1 >= 0 && id1 < static_cast<int>(numLowResParticles)) {
                lowResNeighborsCache_[id0].push_back(id1);
                lowResNeighborsCache_[id1].push_back(id0);
            }
        }
        lowResNeighborsCacheBuilt_ = true;

        size_t numEdges = lowRes_edgeIds.size() / 2;
        lowResEdgeRestLengths_.resize(numEdges);

        for (size_t e = 0; e < numEdges; e++) {
            int id0 = lowRes_edgeIds[e * 2];
            int id1 = lowRes_edgeIds[e * 2 + 1];

            if (id0 >= 0 && id0 < static_cast<int>(numLowResParticles) &&
                id1 >= 0 && id1 < static_cast<int>(numLowResParticles)) {

                glm::vec3 p0(lowResMeshData.verts[id0 * 3],
                             lowResMeshData.verts[id0 * 3 + 1],
                             lowResMeshData.verts[id0 * 3 + 2]);
                glm::vec3 p1(lowResMeshData.verts[id1 * 3],
                             lowResMeshData.verts[id1 * 3 + 1],
                             lowResMeshData.verts[id1 * 3 + 2]);

                lowResEdgeRestLengths_[e] = glm::distance(p0, p1);
            }
        }

        lowResTetRestVolumes_.resize(numLowTets);

        for (size_t t = 0; t < numLowTets; t++) {
            int id0 = lowRes_tetIds[t * 4 + 0];
            int id1 = lowRes_tetIds[t * 4 + 1];
            int id2 = lowRes_tetIds[t * 4 + 2];
            int id3 = lowRes_tetIds[t * 4 + 3];

            glm::vec3 p0(lowResMeshData.verts[id0 * 3],
                         lowResMeshData.verts[id0 * 3 + 1],
                         lowResMeshData.verts[id0 * 3 + 2]);
            glm::vec3 p1(lowResMeshData.verts[id1 * 3],
                         lowResMeshData.verts[id1 * 3 + 1],
                         lowResMeshData.verts[id1 * 3 + 2]);
            glm::vec3 p2(lowResMeshData.verts[id2 * 3],
                         lowResMeshData.verts[id2 * 3 + 1],
                         lowResMeshData.verts[id2 * 3 + 2]);
            glm::vec3 p3(lowResMeshData.verts[id3 * 3],
                         lowResMeshData.verts[id3 * 3 + 1],
                         lowResMeshData.verts[id3 * 3 + 2]);

            lowResTetRestVolumes_[t] = computeSignedVolume(p0, p1, p2, p3);
        }

        lowResConstraintsCacheBuilt_ = true;
    }

    // ========================================
    // 自由頂点を判定
    // ========================================
    std::vector<bool> isFreeVertex(numLowResParticles, false);
    int freeCount = 0;

    for (size_t i = 0; i < numLowResParticles; i++) {
        if (!isAnchoredToParent[i]) {
            isFreeVertex[i] = true;
            freeCount++;
        }
    }

    if (freeCount == 0) return;

    // ========================================
    // パラメータ（調整済み）
    // ========================================
    const int maxIterations = 5;           // 反復回数を増やす
    const float laplacianFactor = 0.3f;    // Laplacianを弱める
    const float edgeFactor = 0.1f;         // エッジ制約を弱める（振動防止）
    const float volumeFactor = 0.2f;       // 体積制約を弱める
    const float temporalBlend = 0.7f;      // 時間的スムージング

    // ========================================
    // 前フレームの位置を保存（時間的スムージング用）
    // ========================================
    std::vector<float> prevFramePositions = lowRes_positions;

    // ========================================
    // メインループ
    // ========================================
    for (int iter = 0; iter < maxIterations; iter++) {

        // ==========================================
        // Phase 1: Laplacianスムージング
        // ==========================================
        std::vector<float> newPositions = lowRes_positions;

        for (size_t i = 0; i < numLowResParticles; i++) {
            if (!isFreeVertex[i]) continue;

            glm::vec3 avgPos(0.0f);
            int validCount = 0;

            for (int neighborId : lowResNeighborsCache_[i]) {
                if (!isFreeVertex[neighborId]) {
                    avgPos.x += lowRes_positions[neighborId * 3];
                    avgPos.y += lowRes_positions[neighborId * 3 + 1];
                    avgPos.z += lowRes_positions[neighborId * 3 + 2];
                    validCount++;
                }
            }

            if (validCount > 0) {
                avgPos /= static_cast<float>(validCount);

                glm::vec3 oldPos(lowRes_positions[i * 3],
                                 lowRes_positions[i * 3 + 1],
                                 lowRes_positions[i * 3 + 2]);

                glm::vec3 newPos = oldPos * (1.0f - laplacianFactor) + avgPos * laplacianFactor;

                newPositions[i * 3]     = newPos.x;
                newPositions[i * 3 + 1] = newPos.y;
                newPositions[i * 3 + 2] = newPos.z;
            }
        }

        lowRes_positions = newPositions;

        // ==========================================
        // Phase 2: エッジ長制約（Jacobiスタイル）
        // ★ポイント: 補正を蓄積してから一括適用
        // ==========================================
        std::vector<glm::vec3> corrections(numLowResParticles, glm::vec3(0.0f));
        std::vector<int> correctionCounts(numLowResParticles, 0);

        size_t numEdges = lowRes_edgeIds.size() / 2;

        for (size_t e = 0; e < numEdges; e++) {
            int id0 = lowRes_edgeIds[e * 2];
            int id1 = lowRes_edgeIds[e * 2 + 1];

            if (id0 < 0 || id0 >= static_cast<int>(numLowResParticles) ||
                id1 < 0 || id1 >= static_cast<int>(numLowResParticles)) {
                continue;
            }

            if (!isFreeVertex[id0] && !isFreeVertex[id1]) continue;

            glm::vec3 p0(lowRes_positions[id0 * 3],
                         lowRes_positions[id0 * 3 + 1],
                         lowRes_positions[id0 * 3 + 2]);
            glm::vec3 p1(lowRes_positions[id1 * 3],
                         lowRes_positions[id1 * 3 + 1],
                         lowRes_positions[id1 * 3 + 2]);

            float restLength = lowResEdgeRestLengths_[e];
            float currentLength = glm::distance(p0, p1);

            if (currentLength < 0.0001f || restLength < 0.0001f) continue;

            glm::vec3 dir = (p1 - p0) / currentLength;
            float diff = currentLength - restLength;

            // 補正量を計算（適用はまだしない）
            glm::vec3 correction = dir * diff * edgeFactor * 0.5f;

            if (isFreeVertex[id0] && isFreeVertex[id1]) {
                corrections[id0] += correction;
                corrections[id1] -= correction;
                correctionCounts[id0]++;
                correctionCounts[id1]++;
            } else if (isFreeVertex[id0]) {
                corrections[id0] += correction * 2.0f;
                correctionCounts[id0]++;
            } else if (isFreeVertex[id1]) {
                corrections[id1] -= correction * 2.0f;
                correctionCounts[id1]++;
            }
        }

        // 補正を一括適用（Jacobi）
        for (size_t i = 0; i < numLowResParticles; i++) {
            if (isFreeVertex[i] && correctionCounts[i] > 0) {
                // 平均化して適用（複数の制約からの補正を平均）
                glm::vec3 avgCorrection = corrections[i] / static_cast<float>(correctionCounts[i]);
                lowRes_positions[i * 3]     += avgCorrection.x;
                lowRes_positions[i * 3 + 1] += avgCorrection.y;
                lowRes_positions[i * 3 + 2] += avgCorrection.z;
            }
        }

        // ==========================================
        // Phase 3: 体積制約（裏返り防止のみ）
        // ★ポイント: 体積が負の時だけ補正
        // ==========================================
        for (size_t t = 0; t < numLowTets; t++) {
            if (!lowRes_tetValid.empty() && !lowRes_tetValid[t]) continue;

            int id0 = lowRes_tetIds[t * 4 + 0];
            int id1 = lowRes_tetIds[t * 4 + 1];
            int id2 = lowRes_tetIds[t * 4 + 2];
            int id3 = lowRes_tetIds[t * 4 + 3];

            bool hasFree = isFreeVertex[id0] || isFreeVertex[id1] ||
                           isFreeVertex[id2] || isFreeVertex[id3];
            if (!hasFree) continue;

            glm::vec3 p0(lowRes_positions[id0 * 3],
                         lowRes_positions[id0 * 3 + 1],
                         lowRes_positions[id0 * 3 + 2]);
            glm::vec3 p1(lowRes_positions[id1 * 3],
                         lowRes_positions[id1 * 3 + 1],
                         lowRes_positions[id1 * 3 + 2]);
            glm::vec3 p2(lowRes_positions[id2 * 3],
                         lowRes_positions[id2 * 3 + 1],
                         lowRes_positions[id2 * 3 + 2]);
            glm::vec3 p3(lowRes_positions[id3 * 3],
                         lowRes_positions[id3 * 3 + 1],
                         lowRes_positions[id3 * 3 + 2]);

            float currentVolume = computeSignedVolume(p0, p1, p2, p3);
            float restVolume = lowResTetRestVolumes_[t];

            // ★体積が負（裏返り）の場合のみ補正
            if (currentVolume < 0.001f && restVolume > 0.001f) {
                glm::vec3 center = (p0 + p1 + p2 + p3) * 0.25f;

                auto applyVolumeCorrection = [&](int id, const glm::vec3& pos) {
                    if (!isFreeVertex[id]) return;

                    // 重心から外向きに押し出す
                    glm::vec3 fromCenter = pos - center;
                    float dist = glm::length(fromCenter);
                    if (dist < 0.0001f) return;

                    glm::vec3 dir = fromCenter / dist;
                    glm::vec3 correction = dir * volumeFactor * 0.1f;  // 小さな補正

                    lowRes_positions[id * 3]     += correction.x;
                    lowRes_positions[id * 3 + 1] += correction.y;
                    lowRes_positions[id * 3 + 2] += correction.z;
                };

                applyVolumeCorrection(id0, p0);
                applyVolumeCorrection(id1, p1);
                applyVolumeCorrection(id2, p2);
                applyVolumeCorrection(id3, p3);
            }
        }
    }

    // ========================================
    // 時間的スムージング（前フレームとブレンド）
    // ★ポイント: 急激な変化を抑制
    // ========================================
    for (size_t i = 0; i < numLowResParticles; i++) {
        if (isFreeVertex[i]) {
            lowRes_positions[i * 3] =
                prevFramePositions[i * 3] * (1.0f - temporalBlend) +
                lowRes_positions[i * 3] * temporalBlend;
            lowRes_positions[i * 3 + 1] =
                prevFramePositions[i * 3 + 1] * (1.0f - temporalBlend) +
                lowRes_positions[i * 3 + 1] * temporalBlend;
            lowRes_positions[i * 3 + 2] =
                prevFramePositions[i * 3 + 2] * (1.0f - temporalBlend) +
                lowRes_positions[i * 3 + 2] * temporalBlend;

            lowRes_prevPositions[i * 3]     = lowRes_positions[i * 3];
            lowRes_prevPositions[i * 3 + 1] = lowRes_positions[i * 3 + 1];
            lowRes_prevPositions[i * 3 + 2] = lowRes_positions[i * 3 + 2];

            lowRes_velocities[i * 3]     = 0.0f;
            lowRes_velocities[i * 3 + 1] = 0.0f;
            lowRes_velocities[i * 3 + 2] = 0.0f;
        }
    }

    if (smoothingDebugMode_) {
        std::cout << "[solveFreeVerticesStable] Free: " << freeCount
                  << " / " << numLowResParticles << std::endl;
    }
}

// =====================================================
// SoftBodyGPUDuo.cpp に追加
// =====================================================

// グラフカラーリングを構築
void SoftBodyGPUDuo::buildLowResColoring() {
    if (lowResColoringBuilt_) return;

    // ----- エッジのカラーリング -----
    int numEdges = static_cast<int>(lowRes_edgeIds.size() / 2);
    lowResEdgeColorGroups_.clear();
    std::vector<int> edgeColors(numEdges, -1);

    // 頂点ごとの隣接エッジ
    std::vector<std::set<int>> vertexEdges(numLowResParticles);
    for (int i = 0; i < numEdges; i++) {
        int v0 = lowRes_edgeIds[i * 2 + 0];
        int v1 = lowRes_edgeIds[i * 2 + 1];
        if (v0 >= 0 && v0 < static_cast<int>(numLowResParticles))
            vertexEdges[v0].insert(i);
        if (v1 >= 0 && v1 < static_cast<int>(numLowResParticles))
            vertexEdges[v1].insert(i);
    }

    // グリーディカラーリング
    for (int i = 0; i < numEdges; i++) {
        int v0 = lowRes_edgeIds[i * 2 + 0];
        int v1 = lowRes_edgeIds[i * 2 + 1];

        std::set<int> usedColors;
        if (v0 >= 0 && v0 < static_cast<int>(numLowResParticles)) {
            for (int neighborEdge : vertexEdges[v0]) {
                if (edgeColors[neighborEdge] >= 0) {
                    usedColors.insert(edgeColors[neighborEdge]);
                }
            }
        }
        if (v1 >= 0 && v1 < static_cast<int>(numLowResParticles)) {
            for (int neighborEdge : vertexEdges[v1]) {
                if (edgeColors[neighborEdge] >= 0) {
                    usedColors.insert(edgeColors[neighborEdge]);
                }
            }
        }

        int color = 0;
        while (usedColors.count(color) > 0) color++;
        edgeColors[i] = color;
    }

    int maxEdgeColor = 0;
    for (int c : edgeColors) maxEdgeColor = std::max(maxEdgeColor, c);
    lowResEdgeColorGroups_.resize(maxEdgeColor + 1);
    for (int i = 0; i < numEdges; i++) {
        if (edgeColors[i] >= 0) {
            lowResEdgeColorGroups_[edgeColors[i]].push_back(i);
        }
    }

    // ----- 四面体のカラーリング -----
    int numTets = static_cast<int>(numLowTets);
    lowResTetColorGroups_.clear();
    std::vector<int> tetColors(numTets, -1);

    // 頂点ごとの隣接四面体
    std::vector<std::set<int>> vertexTets(numLowResParticles);
    for (int t = 0; t < numTets; t++) {
        for (int j = 0; j < 4; j++) {
            int v = lowRes_tetIds[t * 4 + j];
            if (v >= 0 && v < static_cast<int>(numLowResParticles))
                vertexTets[v].insert(t);
        }
    }

    // グリーディカラーリング
    for (int t = 0; t < numTets; t++) {
        std::set<int> usedColors;
        for (int j = 0; j < 4; j++) {
            int v = lowRes_tetIds[t * 4 + j];
            if (v >= 0 && v < static_cast<int>(numLowResParticles)) {
                for (int neighborTet : vertexTets[v]) {
                    if (tetColors[neighborTet] >= 0) {
                        usedColors.insert(tetColors[neighborTet]);
                    }
                }
            }
        }

        int color = 0;
        while (usedColors.count(color) > 0) color++;
        tetColors[t] = color;
    }

    int maxTetColor = 0;
    for (int c : tetColors) maxTetColor = std::max(maxTetColor, c);
    lowResTetColorGroups_.resize(maxTetColor + 1);
    for (int t = 0; t < numTets; t++) {
        if (tetColors[t] >= 0) {
            lowResTetColorGroups_[tetColors[t]].push_back(t);
        }
    }

    lowResColoringBuilt_ = true;

    std::cout << "[SkinningXPBD] Coloring built: "
              << lowResEdgeColorGroups_.size() << " edge colors, "
              << lowResTetColorGroups_.size() << " tet colors" << std::endl;
}




// =====================================================
// SoftBodyGPUDuo.cpp に追加
// =====================================================

SoftBodyGPUDuo::InversionCheckResult SoftBodyGPUDuo::checkLowResInversion() const {
    InversionCheckResult result;
    result.minVolume = std::numeric_limits<float>::max();
    result.maxVolume = std::numeric_limits<float>::lowest();

    for (size_t t = 0; t < numLowTets; t++) {
        // 無効な四面体はスキップ
        if (!lowRes_tetValid.empty() && !lowRes_tetValid[t]) continue;

        int id0 = lowRes_tetIds[t * 4 + 0];
        int id1 = lowRes_tetIds[t * 4 + 1];
        int id2 = lowRes_tetIds[t * 4 + 2];
        int id3 = lowRes_tetIds[t * 4 + 3];

        if (id0 < 0 || id1 < 0 || id2 < 0 || id3 < 0) continue;
        if (id0 >= static_cast<int>(numLowResParticles) ||
            id1 >= static_cast<int>(numLowResParticles) ||
            id2 >= static_cast<int>(numLowResParticles) ||
            id3 >= static_cast<int>(numLowResParticles)) continue;

        result.totalValidTets++;

        glm::vec3 p0(lowRes_positions[id0 * 3],
                     lowRes_positions[id0 * 3 + 1],
                     lowRes_positions[id0 * 3 + 2]);
        glm::vec3 p1(lowRes_positions[id1 * 3],
                     lowRes_positions[id1 * 3 + 1],
                     lowRes_positions[id1 * 3 + 2]);
        glm::vec3 p2(lowRes_positions[id2 * 3],
                     lowRes_positions[id2 * 3 + 1],
                     lowRes_positions[id2 * 3 + 2]);
        glm::vec3 p3(lowRes_positions[id3 * 3],
                     lowRes_positions[id3 * 3 + 1],
                     lowRes_positions[id3 * 3 + 2]);

        // 符号付き体積を計算
        glm::vec3 d1 = p1 - p0;
        glm::vec3 d2 = p2 - p0;
        glm::vec3 d3 = p3 - p0;
        float volume = glm::dot(glm::cross(d1, d2), d3) / 6.0f;

        // 最小・最大を更新
        result.minVolume = std::min(result.minVolume, volume);
        result.maxVolume = std::max(result.maxVolume, volume);

        // 裏返り判定（体積が負または非常に小さい）
        if (volume < 1e-8f) {
            result.invertedCount++;
            result.invertedTetIds.push_back(static_cast<int>(t));
        }
    }

    // 四面体がなかった場合
    if (result.totalValidTets == 0) {
        result.minVolume = 0.0f;
        result.maxVolume = 0.0f;
    }

    return result;
}

bool SoftBodyGPUDuo::hasLowResInversion() const {
    for (size_t t = 0; t < numLowTets; t++) {
        // 無効な四面体はスキップ
        if (!lowRes_tetValid.empty() && !lowRes_tetValid[t]) continue;

        int id0 = lowRes_tetIds[t * 4 + 0];
        int id1 = lowRes_tetIds[t * 4 + 1];
        int id2 = lowRes_tetIds[t * 4 + 2];
        int id3 = lowRes_tetIds[t * 4 + 3];

        if (id0 < 0 || id1 < 0 || id2 < 0 || id3 < 0) continue;
        if (id0 >= static_cast<int>(numLowResParticles) ||
            id1 >= static_cast<int>(numLowResParticles) ||
            id2 >= static_cast<int>(numLowResParticles) ||
            id3 >= static_cast<int>(numLowResParticles)) continue;

        glm::vec3 p0(lowRes_positions[id0 * 3],
                     lowRes_positions[id0 * 3 + 1],
                     lowRes_positions[id0 * 3 + 2]);
        glm::vec3 p1(lowRes_positions[id1 * 3],
                     lowRes_positions[id1 * 3 + 1],
                     lowRes_positions[id1 * 3 + 2]);
        glm::vec3 p2(lowRes_positions[id2 * 3],
                     lowRes_positions[id2 * 3 + 1],
                     lowRes_positions[id2 * 3 + 2]);
        glm::vec3 p3(lowRes_positions[id3 * 3],
                     lowRes_positions[id3 * 3 + 1],
                     lowRes_positions[id3 * 3 + 2]);

        glm::vec3 d1 = p1 - p0;
        glm::vec3 d2 = p2 - p0;
        glm::vec3 d3 = p3 - p0;
        float volume = glm::dot(glm::cross(d1, d2), d3) / 6.0f;

        if (volume < 1e-8f) {
            return true;  // 裏返りあり
        }
    }

    return false;  // 裏返りなし
}

//=============================================================================
// 子メッシュカット時のアンカー管理
//=============================================================================

bool SoftBodyGPUDuo::hasValidTetContainingVertex(int vertexId) const {
    if (vertexId < 0 || static_cast<size_t>(vertexId) >= numLowResParticles) {
        return false;
    }

    for (size_t t = 0; t < numLowTets; t++) {
        if (!lowRes_tetValid[t]) continue;

        for (int j = 0; j < 4; j++) {
            if (lowRes_tetIds[t * 4 + j] == vertexId) {
                return true;
            }
        }
    }
    return false;
}

std::vector<int> SoftBodyGPUDuo::releaseAnchorsForOrphanedVertices() {
    std::vector<int> releasedVertices;

    // 親がない場合は何もしない
    if (!parentSoftBody) {
        return releasedVertices;
    }

    // isAnchoredToParentが空の場合も何もしない
    if (isAnchoredToParent.empty()) {
        return releasedVertices;
    }

    std::cout << "\n[DEBUG] === releaseAnchorsForOrphanedVertices ===" << std::endl;
    std::cout << "  Total particles: " << numLowResParticles << std::endl;
    std::cout << "  Anchored before: " << numAnchoredVertices << std::endl;

    for (size_t i = 0; i < numLowResParticles; i++) {
        // アンカーされていない頂点はスキップ
        if (!isAnchoredToParent[i]) continue;

        // この頂点を含む有効な四面体があるか確認
        if (!hasValidTetContainingVertex(static_cast<int>(i))) {
            // 孤立した頂点：アンカー解除
            isAnchoredToParent[i] = false;

            // 質量を復元（originalInvMassesがある場合）
            if (i < originalInvMasses.size() && originalInvMasses[i] > 0.0f) {
                lowRes_invMasses[i] = originalInvMasses[i];
            } else {
                // originalInvMassesがない場合はデフォルト値
                lowRes_invMasses[i] = 1.0f;
            }

            numAnchoredVertices--;
            releasedVertices.push_back(static_cast<int>(i));
        }
    }

    std::cout << "  Released vertices: " << releasedVertices.size() << std::endl;
    std::cout << "  Anchored after: " << numAnchoredVertices << std::endl;
    std::cout << "============================================\n" << std::endl;

    return releasedVertices;
}

void SoftBodyGPUDuo::restoreAnchorsForRestoredVertices(
    const std::vector<int>& restoredTets,
    const std::vector<bool>& savedAnchorState,
    const std::vector<float>& savedInvMasses)
{
    // 親がない場合は何もしない
    if (!parentSoftBody) {
        return;
    }

    // サイズチェック
    if (savedAnchorState.size() != numLowResParticles ||
        savedInvMasses.size() != numLowResParticles) {
        std::cout << "[WARNING] restoreAnchorsForRestoredVertices: size mismatch" << std::endl;
        return;
    }

    std::cout << "\n[DEBUG] === restoreAnchorsForRestoredVertices ===" << std::endl;
    std::cout << "  Restored tets: " << restoredTets.size() << std::endl;
    std::cout << "  Anchored before: " << numAnchoredVertices << std::endl;

    // 復元された四面体に含まれる頂点を収集
    std::set<int> affectedVertices;
    for (int tetIdx : restoredTets) {
        if (tetIdx < 0 || static_cast<size_t>(tetIdx) >= numLowTets) continue;

        for (int j = 0; j < 4; j++) {
            int vid = lowRes_tetIds[tetIdx * 4 + j];
            affectedVertices.insert(vid);
        }
    }

    // 影響を受けた頂点のアンカー状態を復元
    int restoredCount = 0;
    for (int vid : affectedVertices) {
        if (vid < 0 || static_cast<size_t>(vid) >= numLowResParticles) continue;

        // 元々アンカーされていた場合は復元
        if (savedAnchorState[vid] && !isAnchoredToParent[vid]) {
            isAnchoredToParent[vid] = true;
            lowRes_invMasses[vid] = savedInvMasses[vid];  // 通常は0.0f
            numAnchoredVertices++;
            restoredCount++;
        }
    }

    std::cout << "  Restored anchors: " << restoredCount << std::endl;
    std::cout << "  Anchored after: " << numAnchoredVertices << std::endl;
    std::cout << "================================================\n" << std::endl;
}



void SoftBodyGPUDuo::solveFreeVerticesXPBD() {
    if (parentSoftBody == nullptr) return;

    // グラフカラーリングを構築（初回のみ）
    buildLowResColoring();

    // レスト長・レスト体積のキャッシュを構築（初回のみ）
    if (!lowResConstraintsCacheBuilt_) {
        // エッジのレスト長
        size_t numEdges = lowRes_edgeIds.size() / 2;
        lowResEdgeRestLengths_.resize(numEdges);

        for (size_t e = 0; e < numEdges; e++) {
            int id0 = lowRes_edgeIds[e * 2];
            int id1 = lowRes_edgeIds[e * 2 + 1];

            if (id0 >= 0 && id0 < static_cast<int>(numLowResParticles) &&
                id1 >= 0 && id1 < static_cast<int>(numLowResParticles)) {

                glm::vec3 p0(lowResMeshData.verts[id0 * 3],
                             lowResMeshData.verts[id0 * 3 + 1],
                             lowResMeshData.verts[id0 * 3 + 2]);
                glm::vec3 p1(lowResMeshData.verts[id1 * 3],
                             lowResMeshData.verts[id1 * 3 + 1],
                             lowResMeshData.verts[id1 * 3 + 2]);

                lowResEdgeRestLengths_[e] = glm::distance(p0, p1);
            } else {
                lowResEdgeRestLengths_[e] = 0.0f;
            }
        }

        // 四面体のレスト体積
        lowResTetRestVolumes_.resize(numLowTets);

        for (size_t t = 0; t < numLowTets; t++) {
            int id0 = lowRes_tetIds[t * 4 + 0];
            int id1 = lowRes_tetIds[t * 4 + 1];
            int id2 = lowRes_tetIds[t * 4 + 2];
            int id3 = lowRes_tetIds[t * 4 + 3];

            glm::vec3 p0(lowResMeshData.verts[id0 * 3],
                         lowResMeshData.verts[id0 * 3 + 1],
                         lowResMeshData.verts[id0 * 3 + 2]);
            glm::vec3 p1(lowResMeshData.verts[id1 * 3],
                         lowResMeshData.verts[id1 * 3 + 1],
                         lowResMeshData.verts[id1 * 3 + 2]);
            glm::vec3 p2(lowResMeshData.verts[id2 * 3],
                         lowResMeshData.verts[id2 * 3 + 1],
                         lowResMeshData.verts[id2 * 3 + 2]);
            glm::vec3 p3(lowResMeshData.verts[id3 * 3],
                         lowResMeshData.verts[id3 * 3 + 1],
                         lowResMeshData.verts[id3 * 3 + 2]);

            glm::vec3 d1 = p1 - p0;
            glm::vec3 d2 = p2 - p0;
            glm::vec3 d3 = p3 - p0;
            lowResTetRestVolumes_[t] = glm::dot(glm::cross(d1, d2), d3) / 6.0f;
        }

        lowResConstraintsCacheBuilt_ = true;
    }

    // ========================================
    // Step 1: invMassを設定
    // ========================================
    std::vector<float> tempInvMass(numLowResParticles);
    int freeCount = 0;

    for (size_t i = 0; i < numLowResParticles; i++) {
        if (isAnchoredToParent[i]) {
            tempInvMass[i] = 0.0f;
        } else {
            tempInvMass[i] = lowRes_invMasses[i];
            //tempInvMass[i] = 1.0f;
            freeCount++;
        }
    }

    if (freeCount == 0) return;

    // ========================================
    // Step 2: 隣接キャッシュを構築（初回のみ）★最適化★
    // ========================================
    if (!lowResNeighborsCacheBuilt_) {
        lowResNeighborsCache_.resize(numLowResParticles);
        for (auto& neighbors : lowResNeighborsCache_) {
            neighbors.clear();
        }

        size_t numEdges = lowRes_edgeIds.size() / 2;
        for (size_t e = 0; e < numEdges; e++) {
            int id0 = lowRes_edgeIds[e * 2];
            int id1 = lowRes_edgeIds[e * 2 + 1];

            if (id0 >= 0 && id0 < static_cast<int>(numLowResParticles) &&
                id1 >= 0 && id1 < static_cast<int>(numLowResParticles)) {
                lowResNeighborsCache_[id0].push_back(id1);
                lowResNeighborsCache_[id1].push_back(id0);
            }
        }
        lowResNeighborsCacheBuilt_ = true;
    }

    // ========================================
    // Step 3: 初期位置をLaplacianスムージングで設定 ★最適化★
    // ========================================
    for (size_t i = 0; i < numLowResParticles; i++) {
        if (isAnchoredToParent[i]) continue;

        glm::vec3 avgPos(0.0f);
        int validCount = 0;

        // ★最適化：隣接キャッシュを使用（O(隣接数)）
        for (int neighborId : lowResNeighborsCache_[i]) {
            if (isAnchoredToParent[neighborId]) {
                avgPos.x += lowRes_positions[neighborId * 3];
                avgPos.y += lowRes_positions[neighborId * 3 + 1];
                avgPos.z += lowRes_positions[neighborId * 3 + 2];
                validCount++;
            }
        }

        if (validCount > 0) {
            avgPos /= static_cast<float>(validCount);

            glm::vec3 currentPos(
                lowRes_positions[i * 3],
                lowRes_positions[i * 3 + 1],
                lowRes_positions[i * 3 + 2]
            );

            glm::vec3 newPos = currentPos * 0.5f + avgPos * 0.5f;
            lowRes_positions[i * 3]     = newPos.x;
            lowRes_positions[i * 3 + 1] = newPos.y;
            lowRes_positions[i * 3 + 2] = newPos.z;
        }
    }

    // ========================================
    // Step 4: XPBDソルバー（GSスタイル）
    // ========================================
    const int numIterations = skinningXPBDParams_.numIterations;
    const float dt = 1.0f / 60.0f;
    const float edgeAlpha = skinningXPBDParams_.edgeCompliance / (dt * dt);
    const float volumeAlpha = skinningXPBDParams_.volumeCompliance / (dt * dt);

    size_t numEdges = lowRes_edgeIds.size() / 2;
    std::vector<float> edgeLambdas(numEdges, 0.0f);
    std::vector<float> volumeLambdas(numLowTets, 0.0f);

    for (int iter = 0; iter < numIterations; iter++) {

        // ----- エッジ制約 -----
        for (const auto& colorGroup : lowResEdgeColorGroups_) {
            int groupSize = static_cast<int>(colorGroup.size());

            #pragma omp parallel for
            for (int i = 0; i < groupSize; i++) {
                int edgeIdx = colorGroup[i];

                int id0 = lowRes_edgeIds[edgeIdx * 2 + 0];
                int id1 = lowRes_edgeIds[edgeIdx * 2 + 1];

                if (id0 < 0 || id0 >= static_cast<int>(numLowResParticles) ||
                    id1 < 0 || id1 >= static_cast<int>(numLowResParticles)) continue;

                float w0 = tempInvMass[id0];
                float w1 = tempInvMass[id1];
                float wSum = w0 + w1;
                if (wSum == 0.0f) continue;

                float dx = lowRes_positions[id1 * 3 + 0] - lowRes_positions[id0 * 3 + 0];
                float dy = lowRes_positions[id1 * 3 + 1] - lowRes_positions[id0 * 3 + 1];
                float dz = lowRes_positions[id1 * 3 + 2] - lowRes_positions[id0 * 3 + 2];
                float len = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (len < 1e-7f) continue;

                float invLen = 1.0f / len;
                dx *= invLen; dy *= invLen; dz *= invLen;

                float restLen = lowResEdgeRestLengths_[edgeIdx];
                float C = len - restLen;
                float dLambda = -(C + edgeAlpha * edgeLambdas[edgeIdx]) / (wSum + edgeAlpha);
                edgeLambdas[edgeIdx] += dLambda;

                lowRes_positions[id0 * 3 + 0] -= w0 * dx * dLambda;
                lowRes_positions[id0 * 3 + 1] -= w0 * dy * dLambda;
                lowRes_positions[id0 * 3 + 2] -= w0 * dz * dLambda;
                lowRes_positions[id1 * 3 + 0] += w1 * dx * dLambda;
                lowRes_positions[id1 * 3 + 1] += w1 * dy * dLambda;
                lowRes_positions[id1 * 3 + 2] += w1 * dz * dLambda;
            }
        }

        // ----- 体積制約 -----
        for (const auto& colorGroup : lowResTetColorGroups_) {
            int groupSize = static_cast<int>(colorGroup.size());

            #pragma omp parallel for
            for (int i = 0; i < groupSize; i++) {
                int tetIdx = colorGroup[i];

                if (!lowRes_tetValid.empty() && !lowRes_tetValid[tetIdx]) continue;

                int id0 = lowRes_tetIds[tetIdx * 4 + 0];
                int id1 = lowRes_tetIds[tetIdx * 4 + 1];
                int id2 = lowRes_tetIds[tetIdx * 4 + 2];
                int id3 = lowRes_tetIds[tetIdx * 4 + 3];

                if (id0 < 0 || id1 < 0 || id2 < 0 || id3 < 0) continue;

                float w0 = tempInvMass[id0];
                float w1 = tempInvMass[id1];
                float w2 = tempInvMass[id2];
                float w3 = tempInvMass[id3];

                if (w0 == 0 && w1 == 0 && w2 == 0 && w3 == 0) continue;

                glm::vec3 p0(lowRes_positions[id0 * 3],
                             lowRes_positions[id0 * 3 + 1],
                             lowRes_positions[id0 * 3 + 2]);
                glm::vec3 p1(lowRes_positions[id1 * 3],
                             lowRes_positions[id1 * 3 + 1],
                             lowRes_positions[id1 * 3 + 2]);
                glm::vec3 p2(lowRes_positions[id2 * 3],
                             lowRes_positions[id2 * 3 + 1],
                             lowRes_positions[id2 * 3 + 2]);
                glm::vec3 p3(lowRes_positions[id3 * 3],
                             lowRes_positions[id3 * 3 + 1],
                             lowRes_positions[id3 * 3 + 2]);

                glm::vec3 d1 = p1 - p0;
                glm::vec3 d2 = p2 - p0;
                glm::vec3 d3 = p3 - p0;
                float volume = glm::dot(glm::cross(d1, d2), d3) / 6.0f;

                float restVolume = lowResTetRestVolumes_[tetIdx];
                float C = volume - restVolume;

                glm::vec3 grad0 = glm::cross(p1 - p2, p3 - p2) / 6.0f;
                glm::vec3 grad1 = glm::cross(p2 - p0, p3 - p0) / 6.0f;
                glm::vec3 grad2 = glm::cross(p0 - p1, p3 - p1) / 6.0f;
                glm::vec3 grad3 = glm::cross(p1 - p0, p2 - p0) / 6.0f;

                float wGrad = w0 * glm::dot(grad0, grad0) +
                              w1 * glm::dot(grad1, grad1) +
                              w2 * glm::dot(grad2, grad2) +
                              w3 * glm::dot(grad3, grad3);

                if (wGrad < 1e-10f) continue;

                float dLambda = -(C + volumeAlpha * volumeLambdas[tetIdx]) / (wGrad + volumeAlpha);
                volumeLambdas[tetIdx] += dLambda;

                lowRes_positions[id0 * 3 + 0] += w0 * grad0.x * dLambda;
                lowRes_positions[id0 * 3 + 1] += w0 * grad0.y * dLambda;
                lowRes_positions[id0 * 3 + 2] += w0 * grad0.z * dLambda;

                lowRes_positions[id1 * 3 + 0] += w1 * grad1.x * dLambda;
                lowRes_positions[id1 * 3 + 1] += w1 * grad1.y * dLambda;
                lowRes_positions[id1 * 3 + 2] += w1 * grad1.z * dLambda;

                lowRes_positions[id2 * 3 + 0] += w2 * grad2.x * dLambda;
                lowRes_positions[id2 * 3 + 1] += w2 * grad2.y * dLambda;
                lowRes_positions[id2 * 3 + 2] += w2 * grad2.z * dLambda;

                lowRes_positions[id3 * 3 + 0] += w3 * grad3.x * dLambda;
                lowRes_positions[id3 * 3 + 1] += w3 * grad3.y * dLambda;
                lowRes_positions[id3 * 3 + 2] += w3 * grad3.z * dLambda;
            }
        }
    }

    // ========================================
    // Step 5: 速度をゼロに
    // ========================================
    for (size_t i = 0; i < numLowResParticles; i++) {
        if (!isAnchoredToParent[i]) {
            lowRes_prevPositions[i * 3]     = lowRes_positions[i * 3];
            lowRes_prevPositions[i * 3 + 1] = lowRes_positions[i * 3 + 1];
            lowRes_prevPositions[i * 3 + 2] = lowRes_positions[i * 3 + 2];

            lowRes_velocities[i * 3]     = 0.0f;
            lowRes_velocities[i * 3 + 1] = 0.0f;
            lowRes_velocities[i * 3 + 2] = 0.0f;
        }
    }
}


void SoftBodyGPUDuo::buildHighResTetAdjacency() {
    if (highResTetAdjacencyBuilt) return;

    auto startTime = std::chrono::high_resolution_clock::now();

    size_t numTets = highResMeshData.tetIds.size() / 4;

    // 面→四面体のマップを構築
    std::map<std::array<int, 3>, std::vector<int>> faceToTets;

    for (size_t tetIdx = 0; tetIdx < numTets; tetIdx++) {
        // 四面体の4つの面
        std::array<std::array<int, 3>, 4> faces = {{
            {highResMeshData.tetIds[tetIdx*4+0], highResMeshData.tetIds[tetIdx*4+1], highResMeshData.tetIds[tetIdx*4+2]},
            {highResMeshData.tetIds[tetIdx*4+0], highResMeshData.tetIds[tetIdx*4+1], highResMeshData.tetIds[tetIdx*4+3]},
            {highResMeshData.tetIds[tetIdx*4+0], highResMeshData.tetIds[tetIdx*4+2], highResMeshData.tetIds[tetIdx*4+3]},
            {highResMeshData.tetIds[tetIdx*4+1], highResMeshData.tetIds[tetIdx*4+2], highResMeshData.tetIds[tetIdx*4+3]}
        }};

        for (auto& face : faces) {
            std::sort(face.begin(), face.end());
            faceToTets[face].push_back(tetIdx);
        }
    }

    // 隣接リスト構築
    highResTetAdjacency.clear();
    highResTetAdjacency.resize(numTets);

    size_t totalEdges = 0;
    for (const auto& pair : faceToTets) {
        if (pair.second.size() == 2) {
            int t1 = pair.second[0];
            int t2 = pair.second[1];
            highResTetAdjacency[t1].push_back(t2);
            highResTetAdjacency[t2].push_back(t1);
            totalEdges++;
        }
    }

    highResTetAdjacencyBuilt = true;

    auto endTime = std::chrono::high_resolution_clock::now();
    double buildTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    std::cout << "[SoftBody] Adjacency built for " << numTets << " tets" << std::endl;
    std::cout << "  Edges: " << totalEdges << ", Time: " << buildTime << " ms" << std::endl;
}
