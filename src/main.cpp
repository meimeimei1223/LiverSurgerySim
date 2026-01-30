
#include <iostream>
#include <sstream>
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <string>
#include "ShaderProgram.h"
#include <cmath>
#include "mCutMesh.h"
#include <omp.h>
#include "CentVoxTetrahedralizerHybrid.h"
// #include <fcntl.h>  // 削除: 未使用
#include <filesystem>
#include "CutterMeshManager.h"
#include "SoftBodyGPUDuo.h"
#include "RayCastGPUDuo.h"
#include "GrabberGPUDuo.h"
#include "CentVoxTetrahedralizerHybrid.h"
#include "SphereDeformGPUDuo.h"
#include "MeshDataTypes.h"
#include <thread>
#include "SoftBodyParallelSolver.h"
#include "VoxelSkeletonSegmentation.h"
#include "MeshCuttingGPUDuo.h"
#include "SoftBodyCutManager.h"
#include "CutSegmentManager.h"
#include "simple_multi_obj_processor.h"
#include "FullSphereCamera.h"
#include <cstdlib>  // getenv用
// グローバル変数に追加
#include "CRSlicerCrossSec.h"
// ============================================================================
// パス設定（環境変数対応）
// ============================================================================
std::string MODEL_PATH = "../../../model/";
std::string SHADER_PATH = "../../../shader/";
std::string SHADERS_PATH = "../../../shaders/";
std::string DATA_PATH = "../../../data/";

void initPaths() {
    std::cout << "========================================" << std::endl;
    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;
    std::cout << "========================================" << std::endl;

    // 環境変数があれば優先
    const char* modelEnv = std::getenv("LIVER_MODEL_PATH");
    if (modelEnv) {
        MODEL_PATH = modelEnv;
        if (!MODEL_PATH.empty() && MODEL_PATH.back() != '/') MODEL_PATH += '/';
        std::cout << "[Path] Model (env): " << MODEL_PATH << std::endl;
    } else {
        // 自動検出: 複数のパスを試す
        std::vector<std::string> candidates = {
            "../../../../model/",        // build/Desktop-Release/Release/bin/ から ← 追加！
            "../../../model/",           // build/Release/bin/ から
            "../../model/",
            "../model/",
            "model/",
            "./model/"
        };

        for (const auto& path : candidates) {
            std::string testFile = path + "liver2.obj";
            if (std::filesystem::exists(testFile)) {
                MODEL_PATH = path;
                std::cout << "[Path] Model (auto): " << MODEL_PATH << std::endl;
                break;
            }
        }
    }

    // シェーダーも同様
    std::vector<std::string> shaderCandidates = {
        "../../../../shader/",        // ← 追加！
        "../../../shader/",
        "../../shader/",
        "../shader/",
        "shader/"
    };

    for (const auto& path : shaderCandidates) {
        std::string testFile = path + "basicDuo.vert";
        if (std::filesystem::exists(testFile)) {
            SHADER_PATH = path;
            std::cout << "[Path] Shader (auto): " << SHADER_PATH << std::endl;
            break;
        }
    }

    // shaders
    std::vector<std::string> shadersCandidates = {
        "../../../../shaders/",       // ← 追加！
        "../../../shaders/",
        "../../shaders/",
        "../shaders/",
        "shaders/"
    };

    for (const auto& path : shadersCandidates) {
        std::string testFile = path + "texture.vert";
        if (std::filesystem::exists(testFile)) {
            SHADERS_PATH = path;
            std::cout << "[Path] Shaders (auto): " << SHADERS_PATH << std::endl;
            break;
        }
    }

    // data
    std::vector<std::string> dataCandidates = {
        "../../../../data/",          // ← 追加！
        "../../../data/",
        "../../data/",
        "../data/",
        "data/"
    };

    for (const auto& path : dataCandidates) {
        if (std::filesystem::exists(path)) {
            DATA_PATH = path;
            std::cout << "[Path] Data (auto): " << DATA_PATH << std::endl;
            break;
        }
    }

    std::cout << "========================================" << std::endl;
    std::cout << "Final paths:" << std::endl;
    std::cout << "  MODEL_PATH:   " << MODEL_PATH << std::endl;
    std::cout << "  SHADER_PATH:  " << SHADER_PATH << std::endl;
    std::cout << "  SHADERS_PATH: " << SHADERS_PATH << std::endl;
    std::cout << "  DATA_PATH:    " << DATA_PATH << std::endl;
    std::cout << "========================================" << std::endl;
}

// === グローバル変数 ===
int gWindowWidth = 1024, gWindowHeight = 768;
GLFWwindow* gWindow = NULL;
bool gWireframe = false;
glm::mat4 model(1.0), view(1.0), projection(1.0);
glm::vec3 objPos = glm::vec3(0.0f, 0.0f, 0.0f);

FullSphereCamera OrbitCam;

// グローバル変数の初期化を明示
mCutMesh *cutterMesh = nullptr;

// 前方宣言
void glfw_onKey(GLFWwindow* window, int key, int scancode, int action, int mode);
void glfw_OnFramebufferSize(GLFWwindow* window, int width, int height);
void glfw_onMouseMoveOrbit(GLFWwindow* window, double posX, double posY);
void glfw_onMouseScroll(GLFWwindow* window, double deltaX, double deltaY);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
bool initOpenGL();
void showFPS(GLFWwindow* window);

// ============================================
// グローバル変数に追加
// ============================================
bool cameraMode = false;  // true: カメラモード、false: 操作モード（デフォルトはカメラモード）
bool ignoreNextScroll = false; // ミドルクリック直後のスクロールを無視

bool cutMode = false;

bool performCutOperation = false;

SoftBodyGPUDuo* liver = nullptr;
SoftBodyGPUDuo* portal = nullptr;
SoftBodyGPUDuo* vein = nullptr;
SoftBodyGPUDuo* tumor = nullptr;
SoftBodyGPUDuo* gb = nullptr;

GrabberGPUDuo* gGrabberBunny = nullptr;
CutterMeshState cutterStateGPUDuo;

// フラグ
bool g_showTimingInfo = false;
std::chrono::high_resolution_clock::time_point g_frameStart;

//=============================================================================
// スケルトン統合用グローバル変数
//=============================================================================
VoxelSkeleton::VesselSegmentation* portalSkeleton = nullptr;

bool showSegmentColors = false;

#include "TextRenderer.h"

// グローバル変数
TextRenderer* textRenderer = nullptr;
bool showVolumeOverlay = false;

//=============================================================================
// TextRenderer 初期化
//=============================================================================
void initTextRenderer() {
    textRenderer = new TextRenderer();
    if (!textRenderer->initWithSystemFont(18.0f)) {
        std::cerr << "Failed to initialize text renderer" << std::endl;
        delete textRenderer;
        textRenderer = nullptr;
    } else {
        textRenderer->setWindowSize(gWindowWidth, gWindowHeight);
    }
}

//=============================================================================
// 体積情報オーバーレイ描画
//=============================================================================
void renderVolumeOverlay() {
    if (!textRenderer || !showVolumeOverlay || !liver) return;

    bool useOBJ = portalSkeleton && portalSkeleton->isUsingOBJSegmentation();
    int selectedOBJ = useOBJ ? portalSkeleton->getSelectedOBJSegment() : -1;

    float totalVolume = liver->calculateTotalVolume();
    float totalCm3 = totalVolume / 1000.0f;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);

    oss << "=== Volume Info ===\n";
    oss << "Total: " << totalCm3 << " cm3 (" << (totalCm3/1000.0f) << " L)\n";
    oss << "\n";

    if (useOBJ) {
        //=====================================================================
        // OBJセグメンテーションモード
        //=====================================================================
        oss << "[OBJ Segmentation]\n";

        if (selectedOBJ > 0) {
            float selVol = liver->calculateSegmentVolume(selectedOBJ, true);
            float ratio = (totalVolume > 0) ? (selVol / totalVolume * 100.0f) : 0.0f;
            oss << "Selected: S" << selectedOBJ << "\n";
            oss << "  " << (selVol/1000.0f) << " cm3 (" << ratio << "%)\n";
            oss << "\n";
        }

        oss << "Segments:\n";
        auto volumeMap = liver->calculateAllSegmentVolumes(true);
        for (const auto& kv : volumeMap) {
            float ratio = (totalVolume > 0) ? (kv.second / totalVolume * 100.0f) : 0.0f;
            float cm3 = kv.second / 1000.0f;

            // 選択中のセグメントをハイライト
            if (kv.first == selectedOBJ) {
                oss << " >S" << kv.first << ": " << cm3 << " cm3 (" << ratio << "%)\n";
            } else {
                oss << "  S" << kv.first << ": " << cm3 << " cm3 (" << ratio << "%)\n";
            }
        }

    } else {
        //=====================================================================
        // スケルトンセグメンテーションモード
        //=====================================================================
        oss << "[Skeleton Mode]\n";

        // ★ 選択中のセグメントの体積を表示
        const auto& selectedSegments = liver->getSelectedSegments();

        if (!selectedSegments.empty()) {
            // 選択セグメントの合計体積を計算
            float selectedVolume = 0.0f;
            for (int segId : selectedSegments) {
                selectedVolume += liver->calculateSegmentVolume(segId, false);
            }

            float ratio = (totalVolume > 0) ? (selectedVolume / totalVolume * 100.0f) : 0.0f;
            float selCm3 = selectedVolume / 1000.0f;

            oss << "\n";
            oss << "Selected: " << selectedSegments.size() << " segments\n";
            oss << "  Volume: " << selCm3 << " cm3\n";
            oss << "  Ratio:  " << ratio << "%\n";

            // 選択セグメントIDを表示（最大10個）
            oss << "  IDs: ";
            int count = 0;
            for (int segId : selectedSegments) {
                if (count >= 10) {
                    oss << "...";
                    break;
                }
                if (count > 0) oss << ", ";
                oss << segId;
                count++;
            }
            oss << "\n";
        } else {
            oss << "\n";
            oss << "No segments selected\n";
            oss << "(Shift+Click to select)\n";
        }
    }

    // 背景付きでテキストを描画
    textRenderer->renderTextWithBackground(
        oss.str(),
        10.0f,                               // X座標
        10.0f,                               // Y座標
        1.0f,                                // スケール
        glm::vec3(1.0f, 1.0f, 1.0f),        // テキスト色（白）
        glm::vec4(0.1f, 0.1f, 0.2f, 0.85f), // 背景色（暗い青）
        12.0f,                               // パディング
        1.2f                                 // 行間
        );
}

//=============================================================================
// main関数
//=============================================================================
bool performSegmentCutOperation = false;  // ★ 追加

static double physicsAccumulator = 0.0;
const double PHYSICS_TIME_STEP = 1.0 / 60.0;
int numSubsteps = 5;

int profCount = 0;
double sumPhys = 0, sumSync = 0, sumMesh = 0, sumTotal = 0;

// ★★★ カット後ダンピング用カウンター ★★★
int liverDampingFrame = 3;
int portalDampingFrames = 0;
int veinDampingFrames = 0;

const int POST_CUT_DAMPING_FRAMES = 3;  // カット後60フレームダンピング適用
bool showTimingInfo = true;
float velocityThreshold = 0.0f;

// ★★★ 追加: CPU並列ソルバー ★★★
SoftBodyParallelSolver cpuParallelSolver;
SoftBodyParallelSolver cpuParallelSolver2;
SoftBodyParallelSolver cpuParallelSolver3;

bool useCPUParallel = true;  // CPU並列モードを使うかどうか

// グローバル変数として配列を用意
std::vector<SoftBodyGPUDuo*> softBodies;

SoftBodyCutManager cutManager;
std::vector<CutUndoData> cutUndoHistory;
//const int MAX_CUT_UNDO_HISTORY = 10;
bool performCutUndo = false;
//bool DuoMeshMode = true;

// 追加:
SphereHandleManager sphereManager;  // ★追加

// グローバル変数
CutSegmentManager cutSegmentManager;



CRSlicerCrossSec* crSlicer = nullptr;

bool DuoMeshMode = true;

int main() {
    if (!initOpenGL()) {
        std::cerr << "GLFW initialization failed" << std::endl;
        return -1;
    }

    // ★★★ 最初にパスを初期化 ★★★
    initPaths();

    // シェーダー読み込み（MODEL_PATH使用）
    ShaderProgram shaderProgramGPUDuo;
    shaderProgramGPUDuo.loadShaders(
        (SHADER_PATH + "basicDuo.vert").c_str(),
        (SHADER_PATH + "basicDuo.frag").c_str()
        );

    ShaderProgram shaderProgramCube;
    shaderProgramCube.loadShaders(
        (SHADERS_PATH + "texture.vert").c_str(),
        (SHADERS_PATH + "texture.frag").c_str()
        );

    // メッシュの読み込み
    std::string cutMeshFilePathCube = DATA_PATH + "cube.obj";

    // ファイルパス設定（MODEL_PATH使用）
    std::vector<std::string> ObjTarget = {
        MODEL_PATH + "liver2.obj",
        MODEL_PATH + "portal2.obj",
        MODEL_PATH + "vein2.obj",
        MODEL_PATH + "tumor.obj",
        MODEL_PATH + "gb.obj",
        MODEL_PATH + "S1.obj",
        MODEL_PATH + "S2.obj",
        MODEL_PATH + "S3.obj",
        MODEL_PATH + "S4.obj",
        MODEL_PATH + "S5.obj",
        MODEL_PATH + "S6.obj",
        MODEL_PATH + "S7.obj",
        MODEL_PATH + "S8.obj"
    };

    std::vector<std::string> ObjTetConnectSoftBody = {
        MODEL_PATH + "soft_liver.obj",
        MODEL_PATH + "soft_portal.obj",
        MODEL_PATH + "soft_vein.obj",
        MODEL_PATH + "soft_tumor.obj",
        MODEL_PATH + "soft_gb.obj",
        MODEL_PATH + "soft_S1.obj",
        MODEL_PATH + "soft_S2.obj",
        MODEL_PATH + "soft_S3.obj",
        MODEL_PATH + "soft_S4.obj",
        MODEL_PATH + "soft_S5.obj",
        MODEL_PATH + "soft_S6.obj",
        MODEL_PATH + "soft_S7.obj",
        MODEL_PATH + "soft_S8.obj"
    };

    // カッターメッシュ
    std::string cutMeshFilePath = MODEL_PATH + "Icosphre4.obj";
    cutterMesh = new mCutMesh(cutMeshFilePath.c_str());

    // メッシュパス
    std::string Liver_lowResPath = MODEL_PATH + "liver_lowRes_mesh.txt";
    std::string Liver_highResPath = MODEL_PATH + "liver_highRes_mesh.txt";
    std::string Portal_lowResPath = MODEL_PATH + "portal_lowRes_mesh.txt";
    std::string Portal_highResPath = MODEL_PATH + "portal_highRes_mesh.txt";
    std::string Vein_lowResPath = MODEL_PATH + "vein_lowRes_mesh.txt";
    std::string Vein_highResPath = MODEL_PATH + "vein_highRes_mesh.txt";
    std::string GB_lowResPath = MODEL_PATH + "GB_lowRes_mesh.txt";
    std::string GB_highResPath = MODEL_PATH + "GB_highRes_mesh.txt";
    std::string Tumor_lowResPath = MODEL_PATH + "tumor_lowRes_mesh.txt";
    std::string Tumor_highResPath = MODEL_PATH + "tumor_highRes_mesh.txt";

    // セグメントパス（546行目付近）
    std::vector<std::string> segmentPaths = {
        MODEL_PATH + "S1.obj",
        MODEL_PATH + "S2.obj",
        MODEL_PATH + "S3.obj",
        MODEL_PATH + "S4.obj",
        MODEL_PATH + "S5.obj",
        MODEL_PATH + "S6.obj",
        MODEL_PATH + "S7.obj",
        MODEL_PATH + "S8.obj"
    };


    // Before
    OrbitCam.setIntrinsics(800.0f, 800.0f, gWindowWidth/2.0f, gWindowHeight/2.0f);
    OrbitCam.printCameraInfo();

    // After（追加の初期化）
    OrbitCam.setWindowSizePointers(&gWindowWidth, &gWindowHeight);
    OrbitCam.setGlobalMatrixPointers(&view, &projection, &model, &objPos);
    OrbitCam.setIntrinsics(800.0f, 800.0f, gWindowWidth/2.0f, gWindowHeight/2.0f);
    OrbitCam.printCameraInfo();



    SimpleOBJ::ProcessingResult result = SimpleOBJ::processMultipleOBJSimple(ObjTarget, ObjTetConnectSoftBody, 3.0);

    if (result.success) {
        // 変換情報を保存（後で逆変換に使用可能）
        std::cout << result.transform.toString() << std::endl;
    }

    float dt = 1.0f / 60.0f;
    glm::vec3 gravity(0.0f, 0.0f, 0.0f);

    for (size_t i = 0; i < cutterMesh->mVertices.size()/3; i++) {
        cutterMesh->mVertices[i * 3] *= 0.4f;
        cutterMesh->mVertices[i * 3 + 1] *= 1.0f;
        cutterMesh->mVertices[i * 3 + 2] *= 1.0f;
    }
    setUp(*cutterMesh);
    cutterStateGPUDuo.initialize(*cutterMesh);

    double lastTime = glfwGetTime();

    // オブジェクトだけ作成、ウィンドウは開かない
    crSlicer = new CRSlicerCrossSec();
    // ★★★ パフォーマンス最適化設定 ★★★
    crSlicer->setMaskResolution(512);    // 解像度を下げる
    crSlicer->setNumSubSlices(10);        // サブスライス数を減らす
    crSlicer->setParallelMode(CRSlicerCrossSec::ParallelMode::PARALLEL_AXES);
    crSlicer->setDebugTiming(true);
    std::cout << "CR Slicer created (press R to open windows)" << std::endl;

    // ★★★ TextRenderer初期化を追加 ★★★
    initTextRenderer();

    {
        CentVoxTetrahedralizerHybrid tetrahedralizer2(
            20,
            ObjTetConnectSoftBody[0],
            Liver_lowResPath,
            CentVoxTetrahedralizerHybrid::DetectionMode::HYBRID,
            1,  // mergeBlockSize
            1   // protectionLayers
            );

        CentVoxTetrahedralizerHybrid::SmoothingSettings smoothSettings2;
        smoothSettings2.enabled = true;
        smoothSettings2.iterations = 0;
        smoothSettings2.smoothFactor = 0.9;
        smoothSettings2.preserveVolume = true;
        smoothSettings2.rescaleToOriginal = true;
        tetrahedralizer2.setSmoothingSettings(smoothSettings2);
        tetrahedralizer2.execute();

        CentVoxTetrahedralizerHybrid tetrahedralizer3(
            40,
            ObjTetConnectSoftBody[0],
            Liver_highResPath,
            CentVoxTetrahedralizerHybrid::DetectionMode::HYBRID,
            1,
            2
            );

        CentVoxTetrahedralizerHybrid::SmoothingSettings smoothSettings3;
        smoothSettings3.enabled = true;
        smoothSettings3.iterations = 3;
        smoothSettings3.smoothFactor = 0.5f;
        smoothSettings3.preserveVolume = true;
        smoothSettings3.rescaleToOriginal = true;
        tetrahedralizer3.setSmoothingSettings(smoothSettings3);
        tetrahedralizer3.execute();

        CentVoxTetrahedralizerHybrid tetrahedralizer4(
            8,
            ObjTetConnectSoftBody[1],
            Portal_lowResPath,
            CentVoxTetrahedralizerHybrid::DetectionMode::HYBRID,
            1,  // mergeBlockSize
            1   // protectionLayers
            );

        CentVoxTetrahedralizerHybrid::SmoothingSettings smoothSettings4;
        smoothSettings4.enabled = true;
        smoothSettings4.iterations = 0;
        smoothSettings4.smoothFactor = 0.9;
        smoothSettings4.preserveVolume = true;
        smoothSettings4.rescaleToOriginal = true;
        tetrahedralizer4.setSmoothingSettings(smoothSettings4);
        tetrahedralizer4.execute();

        CentVoxTetrahedralizerHybrid tetrahedralizer5(
            80,
            ObjTetConnectSoftBody[1],
            Portal_highResPath,
            CentVoxTetrahedralizerHybrid::DetectionMode::HYBRID,
            2,
            3
            );

        CentVoxTetrahedralizerHybrid::SmoothingSettings smoothSettings5;
        smoothSettings5.enabled = true;
        smoothSettings5.iterations = 3;
        smoothSettings5.smoothFactor = 0.5f;
        smoothSettings5.preserveVolume = true;
        smoothSettings5.rescaleToOriginal = true;
        tetrahedralizer5.setSmoothingSettings(smoothSettings5);
        tetrahedralizer5.execute();

        CentVoxTetrahedralizerHybrid tetrahedralizer6(
            8,
            ObjTetConnectSoftBody[2],
            Vein_lowResPath,
            CentVoxTetrahedralizerHybrid::DetectionMode::HYBRID,
            1,  // mergeBlockSize
            1   // protectionLayers
            );

        tetrahedralizer6.setSmoothingSettings(smoothSettings4);
        tetrahedralizer6.execute();

        CentVoxTetrahedralizerHybrid tetrahedralizer7(
            80,
            ObjTetConnectSoftBody[2],
            Vein_highResPath,
            CentVoxTetrahedralizerHybrid::DetectionMode::HYBRID,
            2,
            3
            );

        tetrahedralizer7.setSmoothingSettings(smoothSettings5);
        tetrahedralizer7.execute();


        CentVoxTetrahedralizerHybrid tetrahedralizer8(
            8,
            ObjTetConnectSoftBody[3],
            GB_lowResPath,
            CentVoxTetrahedralizerHybrid::DetectionMode::HYBRID,
            1,  // mergeBlockSize
            1   // protectionLayers
            );

        tetrahedralizer8.setSmoothingSettings(smoothSettings2);
        tetrahedralizer8.execute();

        CentVoxTetrahedralizerHybrid tetrahedralizer9(
            20,
            ObjTetConnectSoftBody[3],
            GB_highResPath,
            CentVoxTetrahedralizerHybrid::DetectionMode::HYBRID,
            2,
            2
            );
        tetrahedralizer9.setSmoothingSettings(smoothSettings3);
        tetrahedralizer9.execute();

        CentVoxTetrahedralizerHybrid tetrahedralizer10(
            8,
            ObjTetConnectSoftBody[4],
            Tumor_lowResPath,
            CentVoxTetrahedralizerHybrid::DetectionMode::HYBRID,
            1,  // mergeBlockSize
            1   // protectionLayers
            );

        tetrahedralizer10.setSmoothingSettings(smoothSettings2);
        tetrahedralizer10.execute();

        CentVoxTetrahedralizerHybrid tetrahedralizer11(
            20,
            ObjTetConnectSoftBody[4],
            Tumor_highResPath,
            CentVoxTetrahedralizerHybrid::DetectionMode::HYBRID,
            2,
            2
            );
        tetrahedralizer11.setSmoothingSettings(smoothSettings3);
        tetrahedralizer11.execute();

    }

    liver = SoftBodyGPUDuo::createFromPaths(Liver_lowResPath, Liver_highResPath, 0.1f, 0.0f, SoftBodyGPUDuo::MeshPreset::LIVER);
    portal = SoftBodyGPUDuo::createFromPaths(Portal_lowResPath, Portal_highResPath, 0.1f, 0.0f, SoftBodyGPUDuo::MeshPreset::VESSEL, liver);
    vein = SoftBodyGPUDuo::createFromPaths(Vein_lowResPath, Vein_highResPath, 0.1f, 0.0f, SoftBodyGPUDuo::MeshPreset::VESSEL, liver);
    gb = SoftBodyGPUDuo::createFromPaths(GB_lowResPath, GB_highResPath, 0.1f, 0.0f, SoftBodyGPUDuo::MeshPreset::VESSEL, liver);
    tumor = SoftBodyGPUDuo::createFromPaths(Tumor_lowResPath, Tumor_highResPath, 0.1f, 0.0f, SoftBodyGPUDuo::MeshPreset::VESSEL, liver);
    glFinish();
    // 親子関係のスキニングを再計算
    if (portal->hasParentSoftBody()) {
        portal->computeSkinningToParent();
        portal->updateFromParent();
    }
    if (vein->hasParentSoftBody()) {
        vein->computeSkinningToParent();
        vein->updateFromParent();
    }
    if (gb->hasParentSoftBody()) {
        gb->computeSkinningToParent();
        gb->updateFromParent();
    }
    if (tumor->hasParentSoftBody()) {
        tumor->computeSkinningToParent();
        tumor->updateFromParent();
    }

    glFinish();  // 再度同期
    std::cout << "=== Parent-child skinning recomputed ===" << std::endl;

    //=========================================================================
    //門脈スケルトンの解析
    //=========================================================================
    std::cout << "\n=== Analyzing Portal Vein Skeleton ===" << std::endl;

    portalSkeleton = VoxelSkeleton::VesselSegmentation::create(
        ObjTetConnectSoftBody[1], segmentPaths, liver, portal);
    // Skeletonバインド後に呼ぶ
    liver->captureOriginalVolumes();
    //=========================================================================

    GrabberGPUDuo grabberBunny;
    gGrabberBunny = &grabberBunny;
    gGrabberBunny->setPhysicsObject(liver);
    gGrabberBunny->setGlobalRefs(&view, &projection, &gWindowWidth, &gWindowHeight);
    glfwSetMouseButtonCallback(gWindow, mouse_button_callback);

    // ★★★ 追加: CPU並列ソルバー初期化 ★★★
    if (useCPUParallel) {
        // liver用（新インターフェース）
        cpuParallelSolver.initialize(liver, liver->lowResEdgeCompliance, liver->lowResVolCompliance);
        cpuParallelSolver.setNumIterations(3, 2);
        cpuParallelSolver.setSolveType(SoftBodyParallelSolver::HYBRID);
        cpuParallelSolver.setJacobiScale(0.25f);
        cpuParallelSolver.setHybridParams(18, 37, 1);
        std::cout << "CPU Parallel Solver initialized for liver" << std::endl;
        std::cout << "  Edge colors: " << cpuParallelSolver.getNumEdgeColors() << std::endl;
        std::cout << "  Tet colors: " << cpuParallelSolver.getNumTetColors() << std::endl;

        // portal用（新インターフェース）
        cpuParallelSolver2.initialize(portal, portal->lowResEdgeCompliance, portal->lowResVolCompliance);
        cpuParallelSolver2.setNumIterations(3, 2);
        cpuParallelSolver2.setSolveType(SoftBodyParallelSolver::HYBRID);
        cpuParallelSolver2.setJacobiScale(0.25f);
        cpuParallelSolver2.setHybridParams(18, 37, 1);  // ← 修正：cpuParallelSolver2
        std::cout << "CPU Parallel Solver initialized for portal" << std::endl;

        // portal用（新インターフェース）
        cpuParallelSolver3.initialize(vein, vein->lowResEdgeCompliance, vein->lowResVolCompliance);
        cpuParallelSolver3.setNumIterations(3, 2);
        cpuParallelSolver3.setSolveType(SoftBodyParallelSolver::HYBRID);
        cpuParallelSolver3.setJacobiScale(0.25f);
        cpuParallelSolver3.setHybridParams(18, 37, 1);  // ← 修正：cpuParallelSolver2
        std::cout << "CPU Parallel Solver initialized for vein" << std::endl;
    }

    // 初期化
    softBodies.push_back(liver);   // 0
    softBodies.push_back(portal);  // 1
    softBodies.push_back(vein);    // 2
    softBodies.push_back(gb);    // 2
    softBodies.push_back(tumor);    // 2
    // 初期化時
    cutSegmentManager.setLiver(liver);
    cutSegmentManager.setPortal(portal);
    cutSegmentManager.setSkeleton(portalSkeleton);

    vein->setXPBDMode(false);
    portal->setXPBDMode(false);
    gb->setXPBDMode(false);
    tumor->setXPBDMode(false);

    int warmupFrames = 5000;


    std::cout << "\n=== Starting Main Loop ===" << std::endl;
    // === メインループ ===
    while (!glfwWindowShouldClose(gWindow)) {
        showFPS(gWindow);
        glfwSwapInterval(1);

        glfwPollEvents();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        double currentTime = glfwGetTime();
        float deltaTime = static_cast<float>(currentTime - lastTime);
        lastTime = currentTime;
        OrbitCam.UpdateCamera(deltaTime);

        if (DuoMeshMode) {
            // ★★★ 最初にGL状態をリセット ★★★
            glDisable(GL_BLEND);

            auto tFrameStart = std::chrono::high_resolution_clock::now();
            gGrabberBunny->update(dt);

            view = glm::lookAt(OrbitCam.cameraPos, OrbitCam.cameraTarget, OrbitCam.cameraUp);
            projection = glm::perspective(glm::radians(OrbitCam.gFOV),
                                          float(gWindowWidth) / float(gWindowHeight), 0.1f, 100.0f);
            model = glm::mat4(1.0f);

            shaderProgramGPUDuo.use();
            shaderProgramGPUDuo.setUniform("model", model);
            shaderProgramGPUDuo.setUniform("view", view);
            shaderProgramGPUDuo.setUniform("projection", projection);
            shaderProgramGPUDuo.setUniform("lightPos", OrbitCam.cameraPos);

            const double MAX_DELTA_TIME = 1.0 / 30.0;
            if (deltaTime > MAX_DELTA_TIME) {
                deltaTime = MAX_DELTA_TIME;
            }

            physicsAccumulator += deltaTime;
            const double MAX_ACCUMULATOR = PHYSICS_TIME_STEP * 2;
            if (physicsAccumulator > MAX_ACCUMULATOR) {
                physicsAccumulator = MAX_ACCUMULATOR;
            }

            int physicsStepCount = 0;

            auto t0 = std::chrono::high_resolution_clock::now();

            // 物理シミュレーション
            //=====================================================================
            int numSubsteps = 7;
            auto invResultPortal = portal->checkLowResInversion();
            auto invResultVein = vein->checkLowResInversion();
            auto invResultGB = gb->checkLowResInversion();

            //if(!cutMode){
            {
                while (physicsAccumulator >= PHYSICS_TIME_STEP) {
                    physicsStepCount++;
                    float stepDt = static_cast<float>(PHYSICS_TIME_STEP) / float(numSubsteps);

                    if (useCPUParallel && cpuParallelSolver.isInitialized()) {
                        // === 親メッシュのXPBD（常に実行）===
                        for (int substep = 0; substep < numSubsteps; substep++) {
                            cpuParallelSolver.solveStep(stepDt, gravity);
                            // ★★★ カット後ダンピング（liver用）★★★
                            if (liverDampingFrame > 0) {
                                liver->applySmoothedDamping();
                                liverDampingFrame--;
                                if (liverDampingFrame == 0) {
                                    std::cout << "[CutDamping] Liver damping completed" << std::endl;
                                }
                            }
                        }

                        // === portal ===
                        if (portal->hasParentSoftBody()) portal->updateFromParent();
                        if (portal->isXPBDModeEnabled()) {
                            for (int substep = 0; substep < numSubsteps * 2; substep++) {
                                cpuParallelSolver2.solveStep(stepDt, gravity);
                            }
                        } else {
                            portal->solveFreeVerticesXPBD();
                        }

                        // === vein ===
                        if (vein->hasParentSoftBody()) vein->updateFromParent();
                        if (vein->isXPBDModeEnabled()) {
                            // パターンA: XPBDで物理シミュレーション
                            for (int substep = 0; substep < numSubsteps * 2; substep++) {
                                cpuParallelSolver3.solveStep(stepDt, gravity);
                            }
                        } else {
                            vein->solveFreeVerticesXPBD();
                        }

                        // === gb ===
                        if (gb->hasParentSoftBody()) gb->updateFromParent();
                        gb->solveFreeVerticesXPBD();

                        // === gb ===
                        if (tumor->hasParentSoftBody()) tumor->updateFromParent();
                        tumor->solveFreeVerticesXPBD();

                    }
                    physicsAccumulator -= PHYSICS_TIME_STEP;
                }
            }
            //}

            // 腫瘍を親に追従させる
            auto t1 = std::chrono::high_resolution_clock::now();
            auto t2 = std::chrono::high_resolution_clock::now();

            //=====================================================================
            // ★★★ 描画前に親追従を確実に実行 ★★★
            //=====================================================================
            if (portal && portal->hasParentSoftBody()) portal->updateFromParent();
            if (vein && vein->hasParentSoftBody()) vein->updateFromParent();
            if (gb && gb->hasParentSoftBody()) gb->updateFromParent();
            if (tumor && tumor->hasParentSoftBody()) tumor->updateFromParent();

            //=====================================================================
            //=====================================================================
            // 描画：肝臓（セグメント色分け）
            //=====================================================================
            if (liver->smoothDisplayMode) {
                liver->updateHighResPositions();
                static int smoothCounter = 0;
                smoothCounter++;
                if (warmupFrames > 0 || smoothCounter % 2 == 0) {
                    liver->updateSmoothMesh();
                }

                // ★★★ CutSegmentModeのオーバーレイを考慮 ★★★
                bool useCutSegmentOverlay = cutSegmentManager.isCutModeActive() && liver->useOBJSegmentColors_;
                //bool useCutSegmentOverlay = (cutSegmentMode != CutSegmentModeType::None) && liver->useOBJSegmentColors_;
                bool useSegmentColoring = (showSegmentColors && liver->isSkeletonBound()) || useCutSegmentOverlay;

                if (useSegmentColoring) {
                    glUniform1i(glGetUniformLocation(shaderProgramGPUDuo.getProgram(), "useVertexColor"), 1);

                    // OBJセグメント色またはCutSegmentOverlayを使用
                    if ((portalSkeleton && portalSkeleton->isUsingOBJSegmentation() && liver->useOBJSegmentColors_) || useCutSegmentOverlay) {
                        liver->drawSmoothMeshWithOBJSegments(shaderProgramGPUDuo);
                    } else {
                        liver->drawSmoothMeshWithSegments(shaderProgramGPUDuo);
                    }
                    glUniform1i(glGetUniformLocation(shaderProgramGPUDuo.getProgram(), "useVertexColor"), 0);
                } else {
                    // 通常色で描画
                    glUniform1i(glGetUniformLocation(shaderProgramGPUDuo.getProgram(), "useVertexColor"), 0);
                    shaderProgramGPUDuo.setUniform("objectColor", glm::vec3(0.8f, 0.2f, 0.2f));
                    liver->drawSmoothMesh(shaderProgramGPUDuo);
                }
            }

            //=====================================================================
            // 描画：門脈（スムースメッシュ）
            //=====================================================================
            if (portal) {
                if (portal->isXPBDModeEnabled()) {
                    portal->updateHighResPositions();           // XPBDモード
                } else {
                    //portal->updateHighResPositionsOnlySkinning();  // スキニングのみ
                    portal->updateHighResPositionsSimpleSkinning();  // スキニングのみ

                    //portal->updateHighResPositions();
                }
                static int smoothCounter2 = 0;
                smoothCounter2++;

                if (warmupFrames > 0 || smoothCounter2 % 2 == 0) {
                    portal->updateSmoothMesh();
                }

                // ★★★ CutSegmentModeのオーバーレイを考慮（肝臓と同じロジック） ★★★
                bool useCutSegmentOverlay = cutSegmentManager.isCutModeActive() && portal->useOBJSegmentColors_;
                bool useSegmentColoring = (showSegmentColors && portal->isSkeletonBound()) || useCutSegmentOverlay;

                if (useSegmentColoring) {
                    glUniform1i(glGetUniformLocation(shaderProgramGPUDuo.getProgram(), "useVertexColor"), 1);

                    // OBJセグメント色またはCutSegmentOverlayを使用
                    if ((portalSkeleton && portalSkeleton->isUsingOBJSegmentation() && portal->useOBJSegmentColors_) || useCutSegmentOverlay) {
                        portal->drawSmoothMeshWithOBJSegments(shaderProgramGPUDuo);
                    } else {
                        portal->drawSmoothMeshWithSegments(shaderProgramGPUDuo);
                    }
                    glUniform1i(glGetUniformLocation(shaderProgramGPUDuo.getProgram(), "useVertexColor"), 0);
                } else {
                    // 通常色で描画
                    glUniform1i(glGetUniformLocation(shaderProgramGPUDuo.getProgram(), "useVertexColor"), 0);
                    shaderProgramGPUDuo.setUniform("objectColor", glm::vec3(0.8f, 0.2f, 0.8f));
                    portal->drawSmoothMesh(shaderProgramGPUDuo);
                }
            }

            if (vein) {
                if (vein->isXPBDModeEnabled()) {
                    vein->updateHighResPositions();           // XPBDモード
                } else {
                    //vein->updateHighResPositionsOnlySkinning();  // スキニングのみ
                    vein->updateHighResPositionsSimpleSkinning();  // スキニングのみ
                }
                static int smoothCounter2 = 0;
                smoothCounter2++;
                if (warmupFrames > 0 || smoothCounter2 % 2 == 0) {
                    vein->updateSmoothMesh();
                }
                glUniform1i(glGetUniformLocation(shaderProgramGPUDuo.getProgram(), "useVertexColor"), 0);
                shaderProgramGPUDuo.setUniform("objectColor", glm::vec3(0.2f, 0.8f, 0.8f));
                vein->drawSmoothMesh(shaderProgramGPUDuo);
            }

            if (gb) {
                gb->updateHighResPositions();
                static int smoothCounter2 = 0;
                smoothCounter2++;
                if (warmupFrames > 0 || smoothCounter2 % 2 == 0) {
                    gb->updateSmoothMesh();
                }
                glUniform1i(glGetUniformLocation(shaderProgramGPUDuo.getProgram(), "useVertexColor"), 0);
                shaderProgramGPUDuo.setUniform("objectColor", glm::vec3(0.2f, 0.8f, 0.2f));
                gb->drawSmoothMesh(shaderProgramGPUDuo);
            }

            if (tumor) {
                tumor->updateHighResPositions();
                static int smoothCounter2 = 0;
                smoothCounter2++;
                if (warmupFrames > 0 || smoothCounter2 % 2 == 0) {
                    tumor->updateSmoothMesh();
                }
                glUniform1i(glGetUniformLocation(shaderProgramGPUDuo.getProgram(), "useVertexColor"), 0);
                shaderProgramGPUDuo.setUniform("objectColor", glm::vec3(0.4f, 0.8f, 0.4f));
                tumor->drawSmoothMesh(shaderProgramGPUDuo);
            }


            // ★★★ カッター描画前のエラーチェック ★★★
            GLenum errBeforeCutter = glGetError();
            if (errBeforeCutter != GL_NO_ERROR) {
                //・std::cout << "ERROR BEFORE CUTTER DRAW: " << errBeforeCutter << std::endl;
            }

            // ★★★ cutterMeshの描画 ★★★
            if (cutMode && cutterMesh && cutterMesh->VAO != 0) {
                shaderProgramGPUDuo.use();
                shaderProgramGPUDuo.setUniform("model", model);
                shaderProgramGPUDuo.setUniform("view", view);
                shaderProgramGPUDuo.setUniform("projection", projection);
                shaderProgramGPUDuo.setUniform("lightPos", OrbitCam.cameraPos);

                glEnable(GL_BLEND);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

                // ★★★ ヒットしたメッシュに応じて色を変更 ★★★
                cutterMesh->mColor = cutterStateGPUDuo.getCutterColor();

                cutterMesh->draw(shaderProgramGPUDuo, 0.5f, 0.5f);

                glDisable(GL_BLEND);
            }

            if (warmupFrames > 0) {
                warmupFrames--;
            }

            // ★カット後の同期（handleGroupが削除された場合、対応するスフィアも削除）
            if (sphereManager.getSphereCount() > 0) {
                sphereManager.syncWithSoftBodyAfterCut(liver);  // ★重要：カット後同期
                sphereManager.drawAll(shaderProgramGPUDuo, view, projection, OrbitCam.cameraPos);
            }

            auto t3 = std::chrono::high_resolution_clock::now();

            profCount++;
            sumPhys += std::chrono::duration<double, std::milli>(t1 - t0).count();
            sumSync += std::chrono::duration<double, std::milli>(t2 - t1).count();
            sumMesh += std::chrono::duration<double, std::milli>(t3 - t2).count();
            sumTotal += std::chrono::duration<double, std::milli>(t3 - tFrameStart).count();
            if (profCount >= 60) {
                if (showTimingInfo) {
                    std::cout << "=== Performance ===" << std::endl;
                    std::cout << "  Physics: " << sumPhys / 60.0 << " ms" << std::endl;
                    std::cout << "  Sync:    " << sumSync / 60.0 << " ms" << std::endl;
                    std::cout << "  Mesh:    " << sumMesh / 60.0 << " ms" << std::endl;
                    std::cout << "  Total:   " << sumTotal / 60.0 << " ms" << std::endl;
                    std::cout << "  FPS:     " << 1000.0 / (sumTotal / 60.0) << std::endl;
                }
                profCount = 0;
                sumPhys = sumSync = sumMesh = sumTotal = 0;
            }


            // メッシュカット実行
            if (performCutOperation) {
                SoftBodyGPUDuo* target = nullptr;
                SoftBodyParallelSolver* cpuSolver = nullptr;
                int* dampingFramePtr = nullptr;
                int isolatedRemovalMode = 0;
                int fragments = 0;
                std::string targetName;
                std::vector<SoftBodyGPUDuo*> childObjects;  // ★追加

                const auto& hit = cutterStateGPUDuo.hitState;

                if (hit.isLiverHit() && liver) {
                    target = liver;
                    cpuSolver = &cpuParallelSolver;
                    dampingFramePtr = &liverDampingFrame;
                    isolatedRemovalMode = 0;
                    fragments = -1;
                    targetName = "liver";
                    childObjects = {portal, vein};  // ★liverの子オブジェクト
                } else if (hit.isPortalHit() && portal) {
                    target = portal;
                    cpuSolver = &cpuParallelSolver2;
                    dampingFramePtr = &portalDampingFrames;
                    isolatedRemovalMode = 0;
                    fragments = -1;
                    targetName = "portal";
                    // portalには子がいないので空
                } else if (hit.isVeinHit() && vein) {
                    target = vein;
                    cpuSolver = &cpuParallelSolver3;
                    dampingFramePtr = &veinDampingFrames;
                    isolatedRemovalMode = 0;
                    fragments = -1;
                    targetName = "vein";
                    // veinには子がいないので空
                }

                if (target) {
                    int dampingOut = 0;
                    bool success = cutManager.performMeshCut(
                        target, cpuSolver,
                        cutterMesh, targetName,
                        isolatedRemovalMode, fragments,
                        dampingOut, POST_CUT_DAMPING_FRAMES,
                        childObjects);  // ★子オブジェクトを渡す

                    target->invalidateLowResTetsWithoutHighRes();
                    target->invalidateHighResNeighborsCache();
                }
                performCutOperation = false;
            }

            // セグメントカット実行
            if (performSegmentCutOperation) {
                std::vector<int> segmentTets = liver->getSelectedTetsForCut(*portalSkeleton);
                std::vector<SoftBodyGPUDuo*> childObjects = {portal, vein};  // ★追加

                bool success = cutManager.performSegmentCut(
                    liver, &cpuParallelSolver,
                    segmentTets, "liver", 1,
                    childObjects);  // ★子オブジェクトを渡す

                performSegmentCutOperation = false;
            }

            // Undo実行
            if (performCutUndo) {
                std::cout << "[DEBUG] performCutUndo triggered, history size: " << cutManager.getHistorySize() << std::endl;
                cutManager.performUndo();
                performCutUndo = false;
            }

            // ★★★ テキストオーバーレイを描画（最後に） ★★★
            renderVolumeOverlay();

            auto tFrameEnd = std::chrono::high_resolution_clock::now();
            double frameTime = std::chrono::duration<double, std::milli>(tFrameEnd - tFrameStart).count();

            const double TARGET_FRAME_TIME = 1000.0 / 25.0;
            if (frameTime < TARGET_FRAME_TIME) {
                double sleepTime = TARGET_FRAME_TIME - frameTime;
                std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(sleepTime)));
            }
        }


        // CR Slicer更新（ウィンドウが開いている場合のみ）
        if (crSlicer && crSlicer->isWindowOpen()) {

            // 1. カメラ連動
            crSlicer->setSliceAxisFromCamera(OrbitCam.cameraPos,
                                             OrbitCam.cameraTarget,
                                             view);

            // 2. 3Dビューに描画
            crSlicer->drawBoundingBox(view, projection);
            crSlicer->drawSlicePlanes(view, projection);

            // 3. 色モードを同期
            if (showSegmentColors && liver->useOBJSegmentColors_) {
                crSlicer->setColorMode(CRSlicerCrossSec::ColorMode::OBJ_SEGMENT);
            } else if (showSegmentColors) {
                crSlicer->setColorMode(CRSlicerCrossSec::ColorMode::SKELETON);
            } else {
                crSlicer->setColorMode(CRSlicerCrossSec::ColorMode::BASE_COLOR);
            }

            // 4. SoftBodyを設定
            std::vector<SoftBodyGPUDuo*> softBodies = {liver, portal, vein, gb, tumor};
            std::vector<glm::vec4> colors = {
                glm::vec4(0.8f, 0.3f, 0.3f, 0.3f),
                glm::vec4(0.8f, 0.2f, 0.6f, 0.8f),
                glm::vec4(0.2f, 0.4f, 0.8f, 0.8f),
                glm::vec4(0.2f, 0.8f, 0.2f, 0.3f),
                glm::vec4(0.9f, 0.8f, 0.9f, 0.3f)
            };
            crSlicer->setSoftBodies(softBodies, colors);

            // 5. カッターを設定 ★★★ updateの前に！★★★
            // 5. カッターを設定
            if (cutterMesh) {
                crSlicer->setCutterMesh(
                    reinterpret_cast<const std::vector<float>*>(&cutterMesh->mVertices),
                    reinterpret_cast<const std::vector<unsigned int>*>(&cutterMesh->mIndices),
                    glm::vec4(0.9f, 0.7f, 0.2f, 1.0f)  // 黄色
                    );
            } else {
                crSlicer->clearCutterMesh();
            }
            // 6. 更新（1回だけ！）
            crSlicer->update();

            // 7. メインコンテキストに戻す
            glfwMakeContextCurrent(gWindow);
        }

        glfwSwapBuffers(gWindow);
    }

    {
        // クリーンアップ
        //delete cutMesh;
        delete liver;
        delete portal;
        delete vein;
        delete gb;
        delete tumor;

        // TextRenderer
        if (textRenderer) {
            delete textRenderer;
            textRenderer = nullptr;
            std::cout << "  TextRenderer deleted" << std::endl;
        }

        // スケルトン
        if (portalSkeleton) {
            delete portalSkeleton;
            portalSkeleton = nullptr;
            std::cout << "  Portal skeleton deleted" << std::endl;
        }

        if (cutterMesh) {
            delete cutterMesh;
            cutterMesh = nullptr;
        }

        // クリーンアップ
        if (crSlicer) {
            delete crSlicer;
            crSlicer = nullptr;
        }

        sphereManager.cleanup();

    }

    glfwTerminate();
    return 0;
}

bool initOpenGL()
{

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_STENCIL_BITS, 8);  // ★★★ ステンシルバッファを追加 ★★★

    gWindow = glfwCreateWindow(gWindowWidth, gWindowHeight, "Window", NULL, NULL);


    glfwMakeContextCurrent(gWindow);
    glewExperimental = GL_TRUE;
    glewInit();

    glfwSetKeyCallback(gWindow, glfw_onKey);
    glfwSetMouseButtonCallback(gWindow, mouse_button_callback);
    glfwSetFramebufferSizeCallback(gWindow, glfw_OnFramebufferSize);
    glfwSetCursorPosCallback(gWindow, glfw_onMouseMoveOrbit);
    glfwSetScrollCallback(gWindow, glfw_onMouseScroll);

    // Hides and grabs cursor, unlimited movement
    //glfwSetInputMode(gWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetInputMode(gWindow, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    glfwSetCursorPos(gWindow, gWindowWidth / 2.0, gWindowHeight / 2.0);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    glViewport(0, 0, gWindowWidth, gWindowHeight);
    glEnable(GL_DEPTH_TEST);

    return true;
}

void glfw_onKey(GLFWwindow* window, int key, int scancode, int action, int mode)
{
    if(DuoMeshMode){

        if (action != GLFW_PRESS) return;

        switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, true);
            break;

        case GLFW_KEY_R:
            if (crSlicer) {
                if (crSlicer->isWindowOpen()) {
                    crSlicer->closeWindows();
                } else {
                    crSlicer->initialize(gWindow, 400);
                }
            }

            // CRSlicerの色モードも同期
            if (crSlicer) {
                if (showSegmentColors && liver->useOBJSegmentColors_) {
                    crSlicer->setColorMode(CRSlicerCrossSec::ColorMode::OBJ_SEGMENT);
                } else if (showSegmentColors) {
                    crSlicer->setColorMode(CRSlicerCrossSec::ColorMode::SKELETON);
                } else {
                    crSlicer->setColorMode(CRSlicerCrossSec::ColorMode::BASE_COLOR);
                }
            }
            break;

        case GLFW_KEY_7:  // または他の空いているキー
        {
            // vein の XPBDモード切り替え
            bool currentMode1 = vein->isXPBDModeEnabled();
            vein->setXPBDMode(!currentMode1);
            // portal の XPBDモード切り替え
            bool currentMode2 = portal->isXPBDModeEnabled();
            portal->setXPBDMode(!currentMode2);
            std::cout << "Portal XPBD mode: " << (portal->isXPBDModeEnabled() ? "ON" : "OFF (Skinning only)") << std::endl;

            std::cout << "Vein XPBD mode: " << (vein->isXPBDModeEnabled() ? "ON" : "OFF (Skinning only)") << std::endl;
        }
        break;

        case GLFW_KEY_Q:
            if (cutMode) {
                cutSegmentManager.cycleCutMode();
                if (!cutSegmentManager.isCutModeActive()) {
                    cutSegmentManager.resetCutOverlay();
                }
                std::cout << "CutSegmentMode: " << cutSegmentManager.getCutModeName() << std::endl;
            } else {
                std::cout << "CutSegmentMode is only available in Cut Mode (press X first)" << std::endl;
            }
            break;

        case GLFW_KEY_X:
            cutMode = !cutMode;
            if (gGrabberBunny) {
                gGrabberBunny->forceEndDrag();
            }
            // cutMode OFF 時に CutSegmentMode もリセット
            if (!cutMode) {
                cutSegmentManager.setCutMode(CutSegmentManager::CutMode::None);
                cutSegmentManager.resetCutOverlay();
            }
            if (cutMode) {
                cutManager.enterCutMode();
            }
            std::cout << "Cut Mode: " << (cutMode ? "ON" : "OFF") << std::endl;
            break;  // ★★★ これが必要！ ★★★


        case GLFW_KEY_U:
            if (action == GLFW_PRESS) {
                if (cutManager.canUndo()) {
                    performCutUndo = true;
                    std::cout << "Undo triggered - History size: " << cutManager.getHistorySize() << std::endl;
                } else {
                    std::cout << "No cut history to undo" << std::endl;
                }
            }
            break;

        case GLFW_KEY_A:
            if (crSlicer && crSlicer->isWindowOpen()) {
                crSlicer->cycleAxisOffset();
            }
            break;

        case GLFW_KEY_E:
            if (crSlicer && crSlicer->isWindowOpen()) {
                crSlicer->togglePreviewLock();
            }
            break;

        case GLFW_KEY_N:
            showSegmentColors = !showSegmentColors;
            std::cout << "Segment colors: " << (showSegmentColors ? "ON" : "OFF") << std::endl;

            if (showSegmentColors && portalSkeleton) {
                // ★ 表示ONの時、内部モードをOBJに設定し、色を更新
                portalSkeleton->setSegmentationMode(VoxelSkeleton::SegmentationMode::OBJ);

                std::cout << "========================================" << std::endl;
                std::cout << " Segmentation Mode: " << portalSkeleton->getCurrentModeName() << std::endl;
                std::cout << "========================================" << std::endl;

                // OBJセグメント色を更新
                if (portal) {
                    portal->updateOBJSegmentColors(*portalSkeleton);
                    portal->useOBJSegmentColors_ = true;
                }
                if (liver) {
                    liver->updateOBJSegmentColors(*portalSkeleton);
                    liver->useOBJSegmentColors_ = true;
                }
            } else if (!showSegmentColors) {
                // ★ 表示OFFの時、色モードを無効化
                if (portal) {
                    portal->useOBJSegmentColors_ = false;
                }
                if (liver) {
                    liver->useOBJSegmentColors_ = false;
                }
            }
            break;

        case GLFW_KEY_D:
            if (!sphereManager.isPlaceMode()) {
                sphereManager.startPlaceMode();
                std::cout << "\n=== Entering PLACEMENT MODE ===" << std::endl;
                std::cout << "Current spheres: " << sphereManager.getSphereCount()
                          << "/" << sphereManager.getMaxSpheres() << std::endl;
                if (!sphereManager.isFull()) {
                    std::cout << "Click mesh to place more spheres." << std::endl;
                } else {
                    std::cout << "Max spheres reached. Press Backspace to remove last sphere." << std::endl;
                }
            }
            else {
                std::cout << "Checking handle groups before entering deform mode..." << std::endl;

                // 空のhandleGroupを削除
                std::vector<int> emptyGroupIndices;
                for (int i = (int)liver->handleGroups.size() - 1; i >= 0; i--) {
                    if (liver->handleGroups[i].vertices.empty()) {
                        emptyGroupIndices.push_back(i);
                        std::cout << "Empty handle group detected at index " << i << std::endl;
                    }
                }

                if (!emptyGroupIndices.empty()) {
                    for (int idx : emptyGroupIndices) {
                        sphereManager.removeSphereAt(idx, liver);
                    }
                    std::cout << emptyGroupIndices.size() << " empty sphere(s) removed." << std::endl;
                }

                if (sphereManager.getSphereCount() > 0) {
                    sphereManager.endPlaceMode();
                    std::cout << "\n=== Entering DEFORM MODE ===" << std::endl;
                    std::cout << "Active spheres: " << sphereManager.getSphereCount()
                              << "/" << sphereManager.getMaxSpheres() << std::endl;
                    std::cout << "Drag spheres to deform! Press D to add more spheres." << std::endl;
                }
                else {
                    std::cout << "\n=== No spheres placed ===" << std::endl;
                    std::cout << "Please place at least 1 sphere on the mesh first." << std::endl;
                }
            }
            break;

        case GLFW_KEY_BACKSPACE:
            if (sphereManager.getSphereCount() > 0) {
                sphereManager.removeLastSphere(liver);
                std::cout << "Last handle group removed. Remaining: "
                          << sphereManager.getSphereCount() << std::endl;
            }
            break;


        case GLFW_KEY_DELETE:
            sphereManager.clearAll(liver);
            std::cout << "All handle groups cleared" << std::endl;
            break;

        case GLFW_KEY_P:
            if (!sphereManager.isPlaceMode() && liver) {
                // デフォームモード中: アクティブなハンドルグループを削除
                int activeGroup = liver->activeHandleGroup;

                if (activeGroup >= 0) {
                    std::cout << "\n=== Removing active handle group " << activeGroup << " ===" << std::endl;

                    // グラバーの状態をリセット
                    gGrabberBunny->endSmartGrab();

                    // スフィアとハンドルグループを削除
                    sphereManager.removeSphereAt(activeGroup, liver);

                    std::cout << "Handle group removed - mesh will spring back!" << std::endl;
                    std::cout << "Remaining groups: " << sphereManager.getSphereCount() << std::endl;

                } else {
                    std::cout << "No active handle group to remove (drag a sphere first)" << std::endl;
                }
            } else {
                // 配置モード中またはliverがない場合: Timing Info切り替え
                showTimingInfo = !showTimingInfo;
                std::cout << "Timing Info: " << (showTimingInfo ? "ON" : "OFF") << std::endl;
            }
            break;


        case GLFW_KEY_B:
            // ★★★ オーバーレイ表示をトグル ★★★
            showVolumeOverlay = !showVolumeOverlay;
            std::cout << "Volume overlay: " << (showVolumeOverlay ? "ON" : "OFF") << std::endl;

            // コンソールにも出力（オプション）
            if (liver && showVolumeOverlay) {
                bool useOBJ = portalSkeleton && portalSkeleton->isUsingOBJSegmentation();
                int selectedOBJ = useOBJ ? portalSkeleton->getSelectedOBJSegment() : -1;
                liver->printVolumeInfo(useOBJ, selectedOBJ);
            }
            break;


        case GLFW_KEY_K:
            if (portal) {
                portal->clearSegmentSelection();
            }
            if (liver) {
                liver->clearSegmentSelection();
            }
            if (portalSkeleton) {
                portalSkeleton->clearOBJSegmentSelection();
                portalSkeleton->clearBranchSelection();  // ★ Voronoi3D用も追加

                // ★ 現在のモードに応じて色を再更新
                VoxelSkeleton::SegmentationMode mode = portalSkeleton->getSegmentationMode();

                switch (mode) {
                case VoxelSkeleton::SegmentationMode::OBJ:
                    if (portal) portal->updateOBJSegmentColors(*portalSkeleton);
                    if (liver) liver->updateOBJSegmentColors(*portalSkeleton);
                    break;

                case VoxelSkeleton::SegmentationMode::SkeletonDistance:
                    if (portal) portal->forceUpdateSegmentColors();
                    if (liver) liver->forceUpdateSegmentColors();
                    break;

                case VoxelSkeleton::SegmentationMode::Voronoi3D:
                    if (portal) portal->updateVoronoi3DColors(*portalSkeleton);
                    if (liver) liver->updateVoronoi3DColors(*portalSkeleton);
                    break;
                }
            }
            std::cout << "Selection cleared and colors restored" << std::endl;
            break;

        // 修正後
        case GLFW_KEY_F:
            cutSegmentManager.toggleLiverSelectMode();
            break;

        case GLFW_KEY_G:
            cutSegmentManager.togglePortalSelectMode();
            break;

        // 修正後
        case GLFW_KEY_Z:
            if (cutSegmentManager.isPortalSelectMode() || cutSegmentManager.isLiverSelectMode())
                performSegmentCutOperation = true;
            std::cout << "Segment cut operation scheduled" << std::endl;
            break;


        case GLFW_KEY_H:
            liver->lowRes_positions = liver->getLowResMeshData().verts;
            liver->lowRes_prevPositions = liver->getLowResMeshData().verts;
            std::fill(liver->lowRes_velocities.begin(), liver->lowRes_velocities.end(), 0.0f);
            liver->updateLowResMesh();
            liver->updateHighResMesh();

            std::cout << "Physics reset" << std::endl;
            break;

        case GLFW_KEY_C:
            performCutOperation = true;
            break;

        case GLFW_KEY_O:
            if (portalSkeleton) {
                // モードを切り替え（OBJ → SkeletonDistance → Voronoi3D → OBJ ...）
                portalSkeleton->cycleSegmentationMode();

                std::cout << "========================================" << std::endl;
                std::cout << " Segmentation Mode: " << portalSkeleton->getCurrentModeName() << std::endl;
                std::cout << "========================================" << std::endl;

                // 現在のモードに応じて色を更新
                VoxelSkeleton::SegmentationMode mode = portalSkeleton->getSegmentationMode();

                switch (mode) {
                case VoxelSkeleton::SegmentationMode::OBJ:
                    // OBJモード
                    std::cout << "Updating OBJ segment colors..." << std::endl;
                    if (portal) {
                        portal->updateOBJSegmentColors(*portalSkeleton);
                        portal->useOBJSegmentColors_ = true;
                    }
                    if (liver) {
                        liver->updateOBJSegmentColors(*portalSkeleton);
                        liver->useOBJSegmentColors_ = true;
                    }
                    break;

                case VoxelSkeleton::SegmentationMode::SkeletonDistance:
                    // スケルトン距離モード
                    std::cout << "Updating Skeleton Distance colors..." << std::endl;
                    if (portal) {
                        portal->useOBJSegmentColors_ = false;
                        portal->forceUpdateSegmentColors();
                    }
                    if (liver) {
                        liver->useOBJSegmentColors_ = false;
                        liver->forceUpdateSegmentColors();
                    }
                    break;

                case VoxelSkeleton::SegmentationMode::Voronoi3D:
                    // Voronoi 3Dモード
                    std::cout << "Updating Voronoi 3D colors..." << std::endl;
                    if (portal) {
                        portal->updateVoronoi3DColors(*portalSkeleton);
                        portal->useOBJSegmentColors_ = true;  // 同じバッファを使用
                    }
                    if (liver) {
                        liver->updateVoronoi3DColors(*portalSkeleton);
                        liver->useOBJSegmentColors_ = true;
                    }
                    break;
                }
            }
            break;
        }
    }
}

void glfw_OnFramebufferSize(GLFWwindow* window, int width, int height)
{
    gWindowWidth = width;
    gWindowHeight = height;
    glViewport(0, 0, gWindowWidth, gWindowHeight);
}

void showFPS(GLFWwindow* window) {
    static double previousSeconds = 0.0;
    static int frameCount = 0;
    double currentSeconds = glfwGetTime();
    double elapsedSeconds = currentSeconds - previousSeconds;
    if (elapsedSeconds > 0.25) {
        previousSeconds = currentSeconds;
        double fps = (double)frameCount / elapsedSeconds;
        double msPerFrame = 1000.0 / fps;

        // ソルバーモード文字列
        const char* solverMode;
        if (useCPUParallel) {
            solverMode = "CPU Parallel";
        } else {
            solverMode = "CPU Serial";
        }

        std::ostringstream outs;
        outs.precision(3);
        outs << std::fixed
             << "SoftBody Simulation" << "    "
             << "FPS: " << fps << "    "
             << "Frame Time: " << msPerFrame << " (ms)" << "    "
             << "[" << solverMode << "]";

        // ★★★ モード表示 ★★★
        if (cutMode) {
            outs << "  [CUT";
            if (cutSegmentManager.isCutModeActive()) {
                outs << "+" << cutSegmentManager.getCutModeName();
            }
            outs << "]";
        } else {
            // Deform モードの場合
            outs << "  [Deform:";
            if (sphereManager.isPlaceMode()) {
                outs << "PLACE";
            } else if (sphereManager.getSphereCount() > 0) {
                outs << "ACTIVE";
            } else {
                outs << "IDLE";
            }
            outs << " (" << sphereManager.getSphereCount() << "/" << sphereManager.getMaxSpheres() << ")]";
        }

        glfwSetWindowTitle(window, outs.str().c_str());
        frameCount = 0;
    }
    frameCount++;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if(DuoMeshMode){
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        //=========================================================================
        // 左クリック処理
        //=========================================================================
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
            bool shiftPressed = (mods & GLFW_MOD_SHIFT) != 0;
            // ★★★ デバッグ出力 ★★★
            if (action == GLFW_PRESS) {
                // std::cout << "[DEBUG] mouse_button_callback:" << std::endl;
                // std::cout << "  cutMode=" << cutMode << std::endl;
                // std::cout << "  isDragging=" << isDragging << std::endl;
                // std::cout << "  handlePlaceMode=" << sphereManager.isPlaceMode() << std::endl;
                // std::cout << "  button=" << button << " (LEFT=0, RIGHT=1)" << std::endl;
            }

            // 修正後
            if (cutSegmentManager.isLiverSelectMode() ||
                cutSegmentManager.isPortalSelectMode() ||
                cutSegmentManager.isManualExtendMode() ||
                shiftPressed) {
                cutSegmentManager.handleClick(
                    static_cast<float>(xpos), static_cast<float>(ypos), shiftPressed,
                    view, projection, OrbitCam.cameraPos, gWindowWidth, gWindowHeight);
                return;
            }
        }
        //=========================================================================
        // 通常モード（グラブ操作）
        //=========================================================================
        if (!cutMode) {
            if (button == GLFW_MOUSE_BUTTON_LEFT) {
                if (action == GLFW_PRESS) {
                    if (sphereManager.isPlaceMode()) {
                        if (sphereManager.isFull()) {
                            return;
                        }

                        gGrabberBunny->startGrab(xpos, ypos);

                        if (gGrabberBunny->isDragging()) {
                            glm::vec3 rayHitPos = gGrabberBunny->hit_position;
                            int nearestVertexIdx = liver->findClosestSurfaceVertex(rayHitPos);
                            glm::vec3 actualVertexPos = liver->getVertexPosition(nearestVertexIdx);

                            gGrabberBunny->endGrab();

                            // ★スフィア配置（距離チェック・handleGroup作成・頂点数チェックも内部で処理）
                            // ★handleGroupが作成できない/空の場合は自動でfalseを返す
                            if (!sphereManager.placeSphere(actualVertexPos, liver)) {
                                std::cout << "Sphere placement failed at vertex " << nearestVertexIdx << std::endl;
                                return;
                            }

                            std::cout << "Sphere placed at vertex " << nearestVertexIdx
                                      << " (" << sphereManager.getSphereCount() << "/"
                                      << sphereManager.getMaxSpheres() << ")" << std::endl;
                        }
                    }
                    else {
                        gGrabberBunny->startSmartGrab(xpos, ypos, sphereManager.getHandleRadius());
                    }
                }
                else if (action == GLFW_RELEASE) {
                    if (!sphereManager.isPlaceMode()) {
                        gGrabberBunny->endSmartGrab();
                        sphereManager.syncPositionsFromSoftBody(liver);
                    }
                }
            }
        }
        //=========================================================================
        // カットモード（DuoMeshMode）
        if (cutMode) {
            if (action == GLFW_PRESS && (button == GLFW_MOUSE_BUTTON_LEFT || button == GLFW_MOUSE_BUTTON_RIGHT)) {

                // ★★★ まず SoftBodyGPUDuo 配列に対するヒット判定 ★★★
                std::vector<SoftBodyGPUDuo*> softBodies;
                if (liver) softBodies.push_back(liver);
                if (portal) softBodies.push_back(portal);
                if (vein) softBodies.push_back(vein);

                RayCastGPUDuo::SoftBodyHitResult hitResult = RayCastGPUDuo::FindHitInSoftBodies(
                    xpos, ypos,
                    softBodies,
                    view, projection,
                    OrbitCam.cameraPos,
                    gWindowWidth, gWindowHeight,
                    true
                    );

                if (hitResult.hit) {
                    // ★★★ 変更: isDragging → cutterStateGPUDuo.hitState ★★★
                    cutterStateGPUDuo.resetHit();

                    float targetScale = 1.0f;

                    // ★★★ ヒットしたメッシュに応じてフラグとサイズを設定 ★★★
                    if (hitResult.meshIndex == 0) {
                        cutterStateGPUDuo.setBodyHit(liver, CutterMeshState::LIVER, hitResult.hitPosition);
                        targetScale = cutterStateGPUDuo.targetScales.liver;
                        std::cout << "CutMode: Hit on Liver" << std::endl;
                    } else if (hitResult.meshIndex == 1) {
                        cutterStateGPUDuo.setBodyHit(portal, CutterMeshState::PORTAL, hitResult.hitPosition);
                        targetScale = cutterStateGPUDuo.targetScales.vessel;
                        std::cout << "CutMode: Hit on Portal" << std::endl;
                    } else if (hitResult.meshIndex == 2) {
                        cutterStateGPUDuo.setBodyHit(vein, CutterMeshState::VEIN, hitResult.hitPosition);
                        targetScale = cutterStateGPUDuo.targetScales.vessel;
                        std::cout << "CutMode: Hit on Vein" << std::endl;
                    }

                    std::cout << "  hit_position: (" << hitResult.hitPosition.x << ", "
                              << hitResult.hitPosition.y << ", " << hitResult.hitPosition.z << ")" << std::endl;

                    // ★★★ カッターをヒット位置に移動してスケール適用 ★★★
                    if (cutterMesh) {
                        glm::vec3 currentCenter(0.0f);
                        size_t vertexCount = cutterMesh->mVertices.size() / 3;
                        for (size_t i = 0; i < cutterMesh->mVertices.size(); i += 3) {
                            currentCenter.x += cutterMesh->mVertices[i];
                            currentCenter.y += cutterMesh->mVertices[i + 1];
                            currentCenter.z += cutterMesh->mVertices[i + 2];
                        }
                        currentCenter /= static_cast<float>(vertexCount);

                        glm::vec3 moveVector = hitResult.hitPosition - currentCenter;
                        cutterStateGPUDuo.updateTranslation(moveVector);
                        cutterStateGPUDuo.applyScale(*cutterMesh, targetScale);
                        setUp(*cutterMesh);

                        std::cout << "  Cutter scale: " << targetScale << std::endl;
                    }

                    // ★修正：変換後のcutterMesh->mVerticesから再計算
                    // ★修正：変換後のcutterMesh->mVerticesから再計算
                    // ★★★ カッターをヒット位置に移動してスケール適用 ★★★
                    if (cutterMesh) {
                        glm::vec3 currentCenter(0.0f);
                        size_t vertexCount = cutterMesh->mVertices.size() / 3;
                        for (size_t i = 0; i < cutterMesh->mVertices.size(); i += 3) {
                            currentCenter.x += cutterMesh->mVertices[i];
                            currentCenter.y += cutterMesh->mVertices[i + 1];
                            currentCenter.z += cutterMesh->mVertices[i + 2];
                        }
                        currentCenter /= static_cast<float>(vertexCount);

                        std::cout << "=== DEBUG: Before move ===" << std::endl;
                        std::cout << "  currentCenter: (" << currentCenter.x << ", " << currentCenter.y << ", " << currentCenter.z << ")" << std::endl;
                        std::cout << "  hitResult.hitPosition: (" << hitResult.hitPosition.x << ", " << hitResult.hitPosition.y << ", " << hitResult.hitPosition.z << ")" << std::endl;

                        glm::vec3 moveVector = hitResult.hitPosition - currentCenter;
                        std::cout << "  moveVector: (" << moveVector.x << ", " << moveVector.y << ", " << moveVector.z << ")" << std::endl;

                        cutterStateGPUDuo.updateTranslation(moveVector);
                        cutterStateGPUDuo.applyScale(*cutterMesh, targetScale);
                        setUp(*cutterMesh);

                        std::cout << "  Cutter scale: " << targetScale << std::endl;
                    }


                    if (crSlicer && crSlicer->isWindowOpen() && cutterMesh) {
                        crSlicer->alignSlicesToCutterCenter(&cutterMesh->mVertices);
                    }

                    // 修正後
                    if (cutSegmentManager.isCutModeActive()) {
                        cutSegmentManager.updateFromCutterHit(
                            hitResult.hitPosition,
                            cutterStateGPUDuo.hitState.isLiverHit(),
                            cutterStateGPUDuo.hitState.isPortalHit());
                        cutSegmentManager.applyCutOverlay();
                    }
                }
                else {
                    // ★★★ SoftBodyにヒットしなかった場合 ★★★
                    if (cutterMesh) {
                        // ★★★ 変更: FindHit() → cutterStateGPUDuo.findCutterHit() ★★★
                        cutterStateGPUDuo.findCutterHit(
                            xpos, ypos,
                            view, projection,
                            gWindowWidth, gWindowHeight,
                            *cutterMesh);

                        if (cutterStateGPUDuo.hitState.isDragging) {
                            std::cout << "CutMode: Drag started on cutterMesh directly" << std::endl;
                        }
                    }
                }
            }
            else if (action == GLFW_RELEASE) {
                bool leftStillPressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
                bool rightStillPressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

                if (!leftStillPressed && !rightStillPressed) {
                    cutterStateGPUDuo.hitState.endDrag();  // ★ ドラッグ終了だけ、色は維持
                }
            }
        }
    }
}

void glfw_onMouseMoveOrbit(GLFWwindow* window, double posX, double posY) {
    if(DuoMeshMode){
        static glm::vec2 lastMousePos = glm::vec2(0, 0);
        static bool firstMouse = true;
        if (firstMouse) {
            lastMousePos = glm::vec2(posX, posY);
            firstMouse = false;
            return;
        }
        float deltaX = posX - lastMousePos.x;
        float deltaY = posY - lastMousePos.y;

        // ★★★ ボタン状態を先に取得 ★★★
        bool leftBtn = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        bool rightBtn = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

        // ★★★ 変形モード（!cutMode）★★★
        if (!cutMode) {
            bool grabbing = gGrabberBunny && gGrabberBunny->isDragging();

            if (grabbing && leftBtn) {
                if (!sphereManager.isPlaceMode()) {
                    gGrabberBunny->moveSmartGrab(posX, posY, 1.0f / 60.0f);
                    sphereManager.syncPositionsFromSoftBody(liver);
                }
            }
            else if (!grabbing && leftBtn && !rightBtn) {
                OrbitCam.Rotate(deltaX, deltaY);
            }
            else if (!grabbing && rightBtn && !leftBtn) {
                OrbitCam.Pan(deltaX, deltaY);  // ★追加
            }
        }
        // ★★★ カットモード ★★★
        else {
            // カッターをドラッグ中の場合のみカッター操作
            if (cutterStateGPUDuo.hitState.isDragging) {
                // 左ドラッグ: カッター回転
                if (leftBtn && !rightBtn) {
                    float rotX = deltaY * 0.01f;
                    float rotY = deltaX * 0.01f;
                    glm::mat4 deltaRotation = glm::mat4(1.0f);
                    deltaRotation = glm::rotate(deltaRotation, rotX, OrbitCam.cameraRight);
                    deltaRotation = glm::rotate(deltaRotation, rotY, OrbitCam.cameraUp);
                    cutterStateGPUDuo.updateRotation(deltaRotation);
                    cutterStateGPUDuo.applyScale(*cutterMesh, cutterStateGPUDuo.getCurrentScale());
                    setUp(*cutterMesh);
                }
                // 右ドラッグ: カッター平行移動
                else if (rightBtn && !leftBtn) {
                    float dx = deltaX * OrbitCam.MOUSE_SENSITIVITY;
                    float dy = -deltaY * OrbitCam.MOUSE_SENSITIVITY;
                    glm::vec3 moveDirection = OrbitCam.cameraRight * dx + OrbitCam.cameraUp * dy;
                    cutterStateGPUDuo.updateTranslation(moveDirection);
                    cutterStateGPUDuo.applyScale(*cutterMesh, cutterStateGPUDuo.getCurrentScale());
                    setUp(*cutterMesh);
                }
                // 両ボタン: カッター前後移動
                else if (leftBtn && rightBtn) {
                    glm::vec3 moveForward = OrbitCam.cameraDirection *
                                            ((float)posY - lastMousePos.y) *
                                            OrbitCam.LIGHT_MOUSE_SENSITIVITY;
                    cutterStateGPUDuo.updateTranslation(moveForward);
                    cutterStateGPUDuo.applyScale(*cutterMesh, cutterStateGPUDuo.getCurrentScale());
                    setUp(*cutterMesh);
                }
            }
            // カッターをドラッグしていない → カメラ操作
            else {
                if (leftBtn && !rightBtn) {
                    OrbitCam.Rotate(deltaX, deltaY);
                }
                else if (rightBtn && !leftBtn) {
                    OrbitCam.Pan(deltaX, deltaY);  // ★追加
                }
            }
        }
        lastMousePos.x = (float)posX;
        lastMousePos.y = (float)posY;
    }
}

void glfw_onMouseScroll(GLFWwindow* window, double deltaX, double deltaY) {
    if (ignoreNextScroll) {
        ignoreNextScroll = false;
        return;
    }

    // 1. カメラモード → カメラズーム（最優先）
    if (cameraMode) {
        OrbitCam.Zoom(deltaY);
        return;
    }

    // === 以下ノンカメラモード ===

    bool rightClick = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
    bool leftClick = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);

    // スクロール処理（現在軸のみ移動）
    // === CRSlicer スクロール処理 ===
    if (crSlicer && crSlicer->isWindowOpen()) {
        // どのCRSlicerウィンドウにマウスがあるか確認
        int crWindowAxis = crSlicer->getWindowUnderMouse();

        if (crWindowAxis >= 0) {
            // CRSlicerウィンドウ上 → そのウィンドウの軸を移動
            crSlicer->moveSlice(crWindowAxis, deltaY * 0.02f);
            return;
        }

        // メインウィンドウでの操作
        bool hasPreviewCR = crSlicer->isWindowOpen();

        if (DuoMeshMode && cutMode) {
            if (leftClick && hasPreviewCR) {
                // カットモード + 右クリック + CRプレビュー → 現在軸を移動
                crSlicer->moveSlicePosition(deltaY * 0.02f);
                return;
            }
        }
    }

    // 2. CUTモード
    if (DuoMeshMode && cutMode) {
        bool rightClick = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
        bool leftClick = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);

        if (rightClick && !leftClick) {
            // 右クリック → カッターサイズ変更
            float newScale = cutterStateGPUDuo.getCurrentTargetScale();

            if (deltaY > 0) newScale *= OrbitCam.SCALE_SPEED;
            else if (deltaY < 0) newScale /= OrbitCam.SCALE_SPEED;
            newScale = glm::clamp(newScale, 0.1f, 5.0f);

            cutterStateGPUDuo.setCurrentTargetScale(newScale);
            cutterStateGPUDuo.applyScale(*cutterMesh, newScale);
            setUp(*cutterMesh);
            std::cout << "Cutter scale: " << newScale << std::endl;

        } else if (leftClick) {
            // 左クリック + プレビューなし → カッターサイズ変更
            float newScale = cutterStateGPUDuo.getCurrentTargetScale();

            if (deltaY > 0) newScale *= OrbitCam.SCALE_SPEED;
            else if (deltaY < 0) newScale /= OrbitCam.SCALE_SPEED;
            newScale = glm::clamp(newScale, 0.1f, 5.0f);

            cutterStateGPUDuo.setCurrentTargetScale(newScale);
            cutterStateGPUDuo.applyScale(*cutterMesh, newScale);
            setUp(*cutterMesh);
            std::cout << "Cutter scale: " << newScale << std::endl;

        } else {
            // 何もクリックしていない → カメラズーム
            OrbitCam.Zoom(deltaY);
        }
        return;
    }



    // 3. DEFORMモード
    if (rightClick) {
        // DEFORM + プレビュー + 右クリック → スライス移動
        //ctSlicer->moveSlicePosition(static_cast<float>(deltaY) * 0.02f);
    } else {
        // DEFORM + それ以外 → カメラズーム
        OrbitCam.Zoom(deltaY);
    }
}
