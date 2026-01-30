#ifndef MESH_DATA_TYPES_H
#define MESH_DATA_TYPES_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <glm/glm.hpp>

namespace MeshDataTypes {
    // 純粋なメッシュデータ（描画から独立）
    struct SimpleMeshData {
        std::vector<float> vertices;
        std::vector<unsigned int> indices;
    };

    // OBJファイル読み込み用の独立した関数
    SimpleMeshData loadOBJFile(const char* filePath);
}

#endif // MESH_DATA_TYPES_H
