#ifndef SHADER_PROGRAM_H
#define SHADER_PROGRAM_H

#include <string>
#include <map>
#define GLEW_STATIC
#include "GL/glew.h"
#include "glm/glm.hpp"

using std::string;

class ShaderProgram {
public:
    ShaderProgram();
    ~ShaderProgram();

    // シェーダーの種類を表す列挙型
    enum ShaderType {
        VERTEX,
        FRAGMENT,
        PROGRAM
    };

    // シェーダーの読み込みとコンパイル
    bool loadShaders(const char* vsFilename, const char* fsFilename);

    // シェーダープログラムを有効化
    void use();

    // プログラムIDの取得
    GLuint getProgram() const;

    // ユニフォーム変数の設定
    void setUniform(const GLchar* name, const glm::vec2& v);
    void setUniform(const GLchar* name, const glm::vec3& v);
    void setUniform(const GLchar* name, const glm::vec4& v);
    void setUniform(const GLchar* name, const glm::mat4& m);

    // ユニフォーム変数の位置を取得
    GLint getUniformLocation(const GLchar* name);

    // ユニフォーム変数の設定（既存のメソッドの下に追加）
    void setUniform(const GLchar* name, int value);
    void setUniform(const GLchar* name, float value);
    // シェーダープログラムのハンドル
    GLuint mHandle;
private:


    // ユニフォーム位置のキャッシュ
    std::map<string, GLint> mUniformLocations;

    // ヘルパー関数
    string fileToString(const string& filename);
    void checkCompileErrors(GLuint shader, ShaderType type);
};

#endif // SHADER_PROGRAM_H
