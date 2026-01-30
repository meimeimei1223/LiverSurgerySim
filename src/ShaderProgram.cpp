#include "ShaderProgram.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <glm/gtc/type_ptr.hpp>

//-----------------------------------------------------------------------------
// コンストラクタ
//-----------------------------------------------------------------------------
ShaderProgram::ShaderProgram() : mHandle(0) {
}

//-----------------------------------------------------------------------------
// デストラクタ
//-----------------------------------------------------------------------------
ShaderProgram::~ShaderProgram() {
    // プログラムの削除
    glDeleteProgram(mHandle);
}

//-----------------------------------------------------------------------------
// 頂点シェーダーとフラグメントシェーダーの読み込み
//-----------------------------------------------------------------------------
bool ShaderProgram::loadShaders(const char* vsFilename, const char* fsFilename) {
    // ファイルから文字列を読み込む
    string vsString = fileToString(vsFilename);
    string fsString = fileToString(fsFilename);
    const GLchar* vsSourcePtr = vsString.c_str();
    const GLchar* fsSourcePtr = fsString.c_str();

    // シェーダーの作成
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);

    // シェーダーソースの設定
    glShaderSource(vs, 1, &vsSourcePtr, NULL);
    glShaderSource(fs, 1, &fsSourcePtr, NULL);

    // 頂点シェーダーのコンパイル
    glCompileShader(vs);
    checkCompileErrors(vs, VERTEX);

    // フラグメントシェーダーのコンパイル
    glCompileShader(fs);
    checkCompileErrors(fs, FRAGMENT);

    // プログラムの作成
    mHandle = glCreateProgram();
    if (mHandle == 0) {
        std::cerr << "Unable to create shader program!" << std::endl;
        return false;
    }

    // シェーダーをプログラムにアタッチ
    glAttachShader(mHandle, vs);
    glAttachShader(mHandle, fs);

    // プログラムのリンク
    glLinkProgram(mHandle);
    checkCompileErrors(mHandle, PROGRAM);

    // シェーダーは不要になったので削除
    glDeleteShader(vs);
    glDeleteShader(fs);

    // ユニフォーム位置のキャッシュをクリア
    mUniformLocations.clear();

    return true;
}

//-----------------------------------------------------------------------------
// ファイルを文字列に変換
//-----------------------------------------------------------------------------
string ShaderProgram::fileToString(const string& filename) {
    std::stringstream ss;
    std::ifstream file;

    try {
        file.open(filename, std::ios::in);

        if (!file.fail()) {
            // stringstream を使用してファイル内容を読み込む
            ss << file.rdbuf();
        }

        file.close();
    } catch (std::exception ex) {
        std::cerr << "Error reading shader filename!" << std::endl;
    }

    return ss.str();
}

//-----------------------------------------------------------------------------
// シェーダープログラムを使用する
//-----------------------------------------------------------------------------
void ShaderProgram::use() {
    if (mHandle > 0)
        glUseProgram(mHandle);
}

//-----------------------------------------------------------------------------
// コンパイルエラーのチェック
//-----------------------------------------------------------------------------
void ShaderProgram::checkCompileErrors(GLuint shader, ShaderType type) {
    int status = 0;

    if (type == PROGRAM) {
        // プログラムのリンクステータスを確認
        glGetProgramiv(shader, GL_LINK_STATUS, &status);
        if (status == GL_FALSE) {
            GLint length = 0;
            glGetProgramiv(shader, GL_INFO_LOG_LENGTH, &length);

            // NULL文字を含む長さ
            string errorLog(length, ' ');
            glGetProgramInfoLog(shader, length, &length, &errorLog[0]);
            std::cerr << "Error! Shader program failed to link. " << errorLog << std::endl;
        }
    } else {
        // シェーダーのコンパイルステータスを確認
        glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
        if (status == GL_FALSE) {
            GLint length = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);

            // NULL文字を含む長さ
            string errorLog(length, ' ');
            glGetShaderInfoLog(shader, length, &length, &errorLog[0]);
            std::cerr << "Error! Shader failed to compile. " << errorLog << std::endl;
        }
    }
}

//-----------------------------------------------------------------------------
// プログラムIDの取得
//-----------------------------------------------------------------------------
GLuint ShaderProgram::getProgram() const {
    return mHandle;
}

//-----------------------------------------------------------------------------
// ユニフォーム変数の設定（vec2）
//-----------------------------------------------------------------------------
void ShaderProgram::setUniform(const GLchar* name, const glm::vec2& v) {
    GLint loc = getUniformLocation(name);
    glUniform2f(loc, v.x, v.y);
}

//-----------------------------------------------------------------------------
// ユニフォーム変数の設定（vec3）
//-----------------------------------------------------------------------------
void ShaderProgram::setUniform(const GLchar* name, const glm::vec3& v) {
    GLint loc = getUniformLocation(name);
    glUniform3f(loc, v.x, v.y, v.z);
}

//-----------------------------------------------------------------------------
// ユニフォーム変数の設定（vec4）
//-----------------------------------------------------------------------------
void ShaderProgram::setUniform(const GLchar* name, const glm::vec4& v) {
    GLint loc = getUniformLocation(name);
    glUniform4f(loc, v.x, v.y, v.z, v.w);
}

//-----------------------------------------------------------------------------
// ユニフォーム変数の設定（mat4）
//-----------------------------------------------------------------------------
void ShaderProgram::setUniform(const GLchar* name, const glm::mat4& m) {
    GLint loc = getUniformLocation(name);

    // loc = ユニフォームの位置
    // count = 行列の数（配列でない場合は1）
    // transpose = 列優先なのでfalse
    // value = 設定する行列の値
    glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(m));
}

//-----------------------------------------------------------------------------
// ユニフォーム変数の位置を文字列名から取得
// 注意: シェーダーが現在アクティブである必要があります
//-----------------------------------------------------------------------------
GLint ShaderProgram::getUniformLocation(const GLchar* name) {
    std::map<string, GLint>::iterator it = mUniformLocations.find(name);

    // まだマップにない場合だけシェーダープログラムに問い合わせる
    if (it == mUniformLocations.end()) {
        // 検索して位置をマップに追加
        mUniformLocations[name] = glGetUniformLocation(mHandle, name);
    }

    // 位置を返す
    return mUniformLocations[name];
}

//-----------------------------------------------------------------------------
// ユニフォーム変数の設定（int）
//-----------------------------------------------------------------------------
void ShaderProgram::setUniform(const GLchar* name, int value) {
    GLint loc = getUniformLocation(name);
    glUniform1i(loc, value);
}

//-----------------------------------------------------------------------------
// ユニフォーム変数の設定（float）
//-----------------------------------------------------------------------------
void ShaderProgram::setUniform(const GLchar* name, float value) {
    GLint loc = getUniformLocation(name);
    glUniform1f(loc, value);
}
