#include "TextRenderer.h"

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

//=============================================================================
// コンストラクタ
//=============================================================================
TextRenderer::TextRenderer()
    : textVAO_(0)
    , textVBO_(0)
    , textShader_(0)
    , rectVAO_(0)
    , rectVBO_(0)
    , rectShader_(0)
    , fontSize_(24.0f)
    , initialized_(false)
    , windowWidth_(1024)
    , windowHeight_(768)
{
}

//=============================================================================
// デストラクタ
//=============================================================================
TextRenderer::~TextRenderer() {
    for (auto& pair : characters_) {
        if (pair.second.textureID != 0) {
            glDeleteTextures(1, &pair.second.textureID);
        }
    }
    characters_.clear();

    if (textVAO_ != 0) glDeleteVertexArrays(1, &textVAO_);
    if (textVBO_ != 0) glDeleteBuffers(1, &textVBO_);
    if (textShader_ != 0) glDeleteProgram(textShader_);

    if (rectVAO_ != 0) glDeleteVertexArrays(1, &rectVAO_);
    if (rectVBO_ != 0) glDeleteBuffers(1, &rectVBO_);
    if (rectShader_ != 0) glDeleteProgram(rectShader_);
}

//=============================================================================
// システムフォントで初期化
//=============================================================================
bool TextRenderer::initWithSystemFont(float fontSize) {
    std::vector<std::string> fontPaths = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf"
    };

    for (const auto& path : fontPaths) {
        std::ifstream f(path);
        if (f.good()) {
            f.close();
            std::cout << "TextRenderer: Found font: " << path << std::endl;
            return init(path, fontSize);
        }
    }

    std::cerr << "TextRenderer: No system font found!" << std::endl;
    return false;
}

//=============================================================================
// 初期化
//=============================================================================
bool TextRenderer::init(const std::string& fontPath, float fontSize) {
    fontSize_ = fontSize;

    if (!createTextShader()) {
        std::cerr << "TextRenderer: Failed to create text shader" << std::endl;
        return false;
    }

    if (!createRectShader()) {
        std::cerr << "TextRenderer: Failed to create rect shader" << std::endl;
        return false;
    }

    if (!loadFont(fontPath, fontSize)) {
        std::cerr << "TextRenderer: Failed to load font: " << fontPath << std::endl;
        return false;
    }

    setupTextBuffers();
    setupRectBuffers();
    setWindowSize(windowWidth_, windowHeight_);

    initialized_ = true;
    std::cout << "TextRenderer: Initialized successfully" << std::endl;
    return true;
}

//=============================================================================
// フォント読み込み
//=============================================================================
bool TextRenderer::loadFont(const std::string& fontPath, float fontSize) {
    std::ifstream file(fontPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return false;
    }

    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<unsigned char> fontBuffer(fileSize);
    if (!file.read(reinterpret_cast<char*>(fontBuffer.data()), fileSize)) {
        return false;
    }
    file.close();

    stbtt_fontinfo fontInfo;
    if (!stbtt_InitFont(&fontInfo, fontBuffer.data(), 0)) {
        return false;
    }

    float scale = stbtt_ScaleForPixelHeight(&fontInfo, fontSize);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    for (unsigned char c = 32; c < 127; c++) {
        int width, height, xoff, yoff;
        unsigned char* bitmap = stbtt_GetCodepointBitmap(
            &fontInfo, 0, scale, c, &width, &height, &xoff, &yoff);

        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0,
                     GL_RED, GL_UNSIGNED_BYTE, bitmap);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        int advanceWidth, leftSideBearing;
        stbtt_GetCodepointHMetrics(&fontInfo, c, &advanceWidth, &leftSideBearing);

        Character character = {
            texture,
            glm::ivec2(width, height),
            glm::ivec2(xoff, yoff),
            static_cast<int>(advanceWidth * scale)
        };
        characters_[c] = character;

        stbtt_FreeBitmap(bitmap, nullptr);
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    return true;
}

//=============================================================================
// テキストシェーダー作成
//=============================================================================
bool TextRenderer::createTextShader() {
    const char* vs = R"(
        #version 330 core
        layout (location = 0) in vec4 vertex;
        out vec2 TexCoords;
        uniform mat4 projection;
        void main() {
            gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);
            TexCoords = vertex.zw;
        }
    )";

    const char* fs = R"(
        #version 330 core
        in vec2 TexCoords;
        out vec4 color;
        uniform sampler2D text;
        uniform vec3 textColor;
        void main() {
            float alpha = texture(text, TexCoords).r;
            color = vec4(textColor, alpha);
        }
    )";

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vs, nullptr);
    glCompileShader(vertexShader);

    int success;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glDeleteShader(vertexShader);
        return false;
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fs, nullptr);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        return false;
    }

    textShader_ = glCreateProgram();
    glAttachShader(textShader_, vertexShader);
    glAttachShader(textShader_, fragmentShader);
    glLinkProgram(textShader_);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    glGetProgramiv(textShader_, GL_LINK_STATUS, &success);
    return success != 0;
}

//=============================================================================
// 矩形シェーダー作成
//=============================================================================
bool TextRenderer::createRectShader() {
    const char* vs = R"(
        #version 330 core
        layout (location = 0) in vec2 position;
        uniform mat4 projection;
        void main() {
            gl_Position = projection * vec4(position, 0.0, 1.0);
        }
    )";

    const char* fs = R"(
        #version 330 core
        out vec4 FragColor;
        uniform vec4 rectColor;
        void main() {
            FragColor = rectColor;
        }
    )";

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vs, nullptr);
    glCompileShader(vertexShader);

    int success;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glDeleteShader(vertexShader);
        return false;
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fs, nullptr);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        return false;
    }

    rectShader_ = glCreateProgram();
    glAttachShader(rectShader_, vertexShader);
    glAttachShader(rectShader_, fragmentShader);
    glLinkProgram(rectShader_);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    glGetProgramiv(rectShader_, GL_LINK_STATUS, &success);
    return success != 0;
}

//=============================================================================
// テキスト用バッファセットアップ
//=============================================================================
void TextRenderer::setupTextBuffers() {
    glGenVertexArrays(1, &textVAO_);
    glGenBuffers(1, &textVBO_);

    glBindVertexArray(textVAO_);
    glBindBuffer(GL_ARRAY_BUFFER, textVBO_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, nullptr, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

//=============================================================================
// 矩形用バッファセットアップ
//=============================================================================
void TextRenderer::setupRectBuffers() {
    glGenVertexArrays(1, &rectVAO_);
    glGenBuffers(1, &rectVBO_);

    glBindVertexArray(rectVAO_);
    glBindBuffer(GL_ARRAY_BUFFER, rectVBO_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 2, nullptr, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

//=============================================================================
// ウィンドウサイズ設定
//=============================================================================
void TextRenderer::setWindowSize(int width, int height) {
    windowWidth_ = width;
    windowHeight_ = height;
    projection_ = glm::ortho(0.0f, static_cast<float>(width),
                             0.0f, static_cast<float>(height));
}

//=============================================================================
// フォントサイズ設定
//=============================================================================
void TextRenderer::setFontSize(float size) {
    fontSize_ = size;
}

//=============================================================================
// テキストの幅を計算
//=============================================================================
float TextRenderer::getTextWidth(const std::string& text, float scale) const {
    float maxWidth = 0.0f;
    float currentWidth = 0.0f;

    for (char c : text) {
        if (c == '\n') {
            maxWidth = std::max(maxWidth, currentWidth);
            currentWidth = 0.0f;
        } else {
            auto it = characters_.find(c);
            if (it != characters_.end()) {
                currentWidth += it->second.advance * scale;
            }
        }
    }
    return std::max(maxWidth, currentWidth);
}

//=============================================================================
// 行数を取得
//=============================================================================
int TextRenderer::getLineCount(const std::string& text) const {
    int lines = 1;
    for (char c : text) {
        if (c == '\n') lines++;
    }
    return lines;
}

//=============================================================================
// テキストの高さを計算
//=============================================================================
float TextRenderer::getTextHeight(const std::string& text, float scale, float lineSpacing) const {
    return getLineCount(text) * fontSize_ * scale * lineSpacing;
}

//=============================================================================
// 矩形描画
//=============================================================================
void TextRenderer::drawRect(float x, float y, float width, float height, glm::vec4 color) {
    if (!initialized_) return;

    GLboolean depthTest, blend;
    glGetBooleanv(GL_DEPTH_TEST, &depthTest);
    glGetBooleanv(GL_BLEND, &blend);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);

    float yBottom = windowHeight_ - y - height;
    float yTop = windowHeight_ - y;

    float vertices[6][2] = {
        { x,         yBottom },
        { x + width, yBottom },
        { x + width, yTop },
        { x,         yBottom },
        { x + width, yTop },
        { x,         yTop }
    };

    glUseProgram(rectShader_);
    glUniformMatrix4fv(glGetUniformLocation(rectShader_, "projection"), 1, GL_FALSE, &projection_[0][0]);
    glUniform4f(glGetUniformLocation(rectShader_, "rectColor"), color.r, color.g, color.b, color.a);

    glBindVertexArray(rectVAO_);
    glBindBuffer(GL_ARRAY_BUFFER, rectVBO_);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    if (depthTest) glEnable(GL_DEPTH_TEST);
    if (!blend) glDisable(GL_BLEND);
}

//=============================================================================
// 1行テキスト描画
//=============================================================================
void TextRenderer::renderText(const std::string& text, float x, float y,
                              float scale, glm::vec3 color) {
    if (!initialized_) return;

    GLboolean depthTest, blend;
    glGetBooleanv(GL_DEPTH_TEST, &depthTest);
    glGetBooleanv(GL_BLEND, &blend);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);

    glUseProgram(textShader_);
    glUniformMatrix4fv(glGetUniformLocation(textShader_, "projection"), 1, GL_FALSE, &projection_[0][0]);
    glUniform3f(glGetUniformLocation(textShader_, "textColor"), color.x, color.y, color.z);

    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(textVAO_);

    float yPos = windowHeight_ - y - fontSize_ * scale;

    for (char c : text) {
        auto it = characters_.find(c);
        if (it == characters_.end()) continue;

        const Character& ch = it->second;

        float xpos = x + ch.bearing.x * scale;
        float ypos = yPos - (ch.size.y + ch.bearing.y) * scale;
        float w = ch.size.x * scale;
        float h = ch.size.y * scale;

        float vertices[6][4] = {
            { xpos,     ypos + h, 0.0f, 0.0f },
            { xpos,     ypos,     0.0f, 1.0f },
            { xpos + w, ypos,     1.0f, 1.0f },
            { xpos,     ypos + h, 0.0f, 0.0f },
            { xpos + w, ypos,     1.0f, 1.0f },
            { xpos + w, ypos + h, 1.0f, 0.0f }
        };

        glBindTexture(GL_TEXTURE_2D, ch.textureID);
        glBindBuffer(GL_ARRAY_BUFFER, textVBO_);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        x += ch.advance * scale;
    }

    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);

    if (depthTest) glEnable(GL_DEPTH_TEST);
    if (!blend) glDisable(GL_BLEND);
}

//=============================================================================
// 複数行テキスト描画
//=============================================================================
void TextRenderer::renderTextMultiline(const std::string& text, float x, float y,
                                       float scale, glm::vec3 color, float lineSpacing) {
    if (!initialized_) return;

    float lineHeight = fontSize_ * scale * lineSpacing;
    float currentY = y;
    std::string line;

    for (char c : text) {
        if (c == '\n') {
            if (!line.empty()) renderText(line, x, currentY, scale, color);
            line.clear();
            currentY += lineHeight;
        } else {
            line += c;
        }
    }
    if (!line.empty()) renderText(line, x, currentY, scale, color);
}

//=============================================================================
// 背景付き複数行テキスト描画
//=============================================================================
void TextRenderer::renderTextWithBackground(const std::string& text, float x, float y,
                                            float scale, glm::vec3 textColor, glm::vec4 bgColor,
                                            float padding, float lineSpacing) {
    if (!initialized_) return;

    float textWidth = getTextWidth(text, scale);
    float textHeight = getTextHeight(text, scale, lineSpacing);

    drawRect(x - padding, y - padding, textWidth + padding * 2, textHeight + padding * 2, bgColor);
    renderTextMultiline(text, x, y, scale, textColor, lineSpacing);
}
