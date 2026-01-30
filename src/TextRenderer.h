#ifndef TEXT_RENDERER_H
#define TEXT_RENDERER_H

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <iostream>

class TextRenderer {
public:
    TextRenderer();
    ~TextRenderer();

    bool init(const std::string& fontPath, float fontSize = 24.0f);
    bool initWithSystemFont(float fontSize = 24.0f);

    void renderText(const std::string& text, float x, float y,
                    float scale = 1.0f, glm::vec3 color = glm::vec3(1.0f));

    void renderTextMultiline(const std::string& text, float x, float y,
                             float scale = 1.0f, glm::vec3 color = glm::vec3(1.0f),
                             float lineSpacing = 1.2f);

    void renderTextWithBackground(const std::string& text, float x, float y,
                                  float scale = 1.0f,
                                  glm::vec3 textColor = glm::vec3(1.0f),
                                  glm::vec4 bgColor = glm::vec4(0.0f, 0.0f, 0.0f, 0.7f),
                                  float padding = 10.0f,
                                  float lineSpacing = 1.2f);

    void drawRect(float x, float y, float width, float height, glm::vec4 color);

    void setWindowSize(int width, int height);
    void setFontSize(float size);

    bool isInitialized() const { return initialized_; }
    float getTextWidth(const std::string& text, float scale = 1.0f) const;
    float getTextHeight(const std::string& text, float scale = 1.0f, float lineSpacing = 1.2f) const;
    int getLineCount(const std::string& text) const;

private:
    struct Character {
        GLuint textureID;
        glm::ivec2 size;
        glm::ivec2 bearing;
        int advance;
    };

    std::map<char, Character> characters_;

    // テキスト用
    GLuint textVAO_;
    GLuint textVBO_;
    GLuint textShader_;

    // 矩形用
    GLuint rectVAO_;
    GLuint rectVBO_;
    GLuint rectShader_;

    glm::mat4 projection_;
    float fontSize_;
    bool initialized_;
    int windowWidth_;
    int windowHeight_;

    bool loadFont(const std::string& fontPath, float fontSize);
    bool createTextShader();
    bool createRectShader();
    void setupTextBuffers();
    void setupRectBuffers();
};

#endif
