// fragment_shader.glsl
#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec4 VertexColor;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 objectColor;
uniform float objectAlpha;      // 追加: アルファ値
uniform bool useVertexColor;

out vec4 FragColor;

void main() {
    vec3 baseColor = useVertexColor ? VertexColor.rgb : objectColor;
    float alpha = useVertexColor ? VertexColor.a : objectAlpha;  // 修正: objectAlphaを使用
    
    // 簡易ライティング
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    
    vec3 ambient = 0.3 * baseColor;
    vec3 diffuse = diff * baseColor;
    
    FragColor = vec4(ambient + diffuse, alpha);
}
