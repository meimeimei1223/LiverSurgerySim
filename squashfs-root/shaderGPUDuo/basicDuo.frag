// fragment_shader.glsl
#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec4 VertexColor;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 objectColor;
uniform bool useVertexColor;

out vec4 FragColor;

void main() {
    vec3 baseColor = useVertexColor ? VertexColor.rgb : objectColor;
    float alpha = useVertexColor ? VertexColor.a : 1.0;
    
    // 簡易ライティング
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    
    vec3 ambient = 0.3 * baseColor;
    vec3 diffuse = diff * baseColor;
    
    FragColor = vec4(ambient + diffuse, alpha);
}
