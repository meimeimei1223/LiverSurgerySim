// vertex_shader.glsl
#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in vec4 aColor;  // 追加：頂点カラー

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform bool useVertexColor;

out vec3 FragPos;
out vec3 Normal;
out vec4 VertexColor;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    VertexColor = useVertexColor ? aColor : vec4(1.0);
    
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
