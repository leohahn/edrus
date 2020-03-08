#version 450

layout (set = 0, binding = 0) uniform UBO {
    vec4 u_color;
};

layout (location = 0) out vec4 o_color;

void main() {
    o_color = u_color;
}