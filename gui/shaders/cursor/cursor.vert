#version 450

layout (location = 0) in vec2 a_pos;

layout (set = 0, binding = 1) uniform CoordinateUniformBuffer {
    mat4 u_view_projection;
    mat4 u_model;
};

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    gl_Position = u_view_projection * u_model * vec4(a_pos, 0.0, 1.0);
}