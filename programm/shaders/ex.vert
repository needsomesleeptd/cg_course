#version 330
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
out vec3 interpolated_vertex;

uniform  mat4 view;
uniform  mat4 projection;

void main()
{
    gl_Position =  vec4(position, 1);
    interpolated_vertex = vec3(gl_Position);
}
