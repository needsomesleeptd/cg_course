#version 420

in vec3 vertex;
out vec4 interpolated_vertex;

void main(void){
    //gl_Position = vec4(vertex, 1.0);
    interpolated_vertex = vec4(vertex,0.5);
}