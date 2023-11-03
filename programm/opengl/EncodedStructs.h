//
// Created by Андрей on 13/10/2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_OPENGL_TRINANGLE_ENCODED_H_
#define LAB_03_CG_COURSE_PROGRAMM_OPENGL_TRINANGLE_ENCODED_H_

struct Triangle_encoded {
	vec3 p1, p2, p3;    // Vertex coordinates
	vec3 n1, n2, n3;    // Vertex normal
	vec3 emissive;      // Self luminous parameters
	vec3 baseColor;     // colour
	vec3 param1;        // (subsurface, metallic, specular)
	vec3 param2;        // (specularTint, roughness, anisotropic)
	vec3 param3;        // (sheen, sheenTint, clearcoat)
	vec3 param4;        // (clearcoatGloss, IOR, transmission)
};

#endif //LAB_03_CG_COURSE_PROGRAMM_OPENGL_TRINANGLE_ENCODED_H_
