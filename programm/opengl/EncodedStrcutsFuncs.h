//
// Created by Андрей on 13/10/2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_OPENGL_ENCODEDSTRCUTSFUNCS_H_
#define LAB_03_CG_COURSE_PROGRAMM_OPENGL_ENCODEDSTRCUTSFUNCS_H_

#include "EncodedStructs.h"
#include "sphere.h"

std::vector<SphereEncoded> convertToEncoded(const std::vector<Sphere> &spheres)
{
	std::vector<SphereEncoded>convertedSpheres(spheres.size());
	for (int i = 0; i < spheres.size(); i++)
	{
			Sphere s = spheres[i];
			convertedSpheres[i].center = s.getCenter();
			convertedSpheres[i].r = s.getRadius();
			convertedSpheres[i].M = s.getMaterial();
	}
	return convertedSpheres;
}


#endif //LAB_03_CG_COURSE_PROGRAMM_OPENGL_ENCODEDSTRCUTSFUNCS_H_
