vec4 RayTrace(Ray primary_ray, PrimitiveArrLens lens) {
    vec4 resColor = vec4(0, 0, 0, 1.0);

    Intersection inters;
    inters.t = INF;
    Intersection inters_light;
    inters_light.t = INF;

    vec4 addColor;

    Material lightSourcematerial = Material(vec3(1.0, 1.0, 1.0), vec3(0.0f));
    Sphere lightSourceSphere = Sphere(lightSource.position, R_LSOURCE, lightSourcematerial);

    float fr;
    vec3 nr;

    for (int i = 0; i < MAX_DEPTH; i++)
    {
        inters = findIntersection(primary_ray, spheres, boxes, cylinders, lens);
        if (i == 0 && abs(inters.t - INF) < EPS)
        {

            if (IntersectRaySphere(primary_ray, lightSourceSphere, fr, nr) && fr > 0.0)
            {
                resColor = vec4(lightSource.intensivity, 1);
            }
            else
            {
                resColor = vec4(0.1, 0.2, 0.3, 1.0);
            }
            break;
        }


        if (i == 0)
        {
            if (IntersectRaySphere(primary_ray, lightSourceSphere, fr, nr) && fr > 0.0 && fr < inters.t)
            {
                resColor = vec4(lightSource.intensivity, 1);
                break;
            }
        }

       
        vec3 lightVector = normalize(lightSource.position - inters.point);

        vec3 shadow_orig = dot(lightVector, inters.normal) < 0 ? inters.point - inters.normal * 1e-5 : inters.point + inters.normal * 1e-5;
        Ray lightRay = Ray(shadow_orig, lightVector);
        inters_light = findIntersection(lightRay, spheres, boxes, cylinders, lens);


        if (!(inters_light.t < INF))
        addColor = Phong(inters, primary_ray);
        else
        addColor = vec4(inters.material.color * inters.material.lightKoefs[0], 1.0);

        resColor += addColor;
        resColor = resColor * inters.material.lightKoefs[2];

    }

    return resColor;
}
