vec4 RayTrace(Ray primary_ray, PrimitiveArrLens lens) {
    vec4 resColor = vec4(0, 0, 0, 1.0);

    Intersection inters;
    inters.t = INF;
    Intersection inters_light;
    inters_light.t = INF;

    vec4 addColor;
    vec4 depthFatigue = vec4(1.0, 1.0, 1.0, 1.0);

    Material lightSourcematerial = Material(vec3(1.0, 1.0, 1.0), vec3(0.0f));
    Sphere lightSourceSphere = Sphere(lightSource.position, R_LSOURCE, lightSourcematerial);

    float fr;
    vec3 nr;
    bool lighInter = IntersectRaySphere(primary_ray, lightSourceSphere, fr, nr);

    for (int i = 0; i < MAX_DEPTH; i++)
    {
        inters = findIntersection(primary_ray, spheres, boxes, cylinders, lens);
        if (i == 0)
        {
            if (abs(inters.t - INF) < EPS)
            {

                if (lighInter)
                {
                    resColor = vec4(lightSource.intensivity, 1);
                }
                else
                {
                    resColor = vec4(0.1, 0.2, 0.3, 1.0);
                }
                break;
            }
            else if (lighInter && fr > 0.0 && fr < inters.t)
            {
                resColor = vec4(lightSource.intensivity, 1);
                break;
            }

        }




        vec3 lightVector = normalize(lightSource.position - inters.point);

        vec3 shadow_orig = dot(lightVector, inters.normal) < 0 ? inters.point - inters.normal * 1e-3 : inters.point + inters.normal * 1e-3;
        Ray lightRay = Ray(shadow_orig, lightVector);

        inters_light = findIntersection(lightRay, spheres, boxes, cylinders, lens);

        float light_inter_dist = length(inters.point - lightSource.position);
        float light_point_dist = length(shadow_orig  - inters_light.point);

        if (abs(inters_light.t - INF) > EPS && light_inter_dist > light_point_dist)
        {

            addColor = vec4(inters.material.color * inters.material.lightKoefs[0],1.0);

            primary_ray = calculateReflected(primary_ray, inters);


        }
        else
        {
            addColor = Phong(inters, primary_ray);
        }


        resColor += depthFatigue * addColor;
        depthFatigue *= inters.material.lightKoefs[2];
    }
    return resColor;
}