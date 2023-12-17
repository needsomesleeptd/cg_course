Intersection findIntersection(Ray ray, Sphere spheres[SPHERE_COUNT], Box boxes[BOX_COUNT], Cylinder cylinders[CYLINDERS_COUNT], PrimitiveArrLens sizes)
{
    float minDistance = INF;
    float D = INF;
    vec3 N;
    Intersection inters;
    inters.t = minDistance;
    for (int i = 0; i < sizes.size_spheres; i++)
    {
        if (IntersectRaySphere(ray, spheres[i], D, N))
        {
            if (D < minDistance && D > 0.0f)
            {
                inters.normal = N;
                inters.tracedRay = ray;
                inters.point = ray.origin + ray.direction * D;
                inters.material = spheres[i].material;
                inters.t = D;
                minDistance = D;
            }
        }

    }



    for (int i = 0; i < sizes.size_boxes; i++)
    {
        if (IntersectRayBox(ray, boxes[i], D, N))
        {
            if (D < minDistance && D > 0.0f)
            {
                inters.normal = N;
                inters.tracedRay = ray;
                inters.point = ray.origin + ray.direction * D;
                inters.material = boxes[i].material;
                inters.t = D;
                minDistance = D;
            }
        }
    }


    for (int i = 0; i < sizes.size_cylinders; i++)
    {
        if (IntersectRayCyl(ray, cylinders[i], D, N))
        {

            if (D < minDistance && D > 0.0f)
            {
                inters.normal = N;
                inters.tracedRay = ray;
                inters.point = ray.origin + ray.direction * D;
                inters.material = cylinders[i].material;
                inters.t = D;
                minDistance = D;
            }
        }
    }

    for (int i = 0; i < sizes.size_cones; i++)
    {
        if (IntersectRayCone(ray, cones[i], D, N))
        {

            if (D < minDistance && D > 0.0f)
            {
                inters.normal = N;
                inters.tracedRay = ray;
                inters.point = ray.origin + ray.direction * D;
                inters.material = cones[i].material;
                inters.t = D;
                minDistance = D;
            }
        }
    }


    return inters;

}