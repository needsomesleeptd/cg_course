bool IntersectRayCyl(Ray ray, Cylinder cyl, out float fraction, out vec3 normal)
{
    vec3 ba = cyl.extr_b - cyl.extr_a;
    vec3 oc = ray.origin - cyl.extr_a;
    float baba = dot(ba, ba);
    float bard = dot(ba, ray.direction);
    float baoc = dot(ba, oc);
    float k2 = baba - bard * bard;
    float k1 = baba * dot(oc, ray.direction) - baoc * bard;
    float k0 = baba * dot(oc, oc) - baoc * baoc - cyl.ra * cyl.ra * baba;
    float h = k1 * k1 - k2 * k0;
    if (h < 0.0)
    {
        return false;
    }
    h = sqrt(h);
    float t = (-k1 - h) / k2;
    // body
    float y = baoc + t * bard;
    if (y > 0.0 && y < baba)
    {
        vec4 normal_n = vec4(t, (oc + t * ray.direction - ba * y / baba) / cyl.ra);
        fraction = normal_n.x;
        normal = normal_n.yzw;
        return true;
    }
    
    t = (((y < 0.0) ? 0.0 : baba) - baoc) / bard;
    if (abs(k1 + k2 * t) < h)
    {
        vec4 normal_n = vec4(t, ba * sign(y) / sqrt(baba));
        fraction = normal_n.x;
        normal = normal_n.yzw;
        return true;
    }
    return false;
}


bool IntersectRayCone(Ray r, Cone s, out float fraction, out vec3 normal) 
{
    vec3 co = r.origin - s.c;

    float a = dot(r.direction, s.v) * dot(r.direction, s.v) - s.cosa * s.cosa;
    float b = 2. * (dot(r.direction, s.v) * dot(co, s.v) - dot(r.direction, co) * s.cosa * s.cosa);
    float c = dot(co, s.v) * dot(co, s.v) - dot(co, co) * s.cosa * s.cosa;

    float det = b * b - 4. * a * c;
    if (det < 0.0f) return false;

    det = sqrt(det);
    float t1 = (-b - det) / (2. * a);
    float t2 = (-b + det) / (2. * a);

    
    float t = t1;
    if (t < 0.0f || t2 > 0.0f && t2 < t) t = t2;
    if (t < 0.) return false;

    vec3 cp = r.origin + t * r.direction - s.c;
    float h = dot(cp, s.v);
    if (h < 0. || h > s.h) return false;

    vec3 n = normalize(cp * dot(s.v, cp) / dot(cp, cp) - s.v);
    fraction = t;
    normal = n;
    return true;
}


bool IntersectRaySphere(Ray ray, Sphere sphere, out float fraction, out vec3 normal)
{
    vec3 L = ray.origin - sphere.center;
    float a = dot(ray.direction, ray.direction);
    float b = 2.0 * dot(L, ray.direction);
    float c = dot(L, L) - sphere.radius * sphere.radius;
    float D = b * b - 4 * a * c;

    if (D < 0.0) return false;

    float r1 = (-b - sqrt(D)) / (2.0 * a);
    float r2 = (-b + sqrt(D)) / (2.0 * a);
    if (r1 < 0.0f && r2 < 0.0f)
    return false;
    fraction = INF;
    if (r1 > 0.0)
    fraction = r1;
    else if (r2 > 0.0 && fraction > r2)
    fraction = r2;
    else
    return false;

    vec3 point = ray.direction * fraction + ray.origin;
    normal = normalize(point - sphere.center);

    return true;
}



bool IntersectRayBox(Ray ray, Box box, out float fraction, out vec3 normal)
{
    vec3 rd = box.rotation * ray.direction;
    vec3 ro = box.rotation * (ray.origin - box.position);

    vec3 m = vec3(1.0) / rd;

    vec3 s = vec3((rd.x < 0.0) ? 1.0 : -1.0,
                  (rd.y < 0.0) ? 1.0 : -1.0,
                  (rd.z < 0.0) ? 1.0 : -1.0);
    vec3 t1 = m * (-ro + s * box.halfSize);
    vec3 t2 = m * (-ro - s * box.halfSize);

    float tN = max(max(t1.x, t1.y), t1.z);
    float tF = min(min(t2.x, t2.y), t2.z);

    if (tN > tF || tF < 0.0) return false;

    mat3 txi = transpose(box.rotation);

    if (t1.x > t1.y && t1.x > t1.z)
    normal = txi[0] * s.x;
    else if (t1.y > t1.z)
    normal = txi[1] * s.y;
    else
    normal = txi[2] * s.z;

    fraction = tN;

    return true;
}