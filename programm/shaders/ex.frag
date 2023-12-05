#version 420
//precision HIGHP float;

#define EPS 1e-7
#define INF 100000.0

#define SPHERE_COUNT 40
#define BOX_COUNT 40
#define CONES_COUNT 40
#define CYLINDERS_COUNT 40


#define  K   0.1f

#define MAX_DEPTH 5

#define PI acos(0) * 2.0f

#define R_LSOURCE 0.01f


struct Camera {
    vec3 position;
    vec3 view;
    vec3 up;
    vec3 right;
    mat4 inverseProjectionMatrix;
    mat4 inverseViewMatrix;
};

struct Ray {
    vec3 origin;
    vec3 direction;
};

struct Material {
    vec3 color;
    vec3 lightKoefs;
};

struct Sphere {
    vec3 center;
    float radius;
    Material material;
};

struct Plane {
    vec3 point;
    vec3 normal;
    int material_ind;
};

struct Box
{
    vec3 position;
    mat3 rotation;
    vec3 halfSize;
    Material material;
};


struct Cylinder
{
    vec3 extr_a;
    vec3 extr_b;
    float ra;
    Material material;
};



struct Intersection {
    float t;
    vec3 point;
    vec3 normal;
    Material material;
    Ray tracedRay;
};



struct Cone
{
    float cosa;    // half cone angle
    float h;    // height
    vec3 c;        // tip position
    vec3 v;        // axis
    Material material;    // material
};


struct Light {
    vec3 position;
    vec3 intensivity;
};



in vec3 interpolated_vertex;
out vec4 FragColor;

uniform Light lightSource;
uniform Camera camera;
uniform vec2 scale;

uniform mat4 view;
uniform mat4 projection;







vec3 RandomHemispherePoint(vec2 rand)
{
    float cosTheta = sqrt(1.0 - rand.x);
    float sinTheta = sqrt(rand.x);
    float phi = 2.0 * PI * rand.y;
    return vec3(
        cos(phi) * sinTheta,
        sin(phi) * sinTheta,
        cosTheta
    );
}

vec3 NormalOrientedHemispherePoint(vec2 rand, vec3 n)
{
    vec3 v = RandomHemispherePoint(rand);
    return dot(v, n) < 0.0 ? -v : v;

}


float rand(vec2 co)
{

    highp float a = 12.9898;

    highp float b = 78.233;

    highp float c = 43758.5453;

    highp float dt = dot(co.xy, vec2(a, b));

    highp float sn = mod(dt, 3.14);

    return fract(sin(sn) * c);

}

int rand_val = 0;
vec3 Random3D()
{
    rand_val++;
    return vec3(rand(vec2(interpolated_vertex.x + rand_val, rand_val)), rand(vec2(interpolated_vertex.y + rand_val, rand_val)), rand(vec2(interpolated_vertex.z + rand_val, rand_val)));
}


Ray GenerateRay(Camera camera, vec3 texcoord, vec2 viewportSize) {

    float fov = 0.5f;

    vec2 texDiff = vec2(1.0 - 2.0 * texcoord.x, 2.0 * texcoord.y - 1.0);
    vec2 angleDiff = texDiff * vec2(viewportSize.x / viewportSize.y, 1.0) * tan(fov * 0.5);

    vec3 rayDirection = normalize(vec3(angleDiff, 1.0f));



    vec4 rayDirectiondim = view * vec4(rayDirection, 1.0);
    rayDirection = vec3(rayDirectiondim / rayDirectiondim[3]);
    return Ray(camera.position, normalize(rayDirection));

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


vec3 cylNormal(in vec3 p, Cylinder cyl)
{
    vec3 pa = p - cyl.extr_a;
    vec3 ba = cyl.extr_b - cyl.extr_a;
    float baba = dot(ba, ba);
    float paba = dot(pa, ba);
    float h = dot(pa, ba) / baba;
    return vec3(pa - ba * h) / cyl.ra;
}


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
        return false;//no intersection
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
    // caps
    t = (((y < 0.0) ? 0.0 : baba) - baoc) / bard;
    if (abs(k1 + k2 * t) < h)
    {
        vec4 normal_n = vec4(t, ba * sign(y) / sqrt(baba));
        fraction = normal_n.x;
        normal = normal_n.yzw;
        return true;
    }
    return false;//no intersection
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

    // This is a bit messy; there ought to be a more elegant solution.
    float t = t1;
    if (t < 0.0f || (t2 > 0.0f && t2 < t)) t = t2;
    if (t < 0.) return false;

    vec3 cp = r.origin + t * r.direction - s.c;
    float h = dot(cp, s.v);
    if (h < 0. || h > s.h + 0.1) return false;

    vec3 n = normalize(cp * dot(s.v, cp) / dot(cp, cp) - s.v);
    fraction = t;
    normal = n;
    return true;
}


Ray calculateReflected(Ray primary, Intersection intersection)
{
    vec3 idealReflection = normalize(reflect(primary.direction, intersection.normal));

    vec3 reflectedDirection = idealReflection;
    vec3 newRayOrigin = intersection.point;
    newRayOrigin += intersection.normal * 0.02;
    Ray reflected = Ray(newRayOrigin, reflectedDirection);
    return reflected;
}


vec4 Phong(Intersection intersect, out Ray rayReflected) {

    vec3 rayColor = vec3(0.0);
    vec3 finalColor = vec3(0.0);


    vec3 lightVector = normalize(lightSource.position - intersect.point);


    vec3 shapeNormal = intersect.normal;

    vec3 ambientIntensivity = intersect.material.lightKoefs[0] * intersect.material.color;

    rayColor += ambientIntensivity;

    float diffuseLight = dot(shapeNormal, lightVector);
    Material shapeMaterial = intersect.material;


    float lightIntersect = max(0.0f, diffuseLight);
    rayColor += lightIntersect * shapeMaterial.lightKoefs[1] * intersect.material.color;





    Ray reflected = calculateReflected(intersect.tracedRay, intersect);

    float specularDot = pow(max(dot(-reflected.direction, intersect.tracedRay.direction), 0.0f), 5);

    rayColor += shapeMaterial.lightKoefs[2] * lightSource.intensivity * specularDot;




    rayReflected = reflected;
    return vec4(rayColor, 1);
}


uniform Sphere spheres[SPHERE_COUNT];

uniform Box boxes[BOX_COUNT];

uniform Cylinder cylinders[CYLINDERS_COUNT];

uniform Cone cones[CONES_COUNT];

struct PrimitiveArrLens {
    int size_spheres;
    int size_cylinders;
    int size_boxes;
    int size_cones;
};


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

        vec3 shadow_orig = dot(lightVector, inters.normal) <= 0 ? inters.point - inters.normal * 0.02 : inters.point + inters.normal * 0.02;
        Ray lightRay = Ray(shadow_orig, lightVector);
        inters_light = findIntersection(lightRay, spheres, boxes, cylinders, lens);


        if (abs(inters.t - INF) < EPS)
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


Material gen_random_mat()
{
    Material material = Material(normalize(Random3D()), normalize(Random3D()));
    return material;
}


uniform PrimitiveArrLens prLens;


void main(void) {

    Material material = Material(vec3(0.3f, 0.2f, 0.1f), vec3(0.1f, 0.4f, 0.6f));
    Material new_material = Material(vec3(0.1f, 0.9f, 0.4f), vec3(0.1f, 0.4f, 0.3f));

    Material new_new_material = Material(vec3(0.5f, 0.3f, 1.0f), vec3(0.1f, 0.3f, 0.3f));



    Material cylinder_material = Material(vec3(0.5f, 0.3f, 1.0f), vec3(0.1f, 0.4f, 0.4f));


/* spheres[0] = Sphere(vec3(1.0f, 1.4f, 3.1f), 0.6f, material);
    spheres[1] = Sphere(vec3(3.0f, 0.7f, -0.1f), 0.7f, new_material);
    spheres[2] = Sphere(vec3(1.0f, 0.7f, -0.1f), 0.3f, new_new_material);*/

/*boxes[0] = Box(vec3(0.3f, 2.0f, -1.0f), mat3(1.0), vec3(1.0, 1.0, 1.0), gen_random_mat());

    cylinders[0].extr_a = vec3(1.0, 0.0, 1.0);

    cylinders[0].extr_a = vec3(-1.0, 0.0, -1.0);

    cylinders[0].ra = 0.2;

    cylinders[0].material = new_new_material;





    cones[0] = Cone(PI / 6, 1, vec3(1.0, 2.0, 1.0), vec3(0.0, 1.0, 0.0), new_new_material);*/

    Ray ray = GenerateRay(camera, interpolated_vertex, scale);

    const int agr_count = 1;
    for (int i = 0; i < agr_count; i++)
    {
        FragColor += RayTrace(ray, prLens);
    }
    FragColor /= agr_count;





}


