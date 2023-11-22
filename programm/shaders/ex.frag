#version 420
//precision highp float;

#define EPS 1e-3
#define INF 1000000.0

#define SPHERE_COUNT 3

#define  K   0.1f

#define MAX_DEPTH 3

#define PI acos(0) * 2.0f


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





struct Intersection {
    float t;
    vec3 point;
    vec3 normal;
    Material material;
    Ray tracedRay;
};


struct Square {
    vec3 v1;
    vec3 v2;
    vec3 v3;
    vec3 v4;
    int material_ind;
    vec3 color;
};

struct Light {
    vec3 position;
    vec3 intensivity;
};

struct Triangle {
    vec3 v1;
    vec3 v2;
    vec3 v3;
    int material_ind;
    vec3 color;
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

vec3 Random3D()
{
    return vec3(rand(vec2(1.0f)), rand(vec2(3.0f)), rand(vec2(5.0f)));
}


Ray GenerateRay(Camera camera, vec3 texcoord, vec2 viewportSize) {

    //texcoord.xy = vec2(texcoord.x / scale.x, texcoord.y / scale.y);

    float fov = 0.45f;
    vec2 texDiff = 0.5 * vec2(1.0 - 2.0 * texcoord.x, 2.0 * texcoord.y - 1.0);
    vec2 angleDiff = texDiff * vec2(viewportSize.x / viewportSize.y, 1.0) * tan(fov * 0.5);

    vec3 rayDirection = normalize(vec3(angleDiff, 1.0f));
    mat3 viewToWorld = mat3(
        -normalize(camera.right),
        normalize(camera.up),
        normalize(camera.view)
    );
    return Ray(camera.position, viewToWorld * rayDirection);
}



bool IntersectRaySphere(Ray ray, Sphere sphere, out float fraction, out vec3 normal)
{
    vec3 L = ray.origin - sphere.center;
    float a = dot(ray.direction, ray.direction);
    float b = 2.0 * dot(ray.origin, ray.direction);
    float c = dot(ray.origin, ray.origin) - sphere.radius * sphere.radius;
    float D = b * b - 4 * a * c;

    if (D < 0.0) return false;

    float r1 = (-b - sqrt(D)) / (2.0 * a);
    float r2 = (-b + sqrt(D)) / (2.0 * a);

    if (r1 > 0.0)
    fraction = r1;
    else if (r2 > 0.0)
    fraction = r2;
    else
    return false;

    vec3 point = normalize(ray.direction * fraction + ray.origin);
    normal = normalize(sphere.center - point);

    return true;
}

/*bool IntersectTriangle(Ray ray, vec3 v1, vec3 v2, vec3 v3, out float time) {
    time = -1;
    vec3 A = v2 - v1;
    vec3 B = v3 - v1;
    vec3 N = cross(A, B);
    float NdotRayDirection = dot(N, ray.direction);
    if (abs(NdotRayDirection) < 0.001)   return false;
    float d = dot(N, v1);
    float t = -1. * (dot(N, ray.origin) - d) / NdotRayDirection;
    if (t < 0) return false;
    vec3 P = ray.origin + t * ray.direction;

    vec3 edge1 = v2 - v1;
    vec3 VP1 = P - v1;
    vec3 C = cross(edge1, VP1);
    if (dot(N, C) < 0)  return false;
    vec3 edge2 = v3 - v2;
    vec3 VP2 = P - v2;
    C = cross(edge2, VP2);
    if (dot(N, C) < 0)   return false;
    vec3 edge3 = v1 - v3;
    vec3 VP3 = P - v3;
    C = cross(edge3, VP3);
    if (dot(N, C) < 0)   return false;
    time = t;
    return true;
}*/

bool IntersectSquare(Ray ray, vec3 v1, vec3 v2, vec3 v3, vec3 v4, out float time) {
    time = -1.0;
    vec3 A = v2 - v1;
    vec3 B = v4 - v1;
    vec3 N = cross(A, B);
    float NdotRayDirection = dot(N, ray.direction);
    if (abs(NdotRayDirection) < 0.001)   return false;
    float d = dot(N, v1);
    float t = -1. * (dot(N, ray.origin) - d) / NdotRayDirection;
    if (t < 0.0) return false;
    vec3 P = ray.origin + t * ray.direction;

    vec3 edge1 = v2 - v1;
    vec3 VP1 = P - v1;
    vec3 C = cross(edge1, VP1);
    if (dot(N, C) < 0.0) return false;
    vec3 edge2 = v3 - v2;
    vec3 VP2 = P - v2;
    C = cross(edge2, VP2);
    if (dot(N, C) < 0.0) return false;
    vec3 edge3 = v4 - v3;
    vec3 VP3 = P - v3;
    C = cross(edge3, VP3);
    if (dot(N, C) < 0.0) return false;
    vec3 edge4 = v1 - v4;
    vec3 VP4 = P - v4;
    C = cross(edge4, VP4);
    if (dot(N, C) < 0.0) return false;
    time = t;
    return true;
}

bool Intersect(Ray ray, float start, float final, inout Intersection intersect) {
    bool result = false;
    float time = start;
    intersect.t = final;

/*for (int i = 0;i < vector_size; i++) {
        if (IntersectSphere(sphere_data[i], ray, time) && time < intersect.time) {
            intersect.time = time;
            intersect.point = ray.origin + ray.direction * time;
            intersect.normal = normalize(intersect.point - sphere_data[i].center);
            intersect.color = sphere_data[i].color;
            intersect.material_ind = sphere_data[i].material_ind;
            intersect.light_coeffs = material.light_coeffs;
            result = true;
        }
    }*/
    return result;
}
/*for (int i = 0;i < 4; i++) {
        if (IntersectTriangle(ray, tri[i].v1, tri[i].v2, tri[i].v3, time) && time < intersect.time) {
            intersect.point = ray.origin + ray.direction * time;
            intersect.normal = normalize(cross(tri[i].v1 - tri[i].v2, tri[i].v3 - tri[i].v2));
            intersect.color = tri[i].color;
            intersect.material_ind = tri[i].material_ind;
            intersect.light_coeffs = material.light_coeffs;
            intersect.time = time;
            result = true;
        }
    }*/
/*for (int i = 0;i < 6; i++) {
        if (IntersectSquare(ray, sq[i].v1, sq[i].v2, sq[i].v3, sq[i].v4, time) && time < intersect.time) {
            intersect.point = ray.origin + ray.direction * time;
            intersect.normal = normalize(cross(sq[i].v1 - sq[i].v2, sq[i].v3 - sq[i].v2));
            intersect.color = sq[i].color;
            intersect.material_ind = sq[i].material_ind;
            intersect.light_coeffs = material.light_coeffs;
            intersect.time = time;
            result = true;
        }
    }
    return result;*/
//}



vec4 Phong(Intersection intersect) {

    vec3 finalColor = vec3(0.0);
    for (int i = 0; i < MAX_DEPTH; i++)
    {

        vec3 lightVector = normalize(normalize(lightSource.position) - normalize(intersect.point));
        Ray lightRay = Ray(lightSource.position, lightVector);
        vec3 shapeNormal = normalize(intersect.normal);
        vec3 ambientIntensivity = intersect.material.lightKoefs[0] * intersect.material.color;


        finalColor += ambientIntensivity * intersect.material.color;
        float diffuseLight = dot(shapeNormal, lightVector);
        Material shapeMaterial = intersect.material;


        //std::cout << " diffuseLight" << diffuseLight << std::endl;
        float lightIntersect = max(0.0f, diffuseLight);
        finalColor += lightIntersect * shapeMaterial.lightKoefs[1] * intersect.material.color;



        if (shapeMaterial.lightKoefs[2] > 0.0f)
        {
            vec3 reflectedDirection = reflect(intersect.tracedRay.direction, intersect.normal);
            Ray reflected = Ray(intersect.tracedRay.origin, reflectedDirection);
            float specularDot = pow(max(dot(reflected.direction, intersect.tracedRay.direction), 0.0f), 20);
            if (specularDot > 0.0f)
            {


                finalColor += shapeMaterial.lightKoefs[2] * lightSource.intensivity * specularDot;

            }
        }
        vec3 newRayOrigin = intersect.tracedRay.origin + intersect.t * intersect.tracedRay.direction;
        vec3 hemisphereDistributedDirection = NormalOrientedHemispherePoint(vec2(Random3D()), intersect.normal);
        vec3 randomVec = normalize(2.0 * Random3D() - 1.0);

        vec3 tangent = cross(randomVec, intersect.normal);
        vec3 bitangent = cross(intersect.normal, tangent);
        mat3 transform = mat3(tangent, bitangent, intersect.normal);

        vec3 newRayDirection = transform * hemisphereDistributedDirection;


    }



    return vec4(finalColor, 1);
}

Sphere spheres[SPHERE_COUNT];
vec4 RayTrace(Ray primary_ray, int len) {
    vec4 resColor = vec4(0, 0, 0, 0);
    Ray ray = primary_ray;
    float minDistance = INF;
    float D = -1.0;
    vec3 N;
    Intersection inters;
    for (int i = 0; i < SPHERE_COUNT; i++)
    {
        if ((IntersectRaySphere(ray, spheres[i], D, N)))
        {
            if (D < minDistance)
            {
                inters.normal = N;
                inters.tracedRay = ray;
                inters.point = normalize(ray.origin + ray.direction * D);
                inters.material = spheres[i].material;
                inters.t = D;
                minDistance = D;
                //return vec4(1.0, 0.0, 0.5, 1.0);
            }
        }

    }
    if (minDistance == INF)
    {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }

    resColor = Phong(inters);

    return resColor;
}



void main(void) {




    Material material = Material(vec3(0.9f, 0.2f, 0.1f), vec3(0.1f, 0.4f, 0.1f));
    Material new_material = Material(vec3(0.1f, 0.6f, 0.4f), vec3(0.1f, 0.2f, 0.3f));

    Material new_new_material = Material(vec3(0.0f, 0.0f, 0.6f), vec3(0.1f, 0.4f, 0.0f));


    spheres[0] = Sphere(vec3(-0.5f, 0.4f, 0.0f), 0.2f, material);
    spheres[1] = Sphere(vec3(-0.5f, 0.7f, -0.1f), 0.3f, new_material);
    spheres[2] = Sphere(vec3(-0.5f, 0.6f, 0.0f), 0.4f, new_new_material);


    Ray ray = GenerateRay(camera, interpolated_vertex, scale);
    FragColor = RayTrace(ray, SPHERE_COUNT);





}


