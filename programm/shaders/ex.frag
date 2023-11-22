#version 420
//precision highp float;

#define EPS 1e-3
#define INF 1e6

#define  K   0.1f

#define MAX_DEPTH 3


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

//uniform Light lightSource;
uniform Camera camera;
uniform vec2 scale;

uniform mat4 view;
uniform mat4 projection;

//uniform vec3 light_pos;
//uniform int vector_size;*/
Light lightSource = Light(vec3(0.0, 0.1, 0.0), vec3(1.0));
/*layout(std430, binding = 0) buffer SphereBuffer {
    vec3 sphere_data[];
};*/

/*layout(std430,binding = 1) buffer PlaneBuffer{
    Plane plane_data[];
};*/
//Camera camera = Camera(vec3(0.0,0.0,-1),vec3(0.0,0.0,1.0),vec3(0.0,1.0,0.0),vec3(1.0,0.0,0.0),mat4(1),mat4(1));
//camera.inverseProjectionMatrix = mat4(1);
//uniform Camera camera;




Ray GenerateRay(Camera camera, vec3 texcoord, vec2 viewportSize) {

    //texcoord.xy = vec2(texcoord.x / scale.x, texcoord.y / scale.y);
    float fov = 0.45f;
    vec2 texDiff = 0.5 * vec2(1.0 - 2.0 * texcoord.x, 2.0 * texcoord.y - 1.0);
    vec2 angleDiff = texDiff * vec2(viewportSize.x / viewportSize.y, 1.0) * tan(fov);

    vec3 rayDirection = normalize(vec3(angleDiff, 1.0f));
    mat3 viewToWorld = mat3(
        -camera.right,
        camera.up,
        camera.view
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

/*float Shadow(vec3 pos_light, Intersection intersect) {
    float shad = 1.0;
    vec3 direction = normalize(pos_light - intersect.point);
    float dist_light = distance(pos_light, intersect.point);
    vec3 qwe = direction * EPS;
    Ray shad_ray = Ray(intersect.point + qwe, direction);
    Intersection shad_intersect;
    shad_intersect.time = INF;
    if (Intersect(shad_ray, 0, dist_light, shad_intersect)) {
        shad = 0.0;
    }
    return shad;
}*/

vec4 Phong(Intersection intersect) {


    vec3 lightVector = normalize(normalize(lightSource.position) - normalize(intersect.point));
    Ray lightRay = Ray(lightSource.position, lightVector);
    vec3 shapeNormal = normalize(intersect.normal);
    vec3 ambientIntensivity = intersect.material.lightKoefs[0] * intersect.material.color;
    vec3 finalColor = vec3(0.0);

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
        float specularDot = pow(max(dot(reflected.direction, intersect.tracedRay.direction),0.0f), 20);
        if (specularDot > 0.0f)
        {

            finalColor += shapeMaterial.lightKoefs[2] *  lightSource.intensivity * specularDot;

        }
    }


    return vec4(finalColor,1);
}

vec4 RayTrace(Ray primary_ray, Sphere sphere) {
    vec4 resColor = vec4(0, 0, 0, 0);
    Ray ray = primary_ray;
    float minDistance = INF;
    float D;
    vec3 N;

    Intersection inters;
    if (IntersectRaySphere(ray,sphere,D,N) && D < minDistance)
    {
        inters.normal = N;
        inters.tracedRay = ray;
        inters.point = normalize(ray.origin + ray.direction * D);
        inters.material = sphere.material;
        inters.t = D;
        resColor += Phong(inters);
        minDistance = D;
    }
    else
    {
        resColor += vec4(0.3,0.2,0.1,1.0);
    }



    return resColor;
}


/*Triangle tri[4];
Square sq[6];*/

void main(void) {
    //FragColor = vec4(abs(interpolated_vertex.xy), 0, 1.0);

/*tri[0] = Triangle(vec3(-10, -10, -30), vec3(10, -10, -30), vec3(0, 5, -25), 0, vec3(1, 1, 0));
    tri[1] = Triangle(vec3(-10, -10, -30), vec3(10, -10, -30), vec3(0, -10, -10), 0, vec3(1, 1, 0));
    tri[2] = Triangle(vec3(0, 5, -25), vec3(10, -10, -30), vec3(0, -10, -10), 0, vec3(1, 1, 0));
    tri[3] = Triangle(vec3(0, 5, -25), vec3(0, -10, -10), vec3(-10, -10, -30), 0, vec3(1, 1, 0));

    sq[0] = Square(vec3(10, 0, 0), vec3(20, 0, 0), vec3(20, 10, 0), vec3(10, 10, 0), 0, vec3(0.52, 0, 0.52));
    sq[1] = Square(vec3(20, 0, 0), vec3(20, 0, 10), vec3(20, 10, 10), vec3(20, 10, 0), 0, vec3(0.52, 0, 0.52));
    sq[2] = Square(vec3(20, 0, 10), vec3(10, 0, 10), vec3(10, 10, 10), vec3(20, 10, 10), 0, vec3(0.52, 0, 0.52));
    sq[3] = Square(vec3(10, 0, 10), vec3(10, 0, 0), vec3(10, 10, 0), vec3(10, 10, 10), 0, vec3(0.52, 0, 0.52));
    sq[4] = Square(vec3(10, 0, 0), vec3(20, 0, 0), vec3(20, 0, 10), vec3(10, 0, 10), 0, vec3(0.52, 0, 0.52));
    sq[5] = Square(vec3(10, 10, 0), vec3(20, 10, 0), vec3(20, 10, 10), vec3(10, 10, 10), 0, vec3(0.52, 0, 0.52));*/

    Material material = Material(vec3(0.9f, 0.2f, 0.1f), vec3(0.1f, 0.4f, 0.1f));


    Sphere sphere = Sphere(vec3(-0.5f, 0.0f, 0.0f), 0.7f, material);


    Ray ray = GenerateRay(camera, interpolated_vertex, scale);
    FragColor = RayTrace(ray, sphere);




    //float time = 1.0;
    //}
    //else
    //{
    //        FragColor = vec4(0.5, 0.5, 0.0, 0.5);
    //    }
/*if (IntersectSphere(sphere, ray, time) && time < intersect.time) {

    }*/

    //FragColor = vec4(0.5, 0.5, 0.0, 0.5);

    //FragColor = vec4(abs(ray.direction.xy), 0, 1.0);
}


