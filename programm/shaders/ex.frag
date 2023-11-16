#version 420

#define EPS 1e-3
#define INF 1e6

struct Camera {
    vec3 position;
    vec3 view;
    vec3 up;
    vec3 right;
    mat4 inverseProjectionMatrix;
};

struct Ray {
    vec3 origin;
    vec3 direction;
};

struct Material {
    vec3 _color;
    float k_a;
    float k_d;
    float k_s;
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
    float time;
    vec3 point;
    vec3 normal;
    vec3 color;
    vec4 light_coeffs;
    Material material;
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


in vec4 interpolated_vertex;
out vec4 FragColor;
/*
uniform Light light;
uniform Camera camera;
uniform vec2 scale;
uniform vec3 light_pos;
//uniform int vector_size;*/

/*layout(std430, binding = 0) buffer SphereBuffer {
    vec3 sphere_data[];
};*/

/*layout(std430,binding = 1) buffer PlaneBuffer{
    Plane plane_data[];
};*/

Material material = Material(vec3(0.3, 0.4, 0.4), 0.4, 0.3, 0.3);


Sphere sphere = Sphere(vec3(-10,0,0),0.5,material);

// vec3 light_pos = vec3(10, -5, -5);
/*
Ray GenerateRay(Camera camera) {
    vec2 coords = interpolated_vertex.xy * normalize(scale);
    vec3 direction = camera.view + camera.right * coords.x + camera.up * coords.y;
    return Ray(camera.position, normalize(direction));
}

bool IntersectSphere(Sphere sphere, Ray ray, out float time) {
    ray.origin -= sphere.center;
    float A = dot(ray.direction, ray.direction);
    float B = dot(ray.direction, ray.origin);
    float C = dot(ray.origin, ray.origin) - sphere.radius * sphere.radius;

    float D = B * B - A * C;
    if (D > 0) {
        D = sqrt(D);
        float t1 = (-B - D) / (A);
        float t2 = (-B + D) / (A);
        float mi = min(t1, t2);
        float ma = max(t1, t2);

        if (ma < 0) return false;
        if (mi < 0) {
            time = ma;
            return true;
        }
        time = mi;
        return true;
    }
    return false;
}

bool IntersectTriangle(Ray ray, vec3 v1, vec3 v2, vec3 v3, out float time) {
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
}

bool IntersectSquare(Ray ray, vec3 v1, vec3 v2, vec3 v3, vec3 v4, out float time) {
    time = -1;
    vec3 A = v2 - v1;
    vec3 B = v4 - v1;
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
    vec3 edge3 = v4 - v3;
    vec3 VP3 = P - v3;
    C = cross(edge3, VP3);
    if (dot(N, C) < 0)   return false;
    vec3 edge4 = v1 - v4;
    vec3 VP4 = P - v4;
    C = cross(edge4, VP4);
    if (dot(N, C) < 0)   return false;
    time = t;
    return true;
}

bool Intersect(Ray ray, float start, float final, inout Intersection intersect) {
    bool result = false;
    float time = start;
    intersect.time = final;

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
    //return result;
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
}

vec3 Phong(Intersection intersect, vec3 pos_light, float shadow) {
    vec3 light = normalize(pos_light - intersect.point);
    float diffuse = max(dot(light, intersect.normal), 0.0);
    vec3 view = normalize(camera.position - intersect.point);
    vec3 reflected = reflect(-view, intersect.normal);
    float specular = pow(max(dot(reflected, light), 0.0), intersect.light_coeffs.w);

    return intersect.light_coeffs.x * intersect.color +
    intersect.light_coeffs.y * diffuse * intersect.color * shadow +
    intersect.light_coeffs.z * specular;
}

vec4 RayTrace(Ray primary_ray) {
    vec4 resColor = vec4(0, 0, 0, 0);
    Ray ray = primary_ray;

    Intersection intersect;
    intersect.time = INF;
    float start = 0;
    float final = INF;
    if (IntersectSphere(sphere,ray, final))
    {
        //intersect.time = time;
        intersect.point = ray.origin + ray.direction * intersect.time;
        intersect.normal = normalize(intersect.point - sphere.center);
        intersect.color = sphere.material._color;
        intersect.material = sphere.material;
        intersect.light_coeffs = vec4(material.k_a,material.k_d,material.k_s,0.0);
        float shadowing = Shadow(light_pos, intersect);
        resColor += vec4(Phong(intersect, light_pos, shadowing), 0);
    }
    /*if (Intersect(ray, start, final, intersect)) {
        float shadowing = Shadow(light_pos, intersect);
        resColor += vec4(Phong(intersect, light_pos, shadowing), 0);
    }*/
    //return resColor;
//}


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


    //Ray ray = GenerateRay(camera);
    //FragColor = RayTrace(ray);

    /*if (IntersectSphere(sphere, ray, time) && time < intersect.time) {

    }*/
//    gl_FragColor = vec4(0.5,0.5,0.0,0.5);
    //gl_Position = interpolated_vertex;
    FragColor = interpolated_vertex;

    //FragColor = vec4(abs(ray.direction.xy), 0, 1.0);
}