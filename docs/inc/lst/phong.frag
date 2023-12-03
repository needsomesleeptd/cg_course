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


    vec3 idealReflection = normalize(reflect(intersect.tracedRay.direction, intersect.normal));

    vec3 reflectedDirection = idealReflection;
    vec3 newRayOrigin = intersect.tracedRay.origin + intersect.t * intersect.tracedRay.direction;
    newRayOrigin += intersect.normal * 0.02;
    Ray reflected = Ray(newRayOrigin, reflectedDirection);
    float specularDot = pow(max(dot(-reflected.direction, intersect.tracedRay.direction), 0.0f), 5);

    rayColor += shapeMaterial.lightKoefs[2] * lightSource.intensivity * specularDot;

    rayReflected = reflected;
    return vec4(rayColor, 1);
}