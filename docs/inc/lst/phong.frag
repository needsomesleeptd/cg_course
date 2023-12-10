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
    vec3 point = intersect.tracedRay.origin + intersect.tracedRay.direction * intersect.t;




    rayReflected = reflected;
    return vec4(rayColor, 1);
}