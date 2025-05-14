uniform float time;
uniform float noiseScale;
uniform float pulseSpeed;
uniform float pulseVariation;
uniform float patternScale;
uniform float displacementAmount; // Uniform for controlling base displacement strength

// New animated noise parameters
uniform float animNoiseScale; // Scale of the animated noise
uniform float animNoiseSpeed; // Speed of the animated noise
uniform float animNoiseStrength; // Strength multiplier for the animated noise
uniform float animNoiseOctaves; // Number of octaves for the animated noise

varying vec2 vUv;
varying vec3 vPosition;
varying vec3 vNormal;
varying vec3 vViewPosition;

// Simple 3D noise function (same as in fragment shader)
float mod289(float x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}
vec4 mod289(vec4 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}
vec4 perm(vec4 x) {
    return mod289(((x * 34.0) + 1.0) * x);
}

float noise(vec3 p) {
    vec3 a = floor(p);
    vec3 d = p - a;
    d = d * d * (3.0 - 2.0 * d);

    vec4 b = a.xxyy + vec4(0.0, 1.0, 0.0, 1.0);
    vec4 k1 = perm(b.xyxy);
    vec4 k2 = perm(k1.xyxy + b.zzww);

    vec4 c = k2 + a.zzzz;
    vec4 k3 = perm(c);
    vec4 k4 = perm(c + 1.0);

    vec4 o1 = fract(k3 * (1.0 / 41.0));
    vec4 o2 = fract(k4 * (1.0 / 41.0));

    vec4 o3 = o2 * d.z + o1 * (1.0 - d.z);
    vec2 o4 = o3.yw * d.x + o3.xz * (1.0 - d.x);

    return o4.y * d.y + o4.x * (1.0 - d.y);
}

float fbm(vec3 p, int octaves) {
    float sum = 0.0;
    float amp = 1.0;
    float freq = 1.0;
    // Variable number of octaves
    for(int i = 0; i < 8; i++) {
        if(i >= octaves)
            break;
        sum += amp * noise(p * freq);
        freq *= 2.0;
        amp *= 0.5;
    }
    return sum;
}

void main() {
    vUv = uv;

    // Create coherent noise pattern based on position and time for primary displacement
    vec3 noisePos = position * patternScale + vec3(0.0, 0.0, time * pulseSpeed);
    float displacementNoise = fbm(noisePos * noiseScale, 3);

    // Pulse variation
    float pulseFactor = sin(time * pulseSpeed) * 0.5 + 0.5;
    pulseFactor = pulseFactor * pulseVariation + (1.0 - pulseVariation);

    // Calculate animated noise for displacement variation
    // Use a different seed for this noise to ensure variation
    vec3 animNoisePos = position * patternScale + vec3(time * animNoiseSpeed * 0.5, time * animNoiseSpeed, 0.0);
    float animatedNoise = fbm(animNoisePos * animNoiseScale, int(animNoiseOctaves)) * animNoiseStrength;

    // Add animated noise to displacement amount
    float totalDisplacement = displacementAmount + animatedNoise;

    // Apply displacement along normal direction
    vec3 displacedPosition = position + normal * displacementNoise * totalDisplacement * pulseFactor;

    // Store the original position for use in fragment shader
    vPosition = position;
    vNormal = normalize(normalMatrix * normal);

    // Calculate view position for fresnel effect using the displaced position
    vec4 mvPosition = modelViewMatrix * vec4(displacedPosition, 1.0);
    vViewPosition = -mvPosition.xyz; // Negative because we need direction from eye to vertex

    gl_Position = projectionMatrix * mvPosition;
}