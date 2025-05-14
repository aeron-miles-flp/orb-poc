uniform float time;
uniform float intensity;
uniform float noiseScale;
uniform float pulseSpeed;
uniform float pulseVariation;
uniform float patternScale;
uniform vec3 emberColor;
uniform vec3 glowColor;
uniform float fresnelPower;
uniform float fresnelIntensity;
uniform float fresnelBias;

// Layer 2 uniforms
uniform bool layer2Enabled;
uniform float layer2Intensity;
uniform float layer2NoiseScale;
uniform float layer2PulseSpeed;
uniform float layer2PulseVariation;
uniform float layer2PatternScale;
uniform vec3 layer2EmberColor;
uniform vec3 layer2GlowColor;
uniform float layer2BlendFactor;

varying vec2 vUv;
varying vec3 vPosition;
varying vec3 vNormal;
varying vec3 vViewPosition;

// Simple 3D noise function
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

float fbm(vec3 p) {
    float sum = 0.0;
    float amp = 1.0;
    float freq = 1.0;
    // 4 octaves of noise
    for(int i = 0; i < 3; i++) {
        sum += amp * noise(p * freq);
        freq *= 2.0;
        amp *= 0.5;
    }
    return sum;
}

// Calculate the Fresnel effect
float fresnel(vec3 viewDir, vec3 normal, float power, float bias) {
    float NdotV = max(dot(normalize(normal), normalize(viewDir)), 0.0);
    // Invert to make edges brighter and front faces dimmer
    return bias + (1.0 - bias) * pow(1.0 - NdotV, power);
}

// Function to calculate effect for a single layer with given parameters
vec4 calculateEmberEffect(
    vec3 position,
    float currentTime,
    float layerIntensity,
    float layerNoiseScale,
    float layerPulseSpeed,
    float layerPulseVar,
    float layerPatternScale,
    vec3 layerEmberColor,
    vec3 layerGlowColor
) {
    // Create coherent noise pattern based on position and time
    vec3 noisePos = position * layerPatternScale + vec3(0.0, 0.0, currentTime * layerPulseSpeed);
    float baseNoise = fbm(noisePos * layerNoiseScale);

    // Pulse variation
    float pulseFactor = sin(currentTime * layerPulseSpeed) * 0.5 + 0.5;
    pulseFactor = pulseFactor * layerPulseVar + (1.0 - layerPulseVar);

    // Rim lighting effect (brighten edges)
    float rimLight = 1.0 - max(dot(vNormal, vec3(0.0, 0.0, 1.0)), 0.0);
    rimLight = pow(rimLight, 3.0);

    // Calculate ember glow
    float glow = baseNoise * pulseFactor * layerIntensity;
    // glow = glow * (1.0 + rimLight * 2.0); // Enhance at edges

    // Make the pattern more "fiery" with threshold
    float threshold = 0.55;
    float emberIntensity = smoothstep(threshold, threshold + 0.2, glow);

    // Calculate Fresnel effect - this will be stronger at glancing angles
    float fresnelFactor = fresnel(vViewPosition, vNormal, fresnelPower, fresnelBias);

    // Apply inverse Fresnel to fade out edges and keep front-facing geometry
    float freshelMask = 1.0 - (fresnelFactor * fresnelIntensity);

    // Color interpolation based on intensity
    vec3 finalColor = mix(layerGlowColor * 0.8, layerEmberColor, emberIntensity);

    // Add emissive glow with Fresnel effect applied
    float emission = emberIntensity * layerIntensity * freshelMask;

    return vec4(finalColor, emission);
}

void main() {
    // Calculate base layer effect
    vec4 layer1 = calculateEmberEffect(vPosition, time, intensity, noiseScale, pulseSpeed, pulseVariation, patternScale, emberColor, glowColor);

    // Initialize final color with layer 1
    vec4 finalEffect = layer1;

    // If second layer is enabled, blend it with first layer
    if(layer2Enabled) {
        // Calculate second layer with its own parameters
        vec4 layer2 = calculateEmberEffect(vPosition, time * 1.2, layer2Intensity, layer2NoiseScale, layer2PulseSpeed, layer2PulseVariation, layer2PatternScale, layer2EmberColor, layer2GlowColor);

        // Use layer1's alpha as a mask for layer2 and blend additively
        float maskFactor = layer1.a * layer2BlendFactor;

        // Blend the colors additively, with layer2 masked by layer1's intensity
        finalEffect.rgb = layer1.rgb + (layer2.rgb * maskFactor);

        // Adjust the alpha (emission) value for the combined effect
        finalEffect.a = max(layer1.a, layer2.a * maskFactor);
    }

    // Output the final composited color
    gl_FragColor = finalEffect;
}