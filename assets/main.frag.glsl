precision highp float;

// Varyings from vertex shader
varying float vNoise;       // Normalized LF displacement factor [0,1]
varying vec3 vNormal;       // LF-perturbed normal (view space)
varying vec3 vViewPosition; // Vector from fragment to camera in view space (fragment's position in view is -vViewPosition)
varying vec3 vAdvectedCoord; // LF Advected object-like coord (already scaled by uLfMasterScale)
varying float vPatternTime;   // Time for LF pattern evolution
varying float vGlobalTime;    // Global uTime from vertex shader for HF evolution
varying vec2 vUv; // Texture coordinates from vertex shader
varying float vDisplacementAnimFactor;

// --- New Light Definition ---
#define MAX_LIGHTS 4 // Define a maximum number of lights

struct Light {
  vec3 color;
  float intensity;
  float diffuseContribution;
  float specularContribution;
  float specularShininess; // This will be a base shininess, can be modulated by texture
  vec3 position; // Position in view space
  float diffuseHfNormalAmount; // 0-1, blend factor for using HF normal in diffuse
  float specularHfNormalAmount; // 0-1, blend factor for using HF normal in specular
  bool enabled;
};
uniform Light uLights[MAX_LIGHTS];
uniform int uNumLights; // Actual number of active lights
uniform float uTime;

// Global Ambient Intensity
uniform float uAmbientIntensity;

// Color Uniforms
uniform vec3 uColor1;
uniform vec3 uColor2;
uniform vec3 uColor3;
uniform vec3 uTroughColorMultiplier;
uniform float uTroughMin;
uniform float uTroughMax;
uniform float uTroughGamma;
uniform float uColorNoiseScale;
uniform float uColorNoiseTimeScale;
uniform float uColorNoiseAnimationAmount;

// HF Noise Uniforms
uniform bool uHfEnable;
uniform float uHfMasterScale;
uniform float uHfPatternTimeScale;
uniform float uHfOctaves;
uniform float uHfLacunarity;
uniform float uHfPersistence;
uniform float uHfNormalStrength;
uniform float uHfFlowScale;
uniform float uHfFlowStrength;
uniform float uHfFlowTimeScale;

// Shininess Texture Uniform
uniform sampler2D uShininessMap; // << NEW UNIFORM

// --- Simple 1D Noise with 2 Octaves (Output: -1 to 1) ---
float random1(float n) {
  return fract(sin(n) * 43758.5453123);
}

// Helper function for 1D noise in [0,1] range using smoothstep
float noise1D_01(float x) {
  float i = floor(x);
  float f = fract(x);
  // Smoothstep interpolation (Hermite interpolation: 3f^2 - 2f^3)
  float u = f * f * (3.0 - 2.0 * f);
  return mix(random1(i), random1(i + 1.0), u);
}

float simple_noise1d_pos1(float x) {
  float total = 0.0;
  float frequency = 1.0;
  float amplitude = 1.0;
  float maxValue = 0.0; // Used for normalization

  // Octave 1
  total += noise1D_01(x * frequency) * amplitude;
  maxValue += amplitude;

  // Octave 2
  float lacunarity = 2.0;    // Frequency multiplier for the next octave
  float persistence = 0.5;   // Amplitude multiplier for the next octave

  frequency *= lacunarity;
  amplitude *= persistence;
  total += noise1D_01(x * frequency) * amplitude;
  maxValue += amplitude;

  // Normalize to [0, 1] based on the sum of amplitudes
  // This ensures the result stays within a predictable bound before scaling.
  if(maxValue == 0.0)
    return 0.0; // Avoid division by zero, though unlikely here

  return total / maxValue;
}
// --- End of Simple 1D Noise ---

// --- Simplex Noise Functions (random3_frag, simplex3d_frag) ---
vec3 random3_frag(vec3 c) {
  float j = 4096.0 * sin(dot(c, vec3(17.0, 59.4, 15.0)));
  vec3 r;
  r.z = fract(512.0 * j);
  j *= .125;
  r.x = fract(512.0 * j);
  j *= .125;
  r.y = fract(512.0 * j);
  return r - 0.5;
}
const float F3_FRAG = 0.3333333;
const float G3_FRAG = 0.1666667;
float simplex3d_frag(vec3 p) {
  vec3 s = floor(p + dot(p, vec3(F3_FRAG)));
  vec3 x = p - s + dot(s, vec3(G3_FRAG));
  vec3 e = step(vec3(0.0), x - x.yzx);
  vec3 i1 = e * (1.0 - e.zxy);
  vec3 i2 = 1.0 - e.zxy * (1.0 - e);
  vec3 x1 = x - i1 + G3_FRAG;
  vec3 x2 = x - i2 + 2.0 * G3_FRAG;
  vec3 x3 = x - 1.0 + 3.0 * G3_FRAG;
  vec4 w, d;
  w.x = dot(x, x);
  w.y = dot(x1, x1);
  w.z = dot(x2, x2);
  w.w = dot(x3, x3);
  w = max(0.6 - w, 0.0);
  d.x = dot(random3_frag(s), x);
  d.y = dot(random3_frag(s + i1), x1);
  d.z = dot(random3_frag(s + i2), x2);
  d.w = dot(random3_frag(s + 1.0), x3);
  w *= w;
  w *= w;
  d *= w;
  return dot(d, vec4(52.0));
}
// --- End of Simplex Noise ---

// --- HF Noise Specific Functions ---
vec3 get_hf_flow_vector(vec3 flow_coord_base, float flow_anim_time) {
  float flow_x = simplex3d_frag(vec3(flow_coord_base.x + 101.5, flow_coord_base.y + 63.2, flow_coord_base.z + 12.7 + flow_anim_time));
  float flow_y = simplex3d_frag(vec3(flow_coord_base.x - 47.8, flow_coord_base.y - 91.3, flow_coord_base.z - 39.5 + flow_anim_time));
  float flow_z = simplex3d_frag(vec3(flow_coord_base.x + 78.1, flow_coord_base.y - 123.4, flow_coord_base.z + 55.9 + flow_anim_time));
  return vec3(flow_x, flow_y, flow_z);
}

float fbm_simplex3d_hf_advected(
  vec3 p_advected_spatial_base,
  float pattern_time_val,
  float hf_octaves_float,
  float hf_lacunarity,
  float hf_persistence
) {
  float total = 0.0;
  float frequency = 1.0;
  float amplitude = 1.0;
  float maxValue = 0.0;
  int octaves = int(hf_octaves_float);
  for(int i = 0; i < octaves; i++) {
    if(i >= 8) // Max octaves safeguard
      break;
    vec3 current_p_spatial = p_advected_spatial_base * frequency;
    float current_pattern_time_component = pattern_time_val;
    float noise_sample = simplex3d_frag(vec3(current_p_spatial.xy, current_p_spatial.z + current_pattern_time_component));
    total += noise_sample * amplitude;
    maxValue += amplitude;
    amplitude *= hf_persistence;
    frequency *= hf_lacunarity;
  }
  if(maxValue == 0.0)
    return 0.0;
  return total / maxValue;
}
// --- End of HF Noise ---

void main() {
  // 1. Calculate Base Surface Color
  vec3 advectedCoordForColor = vAdvectedCoord * uColorNoiseScale;
  float colorNoiseInputTime = vPatternTime * uColorNoiseTimeScale;
  float colorNoiseValRaw = simplex3d_frag(vec3(advectedCoordForColor.xy, advectedCoordForColor.z + colorNoiseInputTime));
  float colorNoiseVal = colorNoiseValRaw * 0.5 + 0.5;
  vec3 baseSurfaceColor;
  float segment = colorNoiseVal * 2.0;
  if(segment < 1.0) {
    baseSurfaceColor = mix(uColor1, uColor2, segment);
  } else {
    baseSurfaceColor = mix(uColor2, uColor3, segment - 1.0);
  }
  float colorAnimationFactor = vDisplacementAnimFactor;
  float vNoiseMapped = pow(smoothstep(uTroughMin, uTroughMax, vNoise), mix(uTroughGamma, uTroughGamma * colorAnimationFactor, uColorNoiseAnimationAmount));
  vec3 troughEffectMultiplier = mix(uTroughColorMultiplier, vec3(1.0), vNoiseMapped);
  vec3 finalSurfaceColor = baseSurfaceColor * troughEffectMultiplier;

  // 2. Calculate Normals
  vec3 normal_lf_view = normalize(vNormal); // Base normal (LF perturbed) in view space
  vec3 normal_hf_view = normal_lf_view;     // Default to LF normal if HF is disabled or zero strength

  float vNoiseMappedInv = 1.0 - vNoiseMapped;
  float hf_noise_center;
  if(uHfEnable && uHfNormalStrength > 0.0 && uHfOctaves > 0.0) {
    vec3 hf_sample_coord_master = vAdvectedCoord * uHfMasterScale;
    vec3 hf_flow_sample_coord = hf_sample_coord_master * uHfFlowScale;
    float hf_flow_anim_time = vGlobalTime * uHfFlowTimeScale;
    vec3 hf_flow_vector = get_hf_flow_vector(hf_flow_sample_coord, hf_flow_anim_time);
    vec3 hf_advected_sample_coord = hf_sample_coord_master + hf_flow_vector * uHfFlowStrength;
    float hf_pattern_anim_time = vGlobalTime * uHfPatternTimeScale;

    float hf_eps = 0.01 / (uHfMasterScale > 0.0 ? uHfMasterScale : 1.0);
    if(uHfMasterScale == 0.0)
      hf_eps = 0.01;

    hf_noise_center = fbm_simplex3d_hf_advected(hf_advected_sample_coord, hf_pattern_anim_time, uHfOctaves, uHfLacunarity, uHfPersistence);
    float hf_noise_x_plus = fbm_simplex3d_hf_advected(hf_advected_sample_coord + vec3(hf_eps, 0.0, 0.0), hf_pattern_anim_time, uHfOctaves, uHfLacunarity, uHfPersistence);
    float hf_noise_y_plus = fbm_simplex3d_hf_advected(hf_advected_sample_coord + vec3(0.0, hf_eps, 0.0), hf_pattern_anim_time, uHfOctaves, uHfLacunarity, uHfPersistence);

    vec3 view_tangent = normalize(cross(normal_lf_view, vec3(0.0, 1.0, 0.01)));
    if(length(view_tangent) < 0.001) {
      view_tangent = normalize(cross(normal_lf_view, vec3(1.0, 0.0, 0.0)));
    }
    vec3 view_bitangent = normalize(cross(normal_lf_view, view_tangent));
    mat3 tbn = mat3(view_tangent, view_bitangent, normal_lf_view);

    float hf_noise_z_plus = fbm_simplex3d_hf_advected(hf_advected_sample_coord + vec3(0.0, 0.0, hf_eps), hf_pattern_anim_time, uHfOctaves, uHfLacunarity, uHfPersistence);
    vec3 hf_fbm_gradient_object_space_approx = vec3((hf_noise_x_plus - hf_noise_center) / hf_eps, (hf_noise_y_plus - hf_noise_center) / hf_eps, (hf_noise_z_plus - hf_noise_center) / hf_eps);

    vec3 normal_perturbation_tangent_space = normalize(vec3(hf_fbm_gradient_object_space_approx.x * uHfNormalStrength, hf_fbm_gradient_object_space_approx.y * uHfNormalStrength, 1.0));
    normal_hf_view = normalize(tbn * normal_perturbation_tangent_space);
  }

  // 3. Apply Lighting
  vec3 V = normalize(vViewPosition);
  vec3 fragmentPosView = -vViewPosition;

  vec3 totalDiffuse = vec3(0.0);
  vec3 totalSpecular = vec3(0.0);

  // Get shininess value from texture
  float shininessFromMap = texture2D(uShininessMap, vAdvectedCoord.xy).r; // Assuming shininess is in the red channel

  for(int i = 0; i < MAX_LIGHTS; ++i) {
    if(i >= uNumLights || !uLights[i].enabled)
      continue;

    Light currentLight = uLights[i];
    vec3 L = normalize(currentLight.position - fragmentPosView);

    vec3 N_diffuse = mix(normal_lf_view, normal_hf_view, currentLight.diffuseHfNormalAmount);
    N_diffuse = normalize(N_diffuse);

    float NdotL_diffuse = max(dot(N_diffuse, L), 0.0);
    totalDiffuse += currentLight.color * NdotL_diffuse * currentLight.diffuseContribution * currentLight.intensity;

    vec3 N_specular = mix(normal_lf_view, normal_hf_view, currentLight.specularHfNormalAmount);
    N_specular = normalize(N_specular);

    vec3 H = normalize(L + V);
    float NdotH_specular = max(dot(N_specular, H), 0.0);

    // Modulate specularShininess with texture value. Adjust multiplier as needed (e.g. 256.0)
    float finalShininess = currentLight.specularShininess * shininessFromMap; // << MODIFIED
    if(finalShininess < 1.0)
      finalShininess = 1.0; // Ensure shininess is at least 1

    float specularFactor = pow(NdotH_specular, finalShininess); // << MODIFIED
    totalSpecular += currentLight.color * specularFactor * currentLight.specularContribution * currentLight.intensity * (finalShininess / 255.0);
  }

  vec3 ambientColor = uAmbientIntensity * finalSurfaceColor;
  vec3 litColor = ambientColor + (totalDiffuse * finalSurfaceColor) + totalSpecular;
  litColor = pow(litColor, vec3(1.0 / 2.2));

  gl_FragColor = vec4(litColor, 1.0);
}