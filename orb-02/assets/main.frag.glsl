precision highp float;

float EPSILON = 0.01;
// =============================================================================
// VARYINGS (OUTPUT TO FRAGMENT SHADER)
// =============================================================================
varying float vNoise;                   // Normalized displacement factor [0,1]
varying vec3 vNormal;                   // Final combined normal (view space)
varying vec3 vObjectNormal;             // Object space normal [-1,1] range for xyz
varying vec3 vTangent;                  // Tangent vector (view space)
varying vec3 vBitangent;                // Bitangent vector (view space)
varying vec3 vViewPosition;             // Fragment position relative to camera
varying vec3 vWorldPosition;            // World space position
varying vec3 vAdvectedCoord;            // Flow-advected coordinate for noise sampling
varying vec2 vUvAdvectedCoord;          // Flow-advected UV coordinate
varying float vPatternTime;             // Time component for pattern evolution
varying float vGlobalTime;              // Global time for fragment shader
varying vec2 vUv;                       // Base UV coordinates
varying float vDisplacementAnimFactor;  // Animation factor for displacement

// =============================================================================
// LIGHTING SYSTEM
// =============================================================================
#define MAX_LIGHTS 4

struct Light {
  vec3 color;
  float intensity;
  float diffuseContribution;
  float specularContribution;
  float sssContribution;
  float specularShininess;         // Base shininess (modulated by texture)
  vec3 position;                   // Position in view space
  float diffuseHfNormalAmount;     // Blend factor for HF normal in diffuse [0-1]
  float specularHfNormalAmount;    // Blend factor for HF normal in specular [0-1]
  bool enabled;
};

uniform Light uLights[MAX_LIGHTS];
uniform int uNumLights;
uniform float uAmbientIntensity;

// =============================================================================
// SUBSURFACE SCATTERING UNIFORMS
// =============================================================================
uniform bool uSSSEnable;                    // Enable/disable SSS
uniform float uSSSIntensity;                 // Overall SSS intensity
uniform vec3 uSSSColor;                      // SSS tint color (often warm/red)
uniform float uSSSSurfaceColorContribution;
uniform float uSSSThicknessPower;            // Power curve for thickness attenuation
uniform float uSSSDistortion;                // Normal distortion for transmission
uniform float uSSSAmbient;                   // Ambient SSS contribution
uniform float uSSSAttenuation;               // Distance attenuation factor

// =============================================================================
// SURFACE COLOR UNIFORMS
// =============================================================================
uniform vec3 uColor1;
uniform vec3 uColor2;
uniform vec3 uColor3;
uniform vec3 uTroughColorMultiplier;
uniform float uTroughMin;
uniform float uTroughMax;
uniform float uTroughGamma;
uniform float uColorNoiseScale;
uniform float uColorNoiseGamma;
uniform float uColorNoiseTimeScale;
uniform float uColorNoiseAnimationAmount;
uniform float uColorSaturation;

// =============================================================================
// HIGH FREQUENCY NOISE UNIFORMS
// =============================================================================
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

// =============================================================================
// DISPLACEMENT MAP UNIFORMS
// =============================================================================
uniform bool uDispMapEnable;
uniform float uDispMapNormalStrength;
uniform float uDispMapScale;
uniform vec2 uDispMapOffset;

// =============================================================================
// TEXTURE UNIFORMS
// =============================================================================
uniform sampler2D uShininessMap;
uniform sampler2D uCTAOMap;
uniform sampler2D uNormalMap;
uniform sampler2D uNormalMapLF;
uniform sampler2D uDispMap;

// =============================================================================
// TIME UNIFORM
// =============================================================================
uniform float uTime;

// =============================================================================
// VIEW-OBJECT SPACE TRANSFORMATION METHODS
// =============================================================================

// Note: These methods require the following uniforms to be defined:
uniform mat3 uNormalMatrix;        // For transforming normals/directions from object to view
uniform mat3 uInverseNormalMatrix; // For transforming normals/directions from view to object
uniform mat4 uModelViewMatrix;     // For transforming positions from object to view  
uniform mat4 uInverseModelViewMatrix; // For transforming positions from view to object
uniform mat3 uViewMatrix;          // For transforming normals/directions from world to view
uniform mat3 uInverseViewMatrix;   // For transforming normals/directions from view to world
uniform mat4 uViewMatrixFull;      // For transforming positions from world to view
uniform mat4 uInverseViewMatrixFull; // For transforming positions from view to world

// Transform a vector from view space to world space
vec3 view_to_world(vec3 value) {
  // Use the inverse view matrix to transform directions/normals from view to world space
  return normalize(uInverseViewMatrix * value);
}

// Transform a vector from world space to view space  
vec3 world_to_view(vec3 value) {
  // Use the view matrix to transform directions/normals from world to view space
  return normalize(uViewMatrix * value);
}

// Alternative versions for positions (if needed)
vec3 view_to_world_position(vec3 viewPosition) {
  // Transform a position from view space to world space
  vec4 worldPos = uInverseViewMatrixFull * vec4(viewPosition, 1.0);
  return worldPos.xyz / worldPos.w;
}

vec3 world_to_view_position(vec3 worldPosition) {
  // Transform a position from world space to view space
  vec4 viewPos = uViewMatrixFull * vec4(worldPosition, 1.0);
  return viewPos.xyz / viewPos.w;
}

// Transform a vector from view space to object space
vec3 view_to_object(vec3 value) {
  // Use the inverse normal matrix to transform directions/normals from view to object space
  return normalize(uInverseNormalMatrix * value);
}

// Transform a vector from object space to view space  
vec3 object_to_view(vec3 value) {
  // Use the normal matrix to transform directions/normals from object to view space
  return normalize(uNormalMatrix * value);
}

// Alternative versions for positions (if needed)
vec3 view_to_object_position(vec3 viewPosition) {
  // Transform a position from view space to object space
  vec4 objectPos = uInverseModelViewMatrix * vec4(viewPosition, 1.0);
  return objectPos.xyz / objectPos.w;
}

vec3 object_to_view_position(vec3 objectPosition) {
  // Transform a position from object space to view space
  vec4 viewPos = uModelViewMatrix * vec4(objectPosition, 1.0);
  return viewPos.xyz / viewPos.w;
}

// =============================================================================
// NOISE FUNCTIONS
// =============================================================================

// Simple random function for 1D noise
float random1(float n) {
  return fract(sin(n) * 43758.5453123);
}

// 1D noise in [0,1] range using smoothstep interpolation
float noise1D_01(float x) {
  float i = floor(x);
  float f = fract(x);
  float u = f * f * (3.0 - 2.0 * f); // Hermite interpolation
  return mix(random1(i), random1(i + 1.0), u);
}

// Simple 2-octave 1D noise in [0,1] range
float simple_noise1d_pos1(float x) {
  float total = 0.0;
  float frequency = 1.0;
  float amplitude = 1.0;
  float maxValue = 0.0;

  // Octave 1
  total += noise1D_01(x * frequency) * amplitude;
  maxValue += amplitude;

  // Octave 2
  float lacunarity = 2.0;
  float persistence = 0.5;
  frequency *= lacunarity;
  amplitude *= persistence;
  total += noise1D_01(x * frequency) * amplitude;
  maxValue += amplitude;

  return (maxValue == 0.0) ? 0.0 : total / maxValue;
}

// 3D random vector for simplex noise
vec3 random3_frag(vec3 c) {
  float j = 4096.0 * sin(dot(c, vec3(17.0, 59.4, 15.0)));
  vec3 r;
  r.z = fract(512.0 * j);
  j *= 0.125;
  r.x = fract(512.0 * j);
  j *= 0.125;
  r.y = fract(512.0 * j);
  return r - 0.5;
}

// 3D Simplex noise implementation
const float F3_FRAG = 0.3333333;
const float G3_FRAG = 0.1666667;

float simplex3d_frag(vec3 p) {
  // Skew input space to determine which simplex cell we're in
  vec3 s = floor(p + dot(p, vec3(F3_FRAG)));
  vec3 x = p - s + dot(s, vec3(G3_FRAG));

  // Calculate the other three corners of the tetrahedron
  vec3 e = step(vec3(0.0), x - x.yzx);
  vec3 i1 = e * (1.0 - e.zxy);
  vec3 i2 = 1.0 - e.zxy * (1.0 - e);

  vec3 x1 = x - i1 + G3_FRAG;
  vec3 x2 = x - i2 + 2.0 * G3_FRAG;
  vec3 x3 = x - 1.0 + 3.0 * G3_FRAG;

  // Calculate contributions from each corner
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

// =============================================================================
// HIGH FREQUENCY NOISE FUNCTIONS
// =============================================================================

// Generate flow vector for HF noise advection
vec3 get_hf_flow_vector(vec3 flow_coord_base, float flow_anim_time) {
  float flow_x = simplex3d_frag(vec3(flow_coord_base.x + 101.5, flow_coord_base.y + 63.2, flow_coord_base.z + 12.7 + flow_anim_time));

  float flow_y = simplex3d_frag(vec3(flow_coord_base.x - 47.8, flow_coord_base.y - 91.3, flow_coord_base.z - 39.5 + flow_anim_time));

  float flow_z = simplex3d_frag(vec3(flow_coord_base.x + 78.1, flow_coord_base.y - 123.4, flow_coord_base.z + 55.9 + flow_anim_time));

  return vec3(flow_x, flow_y, flow_z);
}

// Fractal Brownian Motion using simplex noise with advection
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
    if(i >= 8)
      break; // Max octaves safeguard

    vec3 current_p_spatial = p_advected_spatial_base * frequency;
    float current_pattern_time_component = pattern_time_val;
    float noise_sample = simplex3d_frag(vec3(current_p_spatial.xy, current_p_spatial.z + current_pattern_time_component));

    total += noise_sample * amplitude;
    maxValue += amplitude;
    amplitude *= hf_persistence;
    frequency *= hf_lacunarity;
  }

  return (maxValue == 0.0) ? 0.0 : total / maxValue;
}

// =============================================================================
// NORMAL CALCULATION WITH PROPER ADVECTION CORRECTION
// =============================================================================

// Calculate the Jacobian matrix of the UV advection transformation
mat2 calculateUVJacobian(vec2 baseUV) {
  // We need to calculate how the advected UV changes with respect to base UV
  // This requires sampling the flow at neighboring points

  float eps = 0.001; // Small epsilon for finite differences

  // Get current UV distortion
  vec2 currentAdvectedUV = vUvAdvectedCoord;
  vec2 currentDistortion = currentAdvectedUV - vUv;

  // Calculate derivatives using screen-space derivatives
  // These give us how the UV coordinates change across the surface
  vec2 dUVdx = dFdx(vUv);
  vec2 dUVdy = dFdy(vUv);

  // Calculate how the advected UV changes
  vec2 dAdvectedUVdx = dFdx(vUvAdvectedCoord);
  vec2 dAdvectedUVdy = dFdy(vUvAdvectedCoord);

  // The Jacobian describes how a small change in base UV affects advected UV
  // J = d(advectedUV)/d(baseUV)
  mat2 jacobian;

  // Use the chain rule: d(advectedUV)/d(screenSpace) * d(screenSpace)/d(baseUV)
  if(length(dUVdx) > 0.0001 && length(dUVdy) > 0.0001) {
    // Construct the Jacobian from screen-space derivatives
    vec2 dDistortiondx = dAdvectedUVdx - dUVdx;
    vec2 dDistortiondy = dAdvectedUVdy - dUVdy;

    // The Jacobian of the advection is I + gradient of distortion
    jacobian = mat2(1.0 + dDistortiondx.x / dUVdx.x, dDistortiondy.x / dUVdy.y, dDistortiondx.y / dUVdx.x, 1.0 + dDistortiondy.y / dUVdy.y);
  } else {
    // Fallback to identity if derivatives are too small
    jacobian = mat2(1.0);
  }

  return jacobian;
}

// Transform normal from tangent space to view space with advection correction
vec3 transformNormalWithAdvectionCorrection(vec3 normalMapSample) {
  // Decode normal from normal map
  vec3 normalTangent = normalMapSample * 2.0 - 1.0;

  // Get the base tangent space vectors (already in view space from vertex shader)
  vec3 N = normalize(vNormal);
  vec3 T = normalize(vTangent);
  vec3 B = normalize(vBitangent);

  // Ensure orthonormal basis
  T = normalize(T - dot(T, N) * N);
  B = normalize(cross(N, T));

  // Calculate the UV Jacobian
  mat2 uvJacobian = calculateUVJacobian(vUv);

  // The inverse transpose of the Jacobian transforms normals correctly
  mat2 uvJacobianInvT = mat2(uvJacobian[1][1], -uvJacobian[0][1], -uvJacobian[1][0], uvJacobian[0][0]) / (uvJacobian[0][0] * uvJacobian[1][1] - uvJacobian[0][1] * uvJacobian[1][0]);

  // Apply the Jacobian correction to the tangent-space normal's XY components
  vec2 correctedNormalXY = uvJacobianInvT * normalTangent.xy;

  // Reconstruct the corrected tangent-space normal
  vec3 correctedNormalTangent = vec3(correctedNormalXY, normalTangent.z);
  correctedNormalTangent = normalize(correctedNormalTangent);

  // Transform to view space
  mat3 TBN = mat3(T, B, N);
  return normalize(TBN * correctedNormalTangent);
}

// =============================================================================
// SIMPLE ADVECTION TRANSFORMATION METHODS
// =============================================================================

// Transform a vec3 value by advection displacement
vec3 transformByAdvection(vec3 value, vec2 uvAdvected) {
  // Calculate UV displacement
  vec2 uvDisplacement = uvAdvected - vUv;

  // If displacement is negligible, return original value
  if(length(uvDisplacement) < 0.0001) {
    return value;
  }

  // Use screen-space derivatives to estimate spatial gradients
  vec3 gradientU = dFdx(value) / max(abs(dFdx(vUv.x)), 0.0001);
  vec3 gradientV = dFdy(value) / max(abs(dFdy(vUv.y)), 0.0001);

  // Apply displacement using linear approximation
  vec3 transformedValue = value + gradientU * uvDisplacement.x + gradientV * uvDisplacement.y;

  return transformedValue;
}

// Transform a float value by advection displacement  
float transformByAdvection(float value, vec2 uvAdvected) {
  // Calculate UV displacement
  vec2 uvDisplacement = uvAdvected - vUv;

  // If displacement is negligible, return original value
  if(length(uvDisplacement) < 0.0001) {
    return value;
  }

  // Use screen-space derivatives to estimate spatial gradients
  float gradientU = dFdx(value) / max(abs(dFdx(vUv.x)), 0.0001);
  float gradientV = dFdy(value) / max(abs(dFdy(vUv.y)), 0.0001);

  // Apply displacement using linear approximation
  float transformedValue = value + gradientU * uvDisplacement.x + gradientV * uvDisplacement.y;

  return transformedValue;
}

// =============================================================================
// HIGH FREQUENCY NORMAL CALCULATION
// =============================================================================
vec3 calculateHighFrequencyNormal(vec3 normal_lf_view) {
  if(!uHfEnable || uHfNormalStrength <= 0.0 || uHfOctaves <= 0.0) {
    return normal_lf_view;
  }

  // Setup HF sampling coordinates
  vec3 hf_sample_coord_master = vAdvectedCoord * uHfMasterScale;
  vec3 hf_flow_sample_coord = hf_sample_coord_master * uHfFlowScale;
  float hf_flow_anim_time = vGlobalTime * uHfFlowTimeScale;
  vec3 hf_flow_vector = get_hf_flow_vector(hf_flow_sample_coord, hf_flow_anim_time);
  vec3 hf_advected_sample_coord = hf_sample_coord_master + hf_flow_vector * uHfFlowStrength;
  float hf_pattern_anim_time = vGlobalTime * uHfPatternTimeScale;

  // Calculate epsilon for gradient computation
  float hf_eps = (uHfMasterScale > 0.0) ? 0.01 / uHfMasterScale : 0.01;

  // Sample noise at center and offset positions for gradient calculation
  float hf_noise_center = fbm_simplex3d_hf_advected(hf_advected_sample_coord, hf_pattern_anim_time, uHfOctaves, uHfLacunarity, uHfPersistence);

  float hf_noise_x_plus = fbm_simplex3d_hf_advected(hf_advected_sample_coord + vec3(hf_eps, 0.0, 0.0), hf_pattern_anim_time, uHfOctaves, uHfLacunarity, uHfPersistence);

  float hf_noise_y_plus = fbm_simplex3d_hf_advected(hf_advected_sample_coord + vec3(0.0, hf_eps, 0.0), hf_pattern_anim_time, uHfOctaves, uHfLacunarity, uHfPersistence);

  float hf_noise_z_plus = fbm_simplex3d_hf_advected(hf_advected_sample_coord + vec3(0.0, 0.0, hf_eps), hf_pattern_anim_time, uHfOctaves, uHfLacunarity, uHfPersistence);

  // Calculate gradient
  vec3 hf_fbm_gradient = vec3((hf_noise_x_plus - hf_noise_center) / hf_eps, (hf_noise_y_plus - hf_noise_center) / hf_eps, (hf_noise_z_plus - hf_noise_center) / hf_eps);

  // Create tangent space matrix
  vec3 view_tangent = normalize(cross(normal_lf_view, vec3(0.0, 1.0, 0.01)));
  if(length(view_tangent) < 0.001) {
    view_tangent = normalize(cross(normal_lf_view, vec3(1.0, 0.0, 0.0)));
  }
  vec3 view_bitangent = normalize(cross(normal_lf_view, view_tangent));
  mat3 tbn = mat3(view_tangent, view_bitangent, normal_lf_view);

  // Transform gradient to tangent space and apply strength
  vec3 normal_perturbation_tangent_space = normalize(vec3(hf_fbm_gradient.x * uHfNormalStrength, hf_fbm_gradient.y * uHfNormalStrength, 1.0));

  return normalize(tbn * normal_perturbation_tangent_space);
}

vec3 saturation(vec3 inputXYZ, float saturation) {
    // Calculate luminance using standard weights for RGB
    // These weights correspond to human eye sensitivity
  vec3 luminanceWeights = vec3(0.299, 0.587, 0.114);
  float luminance = dot(inputXYZ, luminanceWeights);

    // Create grayscale version (desaturated)
  vec3 grayscale = vec3(luminance);

    // Mix between grayscale and original color based on saturation factor
    // saturation.x controls overall saturation
    // For per-channel control, you could use each component separately
  return mix(grayscale, inputXYZ, saturation);
}

// =============================================================================
// SURFACE COLOR CALCULATION
// =============================================================================
vec3 calculateSurfaceColor() {
  // Generate color noise
  vec3 advectedCoordForColor = vAdvectedCoord * uColorNoiseScale;
  float colorNoiseInputTime = vPatternTime * uColorNoiseTimeScale;
  float colorNoiseValRaw_A = simplex3d_frag(vec3(advectedCoordForColor.xy, advectedCoordForColor.z + colorNoiseInputTime));
  colorNoiseValRaw_A = (colorNoiseValRaw_A * 0.5) + 0.5;
  float colorNoiseValRaw_B = simplex3d_frag(vec3(advectedCoordForColor.xy + vec2(0.333, 0.555), advectedCoordForColor.z - colorNoiseInputTime));
  colorNoiseValRaw_B = (colorNoiseValRaw_B * 0.5) + 0.5;

  // Blend between three colors based on noise
  vec3 baseSurfaceColor = uColor1;
  baseSurfaceColor = mix(baseSurfaceColor, uColor2, smoothstep(1.0 - uColorNoiseGamma, uColorNoiseGamma, colorNoiseValRaw_A));
  baseSurfaceColor = mix(baseSurfaceColor, uColor3, smoothstep(1.0 - uColorNoiseGamma, uColorNoiseGamma, colorNoiseValRaw_B));

  // Apply trough effect based on displacement
  float colorAnimationFactor = vDisplacementAnimFactor;
  float vNoiseMapped = pow(smoothstep(uTroughMin, uTroughMax, vNoise), mix(uTroughGamma, uTroughGamma * colorAnimationFactor, uColorNoiseAnimationAmount));

  float smoothEffect = smoothstep(0.0, 0.2, abs(uTroughMax - uTroughMin));
  vec3 troughEffectMultiplier = mix(uTroughColorMultiplier, vec3(smoothEffect), vNoiseMapped);

  return baseSurfaceColor * troughEffectMultiplier;
}

// =============================================================================
// SUBSURFACE SCATTERING CALCULATION
// =============================================================================
vec3 calculateSubsurfaceScattering(vec3 surfaceColor, vec3 lightColor, vec3 lightDir, vec3 normal, vec3 viewDir, float thickness, float lightIntensity, float sssContribution) {
  if(!uSSSEnable || uSSSIntensity <= 0.0 || sssContribution <= 0.0) {
    return vec3(0.0);
  }

  // Create a distorted normal for transmission calculation
  // This simulates light scattering through the surface
  vec3 distortedNormal = normalize(normal + vec3(0.0, 0.0, uSSSDistortion));

  // Calculate transmission - how much light passes through from behind
  // We use the inverse normal to simulate back-lighting
  float transmissionDot = dot(-distortedNormal, lightDir);
  transmissionDot = max(0.0, transmissionDot);

  // Create a wrap-around lighting effect for softer transmission
  float wrapAroundFactor = 0.5;
  float transmission = pow(max(0.0, transmissionDot + wrapAroundFactor), 2.0);

  // Thickness attenuation - thicker areas transmit less light
  // Use power curve to control the thickness falloff
  float thicknessAttenuation = pow(max(0.0, 1.0 - thickness), uSSSThicknessPower);

  // View-dependent scattering - more pronounced at grazing angles
  float fresnel = 1.0 - max(0.0, dot(normal, viewDir));
  fresnel = pow(fresnel, 2.0);

  // Distance attenuation (optional, for point lights)
  float attenuation = 1.0 / (1.0 + uSSSAttenuation * 0.1);

  // Combine all factors
  float sssAmount = transmission * thicknessAttenuation * (1.0 + fresnel * 0.5) * attenuation;

  // Apply SSS color tint and intensity
  vec3 sssColor = mix(vec3(1.0), surfaceColor, uSSSSurfaceColorContribution) * lightColor * uSSSColor * sssAmount * uSSSIntensity * lightIntensity * sssContribution;

  return sssColor;
}

// Calculate ambient subsurface scattering
vec3 calculateAmbientSSS(vec3 surfaceColor, float thickness) {
  if(!uSSSEnable || uSSSAmbient <= 0.0) {
    return vec3(0.0);
  }

  // Thickness-based ambient contribution
  float thicknessAttenuation = pow(max(0.0, 1.0 - thickness), uSSSThicknessPower);

  // Ambient SSS contribution
  vec3 ambientSSS = surfaceColor * uSSSColor * uSSSAmbient * thicknessAttenuation;

  return ambientSSS;
}

// =============================================================================
// LIGHTING CALCULATION
// =============================================================================
vec3 calculateLighting(vec3 surfaceColor, vec3 normal_lf_view, vec3 normal_hf_view, float thickness) {
  vec3 V = normalize(vViewPosition);
  vec3 fragmentPosView = -vViewPosition;

  vec3 totalDiffuse = vec3(0.0);
  vec3 totalSpecular = vec3(0.0);
  vec3 totalSSS = vec3(0.0);

  // Get shininess modulation from texture
  float shininessFromMap = texture2D(uShininessMap, vUvAdvectedCoord).r;

  // Process each light
  for(int i = 0; i < MAX_LIGHTS; ++i) {
    if(i >= uNumLights || !uLights[i].enabled)
      continue;

    Light currentLight = uLights[i];
    vec3 L = normalize(currentLight.position - fragmentPosView);

    // Diffuse lighting with blended normal
    vec3 N_diffuse = normalize(mix(normal_lf_view, normal_hf_view, currentLight.diffuseHfNormalAmount));
    float NdotL_diffuse = max(dot(N_diffuse, L), 0.0);
    totalDiffuse += currentLight.color * NdotL_diffuse *
      currentLight.diffuseContribution * currentLight.intensity;

    // Specular lighting with blended normal
    vec3 N_specular = normalize(mix(normal_lf_view, normal_hf_view, currentLight.specularHfNormalAmount));
    vec3 H = normalize(L + V);
    float NdotH_specular = max(dot(N_specular, H), 0.0);

    // Modulate shininess with texture
    float finalShininess = max(currentLight.specularShininess * shininessFromMap, 1.0);
    float specularFactor = pow(NdotH_specular, finalShininess);
    totalSpecular += currentLight.color * specularFactor *
      currentLight.specularContribution * currentLight.intensity *
      (finalShininess / 255.0);

    // Subsurface scattering contribution
    totalSSS += calculateSubsurfaceScattering(surfaceColor, currentLight.color, L, N_diffuse, V, thickness, currentLight.intensity, currentLight.sssContribution);
  }

  // Add ambient subsurface scattering
  totalSSS += calculateAmbientSSS(surfaceColor, thickness);

  // Combine lighting components
  vec3 ambientColor = uAmbientIntensity * surfaceColor;
  vec3 litColor = ambientColor + (totalDiffuse * surfaceColor) + totalSpecular + totalSSS;

  return litColor;
}

// =============================================================================
// MAIN FRAGMENT SHADER
// =============================================================================
void main() {
  // Calculate surface color
  vec3 color = saturation(calculateSurfaceColor(), uColorSaturation);

  // CTAO Texture - Curvature, Thickness, Ambient Occlusion
  vec3 ctao = texture2D(uCTAOMap, vUvAdvectedCoord).xyz;
  float thickness = ctao.y;

  // Calculate normal with proper advection correction
  vec3 normMapLF_advected = transformNormalWithAdvectionCorrection(texture2D(uNormalMapLF, vUvAdvectedCoord).xyz);
  vec3 normMap_advected = transformNormalWithAdvectionCorrection(texture2D(uNormalMap, vUvAdvectedCoord).xyz);

  // Apply lighting with subsurface scattering
  vec3 litColor = calculateLighting(color, normMapLF_advected, normMap_advected, thickness);

  vec3 N_diffuse = normalize(mix(normMapLF_advected, normMap_advected, uLights[0].diffuseHfNormalAmount));
  // Apply gamma correction
  litColor = pow(litColor, vec3(1.0 / 2.2));
  // pos_advected = transformByAdvection(vWorldPosition, vUvAdvectedCoord);
  // litColor = vec3(vAdvectedCoord.xy, 0.0);

  gl_FragColor = vec4(vec3(litColor), 1.0);
}