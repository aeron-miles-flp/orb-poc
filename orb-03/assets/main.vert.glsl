// =============================================================================
// VERTEX SHADER - CLEANED AND ENHANCED WITH TANGENT SPACE
// =============================================================================

// =============================================================================
// ATTRIBUTES - MESH DATA
// =============================================================================
attribute vec3 tangent;

// =============================================================================
// UNIFORMS - GLOBAL PARAMETERS
// =============================================================================
uniform float uTime;
uniform float uVertexDisplacementScale;
uniform float uVertexAnimationAmount;
uniform float uVertexAnimationSpeed;

// =============================================================================
// LOW FREQUENCY NOISE UNIFORMS
// =============================================================================
uniform float uLfMasterScale;
uniform float uLfPatternTimeScale;
uniform float uLfOctaves;
uniform float uLfLacunarity;
uniform float uLfPersistence;
uniform float uLfNoiseContrastLower;
uniform float uLfNoiseContrastUpper;
uniform float uLfNormalStrength;

// =============================================================================
// FLOW PARAMETERS FOR LF NOISE
// =============================================================================
uniform float uLfFlowScale;
uniform float uLfFlowStrength;
uniform float uLfFlowTimeScale;

// =============================================================================
// PERIODIC NOISE PARAMETERS
// =============================================================================
uniform bool uEnablePeriodicLf;
uniform vec3 uPeriodicLengthLf;

// =============================================================================
// DISPLACEMENT MAP UNIFORMS
// =============================================================================
uniform sampler2D uDispMap;
uniform float uDispMapStrength;         // Strength multiplier for displacement map
uniform float uDispMapNormalStrength;   // Strength for displacement map normals

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
vec3 random3(vec3 c) {
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
const float F3 = 0.3333333;
const float G3 = 0.1666667;

float simplex3d(vec3 p) {
    // Skew input space to determine which simplex cell we're in
  vec3 s = floor(p + dot(p, vec3(F3)));
  vec3 x = p - s + dot(s, vec3(G3));

    // Calculate the other three corners of the tetrahedron
  vec3 e = step(vec3(0.0), x - x.yzx);
  vec3 i1 = e * (1.0 - e.zxy);
  vec3 i2 = 1.0 - e.zxy * (1.0 - e);

  vec3 x1 = x - i1 + G3;
  vec3 x2 = x - i2 + 2.0 * G3;
  vec3 x3 = x - 1.0 + 3.0 * G3;

    // Calculate contributions from each corner
  vec4 w, d;
  w.x = dot(x, x);
  w.y = dot(x1, x1);
  w.z = dot(x2, x2);
  w.w = dot(x3, x3);
  w = max(0.6 - w, 0.0);

  d.x = dot(random3(s), x);
  d.y = dot(random3(s + i1), x1);
  d.z = dot(random3(s + i2), x2);
  d.w = dot(random3(s + 1.0), x3);

  w *= w;
  w *= w;
  d *= w;

  return dot(d, vec4(52.0));
}

// =============================================================================
// FLOW AND ADVECTION FUNCTIONS
// =============================================================================

// Generate flow vector for noise advection
vec3 get_flow_vector(vec3 flow_coord_base, float flow_anim_time) {
  float flow_x = simplex3d(vec3(flow_coord_base.x + 13.7, flow_coord_base.y + 27.3, flow_coord_base.z + 5.1 + flow_anim_time));

  float flow_y = simplex3d(vec3(flow_coord_base.x - 9.2, flow_coord_base.y - 18.4, flow_coord_base.z - 11.9 + flow_anim_time));

  float flow_z = simplex3d(vec3(flow_coord_base.x + 31.5, flow_coord_base.y - 42.6, flow_coord_base.z + 17.8 + flow_anim_time));

  return vec3(flow_x, flow_y, flow_z);
}

// =============================================================================
// IMPROVED UV ADVECTION FUNCTIONS
// =============================================================================

// Calculate UV flow with reduced stretching
vec2 calculateAdvectedUV(vec2 baseUV, vec3 flow_vector, float flowStrength) {
  // Apply flow to UV coordinates
  vec2 flowOffset = flow_vector.xy * flowStrength;
  return baseUV + flowOffset;
}

// =============================================================================
// PERIODIC NOISE FUNCTIONS
// =============================================================================

// Periodic simplex noise with time evolution
float simplex3d_evolving_periodic(vec3 p_spatial, float time_component, vec3 period) {
  vec3 p_fract_norm = p_spatial / period;
  vec3 Pi0 = floor(p_fract_norm);
  vec3 Pf0 = p_fract_norm - Pi0;
  vec3 u = Pf0 * Pf0 * Pf0 * (Pf0 * (Pf0 * 6.0 - 15.0) + 10.0);

    // Sample noise at all 8 corners of the unit cube
  float n000 = simplex3d(vec3(Pf0.x * period.x, Pf0.y * period.y, Pf0.z * period.z + time_component));
  float n100 = simplex3d(vec3((Pf0.x - 1.0) * period.x, Pf0.y * period.y, Pf0.z * period.z + time_component));
  float n010 = simplex3d(vec3(Pf0.x * period.x, (Pf0.y - 1.0) * period.y, Pf0.z * period.z + time_component));
  float n110 = simplex3d(vec3((Pf0.x - 1.0) * period.x, (Pf0.y - 1.0) * period.y, Pf0.z * period.z + time_component));
  float n001 = simplex3d(vec3(Pf0.x * period.x, Pf0.y * period.y, (Pf0.z - 1.0) * period.z + time_component));
  float n101 = simplex3d(vec3((Pf0.x - 1.0) * period.x, Pf0.y * period.y, (Pf0.z - 1.0) * period.z + time_component));
  float n011 = simplex3d(vec3(Pf0.x * period.x, (Pf0.y - 1.0) * period.y, (Pf0.z - 1.0) * period.z + time_component));
  float n111 = simplex3d(vec3((Pf0.x - 1.0) * period.x, (Pf0.y - 1.0) * period.y, (Pf0.z - 1.0) * period.z + time_component));

    // Trilinear interpolation
  float nx00 = mix(n000, n100, u.x);
  float nx10 = mix(n010, n110, u.x);
  float nx01 = mix(n001, n101, u.x);
  float nx11 = mix(n011, n111, u.x);
  float nxy0 = mix(nx00, nx10, u.y);
  float nxy1 = mix(nx01, nx11, u.y);

  return mix(nxy0, nxy1, u.z);
}

// =============================================================================
// FRACTAL BROWNIAN MOTION
// =============================================================================

// FBM with flow advection and optional periodicity
float fbm_simplex3d_advected(
  vec3 p_advected_spatial_base,
  float pattern_time_val,
  int octaves,
  float lacunarity,
  float persistence,
  bool enablePeriodic,
  vec3 periodicLength
) {
  float total = 0.0;
  float frequency = 1.0;
  float amplitude = 1.0;
  float maxValue = 0.0;

  for(int i = 0; i < octaves; i++) {
    vec3 current_p_spatial = p_advected_spatial_base * frequency;
    float current_pattern_time_component = pattern_time_val;

    float noise_sample;
    if(enablePeriodic && periodicLength.x > 0.0 && periodicLength.y > 0.0 && periodicLength.z > 0.0) {
      noise_sample = simplex3d_evolving_periodic(current_p_spatial, current_pattern_time_component, periodicLength);
    } else {
      noise_sample = simplex3d(vec3(current_p_spatial.xy, current_p_spatial.z + current_pattern_time_component));
    }

    total += noise_sample * amplitude;
    maxValue += amplitude;
    amplitude *= persistence;
    frequency *= lacunarity;
  }

  return (maxValue == 0.0) ? 0.0 : total / maxValue;
}

// =============================================================================
// TANGENT SPACE CALCULATION
// =============================================================================

// Calculate tangent and bitangent from mesh attributes
void calculateTangentSpace(
  vec3 meshNormal,
  vec3 meshTangent,
  out vec3 finalTangent,
  out vec3 finalBitangent
) {
  // Normalize the input normal and tangent
  vec3 N = normalize(meshNormal);
  vec3 T = normalize(meshTangent);

  // Gram-Schmidt orthogonalization to ensure tangent is perpendicular to normal
  T = normalize(T - dot(T, N) * N);

  // Calculate bitangent using cross product
  // Note: The handedness (w component) of the tangent attribute determines the direction
  // For standard meshes, we assume right-handed coordinate system
  vec3 B = cross(N, T);

  finalTangent = T;
  finalBitangent = B;
}

// =============================================================================
// DISPLACEMENT MAP NORMAL CALCULATION
// =============================================================================

// Calculate normals from displacement map using finite differences
vec3 calculateDisplacementMapNormals(vec2 uvCoord, float strength) {
  if(strength <= 0.0) {
    return vec3(0.0, 0.0, 1.0); // Return "no change" normal in tangent space
  }

    // Sample displacement at current position and neighbors
  float dispCenter = texture2D(uDispMap, uvCoord).r;

    // Calculate UV step size (adjust based on texture resolution if needed)
  vec2 texelSize = vec2(1.0) / vec2(1024.0); // Assuming 1024x1024 texture, adjust as needed

    // Sample neighboring pixels
  float dispRight = texture2D(uDispMap, uvCoord + vec2(texelSize.x, 0.0)).r;
  float dispUp = texture2D(uDispMap, uvCoord + vec2(0.0, texelSize.y)).r;

    // Calculate gradient using finite differences
  float gradientX = (dispRight - dispCenter) / texelSize.x;
  float gradientY = (dispUp - dispCenter) / texelSize.y;

    // Convert gradient to normal (in tangent space)
    // The normal points in the direction opposite to the gradient
  vec3 normal_tangent = normalize(vec3(-gradientX * strength, -gradientY * strength, 1.0));

  return normal_tangent;
}

// Transform displacement map normal from tangent space to object space using proper TBN
vec3 transformDisplacementNormalToObjectSpace(
  vec3 normal_tangent,
  vec3 objTangent,
  vec3 objBitangent,
  vec3 objNormal
) {
  // Build TBN matrix in object space using the calculated tangent and bitangent
  mat3 TBN = mat3(normalize(objTangent), normalize(objBitangent), normalize(objNormal));

  // Transform the displacement normal to object space
  return normalize(TBN * normal_tangent);
}

// =============================================================================
// LOW FREQUENCY NORMAL CALCULATION
// =============================================================================

// Calculate normals from LF noise using gradient
vec3 calculateLFNormals(
  vec3 advected_spatial_coord,
  float pattern_anim_time,
  int octaves,
  float lacunarity,
  float persistence,
  bool enablePeriodic,
  vec3 periodicLength,
  float normalStrength
) {
  if(normalStrength <= 0.0 || octaves <= 0) {
    return vec3(0.0); // No perturbation
  }

  float eps = 0.01;

    // Sample noise at center and offset positions for gradient calculation
  float noise_center = fbm_simplex3d_advected(advected_spatial_coord, pattern_anim_time, octaves, lacunarity, persistence, enablePeriodic, periodicLength);

  float noise_x_plus = fbm_simplex3d_advected(advected_spatial_coord + vec3(eps, 0.0, 0.0), pattern_anim_time, octaves, lacunarity, persistence, enablePeriodic, periodicLength);
  float noise_x_minus = fbm_simplex3d_advected(advected_spatial_coord - vec3(eps, 0.0, 0.0), pattern_anim_time, octaves, lacunarity, persistence, enablePeriodic, periodicLength);

  float noise_y_plus = fbm_simplex3d_advected(advected_spatial_coord + vec3(0.0, eps, 0.0), pattern_anim_time, octaves, lacunarity, persistence, enablePeriodic, periodicLength);
  float noise_y_minus = fbm_simplex3d_advected(advected_spatial_coord - vec3(0.0, eps, 0.0), pattern_anim_time, octaves, lacunarity, persistence, enablePeriodic, periodicLength);

  float noise_z_plus = fbm_simplex3d_advected(advected_spatial_coord + vec3(0.0, 0.0, eps), pattern_anim_time, octaves, lacunarity, persistence, enablePeriodic, periodicLength);
  float noise_z_minus = fbm_simplex3d_advected(advected_spatial_coord - vec3(0.0, 0.0, eps), pattern_anim_time, octaves, lacunarity, persistence, enablePeriodic, periodicLength);

    // Calculate gradient using central differences
  vec3 fbm_gradient = vec3((noise_x_plus - noise_x_minus) / (2.0 * eps), (noise_y_plus - noise_y_minus) / (2.0 * eps), (noise_z_plus - noise_z_minus) / (2.0 * eps));

  return fbm_gradient * normalStrength;
}

// =============================================================================
// DISPLACEMENT CALCULATION
// =============================================================================

// Calculate final displacement amount
float calculateDisplacement(float noiseVal, float displacement_anim_factor) {
    // Apply noise-based displacement
  float noiseDisp = noiseVal * uVertexDisplacementScale;
  float procedural_displacement = mix(noiseDisp, noiseDisp * displacement_anim_factor, uVertexAnimationAmount);

    // Sample displacement map
  float dispMapSample = texture2D(uDispMap, vUvAdvectedCoord).r;
  float dispMapDisplacement = (dispMapSample - 0.5) * 0.6;

    // Combine procedural and texture-based displacement
  return procedural_displacement + dispMapDisplacement;
}

// =============================================================================
// MAIN VERTEX SHADER
// =============================================================================

void main() {
  // Store base UV coordinates
  vUv = uv;
  vGlobalTime = uTime;

  // CALCULATE TANGENT SPACE VECTORS
  vec3 objTangent, objBitangent;
  calculateTangentSpace(normal, tangent, objTangent, objBitangent);

  // CALCULATE FLOW AND ADVECTION
  vec3 flow_sample_coord = position * uLfFlowScale;
  float flow_anim_time = uTime * uLfFlowTimeScale;
  vec3 flow_vector = get_flow_vector(flow_sample_coord, flow_anim_time);

  vec3 p_local_spatial = position * uLfMasterScale;
  vec3 advected_spatial_coord = p_local_spatial + flow_vector * uLfFlowStrength;
  float pattern_anim_time = uTime * uLfPatternTimeScale;

  // Store advected coordinates for fragment shader
  vAdvectedCoord = advected_spatial_coord;

  // Calculate advected UV
  vUvAdvectedCoord = calculateAdvectedUV(uv, flow_vector, uLfFlowStrength * 0.3);

  vPatternTime = pattern_anim_time;

  // CALCULATE LOW FREQUENCY NOISE
  // int octaves = int(uLfOctaves);
  // float noiseVal = 0.0;

  // if(octaves > 0) {
  //   noiseVal = fbm_simplex3d_advected(advected_spatial_coord, pattern_anim_time, octaves, uLfLacunarity, uLfPersistence, uEnablePeriodicLf, uPeriodicLengthLf);
  // }

  // CALCULATE DISPLACEMENT ANIMATION FACTOR
  vDisplacementAnimFactor = simple_noise1d_pos1(uTime * uVertexAnimationSpeed);

  // CALCULATE DISPLACEMENT MAP NORMALS
  // vec3 dispMapNormal_tangent = calculateDisplacementMapNormals(vUvAdvectedCoord, uDispMapNormalStrength);
  // vec3 dispMapNormal_object = transformDisplacementNormalToObjectSpace(dispMapNormal_tangent, objTangent, objBitangent, normal);

  // CALCULATE LOW FREQUENCY NORMALS
  // vec3 lfNormalPerturbation = calculateLFNormals(advected_spatial_coord, pattern_anim_time, octaves, uLfLacunarity, uLfPersistence, uEnablePeriodicLf, uPeriodicLengthLf, uLfNormalStrength);

  // COMBINE ALL NORMALS
  vec3 final_normal = normalize(normal);// - lfNormalPerturbation + dispMapNormal_object);
  vObjectNormal = final_normal;

  // CALCULATE FINAL DISPLACEMENT
  float actual_displacement = texture2D(uDispMap, vUvAdvectedCoord).x - 0.5;

  float dispOffset = simplex3d(vec3(position.xy, uTime * uVertexAnimationSpeed));
  actual_displacement = actual_displacement + (vDisplacementAnimFactor * uVertexAnimationAmount * dispOffset);

  // APPLY DISPLACEMENT TO POSITION
  vec3 displacedPosition = position + normal * actual_displacement * uVertexDisplacementScale;

  // TRANSFORM TO SCREEN SPACE
  gl_Position = projectionMatrix * modelViewMatrix * vec4(displacedPosition, 1.0);

  // CALCULATE OUTPUT VARYINGS
  vWorldPosition = (modelMatrix * vec4(displacedPosition, 1.0)).xyz;

  // Transform tangent space vectors to view space for fragment shader
  vNormal = normalize(normalMatrix * final_normal);
  vTangent = normalize(normalMatrix * objTangent);
  vBitangent = normalize(normalMatrix * objBitangent);

  vec4 mvPosition = modelViewMatrix * vec4(displacedPosition, 1.0);
  vViewPosition = -mvPosition.xyz;

  //   // PROCESS NOISE VALUE FOR FRAGMENT SHADER
  // float noiseVal01 = noiseVal * 0.5 + 0.5;
  // noiseVal01 = (noiseVal01 - uLfNoiseContrastLower) / max(uLfNoiseContrastUpper - uLfNoiseContrastLower, 0.00001);
  // vNoise = clamp(noiseVal01, 0.0, 1.0);

  // Override with actual displacement for thickness calculation
  vNoise = actual_displacement;
}