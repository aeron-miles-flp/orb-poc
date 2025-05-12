// Uniforms
uniform float uTime;
uniform float uVertexDisplacementScale;

// LF (Low Frequency) Noise parameters
uniform float uLfMasterScale;
uniform float uLfPatternTimeScale;
uniform float uLfOctaves;
uniform float uLfLacunarity;
uniform float uLfPersistence;
uniform float uLfNoiseContrastLower;
uniform float uLfNoiseContrastUpper;
uniform float uLfNormalStrength; // Strength for perturbing normals based on LF noise

// Flow Parameters for LF noise
uniform float uLfFlowScale;
uniform float uLfFlowStrength;
uniform float uLfFlowTimeScale;

uniform bool uEnablePeriodicLf;
uniform vec3 uPeriodicLengthLf;

// Varyings
varying float vNoise;
varying vec3 vNormal; // This will carry the new, perturbed normal
varying vec3 vViewPosition;
varying vec3 vWorldPosition; // For fragment shader calculations
varying vec3 vAdvectedCoord; // The coordinate used for FBM lookup, for color noise
varying float vPatternTime;   // The time used for FBM lookup, for color noise
varying float vGlobalTime;
varying float vThickness;
varying vec2 vUv; // << NEW: To pass UVs to the fragment shader

// --- Nikita Miropolskiy's Simplex Noise (3D) ---
vec3 random3(vec3 c) {
  float j = 4096.0 * sin(dot(c, vec3(17.0, 59.4, 15.0)));
  vec3 r;
  r.z = fract(512.0 * j);
  j *= .125;
  r.x = fract(512.0 * j);
  j *= .125;
  r.y = fract(512.0 * j);
  return r - 0.5;
}
const float F3 = 0.3333333;
const float G3 = 0.1666667;
float simplex3d(vec3 p) {
  vec3 s = floor(p + dot(p, vec3(F3)));
  vec3 x = p - s + dot(s, vec3(G3));
  vec3 e = step(vec3(0.0), x - x.yzx);
  vec3 i1 = e * (1.0 - e.zxy);
  vec3 i2 = 1.0 - e.zxy * (1.0 - e);
  vec3 x1 = x - i1 + G3;
  vec3 x2 = x - i2 + 2.0 * G3;
  vec3 x3 = x - 1.0 + 3.0 * G3;
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
// --- End of Nikita Miropolskiy's Simplex Noise ---

vec3 get_flow_vector(vec3 flow_coord_base, float flow_anim_time) {
  float flow_x = simplex3d(vec3(flow_coord_base.x + 13.7, flow_coord_base.y + 27.3, flow_coord_base.z + 5.1 + flow_anim_time));
  float flow_y = simplex3d(vec3(flow_coord_base.x - 9.2, flow_coord_base.y - 18.4, flow_coord_base.z - 11.9 + flow_anim_time));
  float flow_z = simplex3d(vec3(flow_coord_base.x + 31.5, flow_coord_base.y - 42.6, flow_coord_base.z + 17.8 + flow_anim_time));
  return vec3(flow_x, flow_y, flow_z);
}

float simplex3d_evolving_periodic(vec3 p_spatial, float time_component, vec3 period) {
  vec3 p_fract_norm = p_spatial / period;
  vec3 Pi0 = floor(p_fract_norm);
  vec3 Pf0 = p_fract_norm - Pi0;
  vec3 u = Pf0 * Pf0 * Pf0 * (Pf0 * (Pf0 * 6.0 - 15.0) + 10.0);

  float n000 = simplex3d(vec3(Pf0.x * period.x, Pf0.y * period.y, Pf0.z * period.z + time_component));
  float n100 = simplex3d(vec3((Pf0.x - 1.0) * period.x, Pf0.y * period.y, Pf0.z * period.z + time_component));
  float n010 = simplex3d(vec3(Pf0.x * period.x, (Pf0.y - 1.0) * period.y, Pf0.z * period.z + time_component));
  float n110 = simplex3d(vec3((Pf0.x - 1.0) * period.x, (Pf0.y - 1.0) * period.y, Pf0.z * period.z + time_component));
  float n001 = simplex3d(vec3(Pf0.x * period.x, Pf0.y * period.y, (Pf0.z - 1.0) * period.z + time_component));
  float n101 = simplex3d(vec3((Pf0.x - 1.0) * period.x, Pf0.y * period.y, (Pf0.z - 1.0) * period.z + time_component));
  float n011 = simplex3d(vec3(Pf0.x * period.x, (Pf0.y - 1.0) * period.y, (Pf0.z - 1.0) * period.z + time_component));
  float n111 = simplex3d(vec3((Pf0.x - 1.0) * period.x, (Pf0.y - 1.0) * period.y, (Pf0.z - 1.0) * period.z + time_component));

  float nx00 = mix(n000, n100, u.x);
  float nx10 = mix(n010, n110, u.x);
  float nx01 = mix(n001, n101, u.x);
  float nx11 = mix(n011, n111, u.x);
  float nxy0 = mix(nx00, nx10, u.y);
  float nxy1 = mix(nx01, nx11, u.y);
  return mix(nxy0, nxy1, u.z);
}

float fbm_simplex3d_advected(vec3 p_advected_spatial_base, float pattern_time_val, int octaves, float lacunarity, float persistence, bool enablePeriodic, vec3 periodicLength) {
  float total = 0.0;
  float frequency = 1.0;
  float amplitude = 1.0;
  float maxValue = 0.0;

  for(int i = 0; i < octaves; i++) {
    float noise_sample;
    vec3 current_p_spatial = p_advected_spatial_base * frequency;
    float current_pattern_time_component = pattern_time_val;

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
  if(maxValue == 0.0)
    return 0.0;
  return total / maxValue;
}

void main() {
    // Pass UV coordinates to fragment shader
  vUv = uv; // << ASSIGN UVs

    // 1. Calculate Flow Field
  vec3 flow_sample_coord = position * uLfFlowScale;
  float flow_anim_time = uTime * uLfFlowTimeScale;
  vec3 flow_vector = get_flow_vector(flow_sample_coord, flow_anim_time);

    // 2. Prepare base coordinate for main FBM pattern
  vec3 p_local_spatial = position * uLfMasterScale;

    // 3. Advect the base coordinate
  vec3 advected_spatial_coord = p_local_spatial + flow_vector * uLfFlowStrength;

    // 4. Determine time for main FBM pattern evolution
  float pattern_anim_time = uTime * uLfPatternTimeScale;

    // PASS ADVECTION INFO TO FRAGMENT SHADER
  vAdvectedCoord = advected_spatial_coord;
  vPatternTime = pattern_anim_time;
  vGlobalTime = uTime; // Pass global time for HF evolution in frag

    // 5. Calculate FBM on advected coordinates for DISPLACEMENT
  int octaves = int(uLfOctaves);
  float noiseVal = 0.0; // This is the FBM output in approx [-1, 1] range

  if(octaves > 0) {
    noiseVal = fbm_simplex3d_advected(advected_spatial_coord, pattern_anim_time, octaves, uLfLacunarity, uLfPersistence, uEnablePeriodicLf, uPeriodicLengthLf);
  }

    // 6. Calculate Perturbed Normal for LF displacement
  vec3 final_normal = normal;
  if(uLfNormalStrength > 0.0 && octaves > 0) {
    float eps = 0.01;
    float noise_x_plus = fbm_simplex3d_advected(advected_spatial_coord + vec3(eps, 0.0, 0.0), pattern_anim_time, octaves, uLfLacunarity, uLfPersistence, uEnablePeriodicLf, uPeriodicLengthLf);
    float noise_x_minus = fbm_simplex3d_advected(advected_spatial_coord - vec3(eps, 0.0, 0.0), pattern_anim_time, octaves, uLfLacunarity, uLfPersistence, uEnablePeriodicLf, uPeriodicLengthLf);
    float noise_y_plus = fbm_simplex3d_advected(advected_spatial_coord + vec3(0.0, eps, 0.0), pattern_anim_time, octaves, uLfLacunarity, uLfPersistence, uEnablePeriodicLf, uPeriodicLengthLf);
    float noise_y_minus = fbm_simplex3d_advected(advected_spatial_coord - vec3(0.0, eps, 0.0), pattern_anim_time, octaves, uLfLacunarity, uLfPersistence, uEnablePeriodicLf, uPeriodicLengthLf);
    float noise_z_plus = fbm_simplex3d_advected(advected_spatial_coord + vec3(0.0, 0.0, eps), pattern_anim_time, octaves, uLfLacunarity, uLfPersistence, uEnablePeriodicLf, uPeriodicLengthLf);
    float noise_z_minus = fbm_simplex3d_advected(advected_spatial_coord - vec3(0.0, 0.0, eps), pattern_anim_time, octaves, uLfLacunarity, uLfPersistence, uEnablePeriodicLf, uPeriodicLengthLf);
    vec3 fbm_gradient = vec3((noise_x_plus - noise_x_minus) / (2.0 * eps), (noise_y_plus - noise_y_minus) / (2.0 * eps), (noise_z_plus - noise_z_minus) / (2.0 * eps));
    final_normal = normalize(normal - fbm_gradient * uLfNormalStrength);
  }

    // 7. Calculate final displaced position
  float actual_displacement = noiseVal * uVertexDisplacementScale;
  vec3 displacedPosition = position + normal * actual_displacement;

    // --- Output to Rasterizer & Fragment Shader ---
  gl_Position = projectionMatrix * modelViewMatrix * vec4(displacedPosition, 1.0);

    // World position of the displaced vertex
  vWorldPosition = (modelMatrix * vec4(displacedPosition, 1.0)).xyz;

  vNormal = normalize(normalMatrix * final_normal);
  vec4 mvPosition = modelViewMatrix * vec4(displacedPosition, 1.0);
  vViewPosition = -mvPosition.xyz;

  float noiseVal01 = noiseVal * 0.5 + 0.5;
  noiseVal01 = (noiseVal01 - uLfNoiseContrastLower) / max(uLfNoiseContrastUpper - uLfNoiseContrastLower, 0.00001);
  vNoise = clamp(noiseVal01, 0.0, 1.0);
  vThickness = vNoise;
}