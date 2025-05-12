# WebGL Animated Orb

A WebGL-based 3D visualization that renders an animated "orb" with procedural noise-based displacement and normal mapping effects built with Three.js and WebGL.

## Recent Optimizations

The visualization has been improved with the following changes:

1. **In-Shader Noise Generation**:
   - Noise generation now happens directly in the main shaders
   - Eliminated separate render targets for noise textures
   - Reduced memory usage and improved performance

2. **Optimized Flow Noise Implementation**:
   - Flow noise computation integrated into vertex and fragment shaders
   - Low-frequency noise for vertex displacement in vertex shader
   - High-frequency noise for normal mapping in fragment shader

3. **Simplified Animation Loop**:
   - Streamlined rendering pipeline with fewer state changes
   - Direct parameter control without intermediate render targets
   - Improved responsiveness of GUI controls

## Features

- **Real-time vertex displacement** using procedural flow noise
- **Detail normal mapping** with high-frequency noise for surface details
- **Interactive parameters** via GUI controls
- **Animated noise evolution** over time
- **Optimized shader implementation** with in-shader noise generation

## Technical Implementation

The visualization uses Three.js to create a WebGL context and set up the scene. The procedural effect is created through:

1. **Main vertex shader** (`main.vert.glsl`):
   - Handles vertex displacement using low-frequency flow noise
   - Computes normal perturbation for the displaced geometry

2. **Main fragment shader** (`main.frag.glsl`):
   - Applies high-frequency noise for detail normal mapping
   - Implements lighting calculations

3. **Flow-based noise generation**:
   - Integrated directly in both shaders
   - Simplex noise patterns for base noise
   - Flow fields that advect the noise coordinates
   - Multi-octave FBM (Fractal Brownian Motion) for detail

## Running the Project

To run the project:

```bash
python -m http.server 8000
```

Then navigate to http://localhost:8000 in your web browser.

**Note**: The project must be run from a web server, not directly from the file system, due to CORS restrictions when loading shader files.

## Controls

The project includes a comprehensive GUI that allows you to modify various parameters:

- **Vertex Displacement Scale**: Controls the amount of displacement
- **Low-Frequency Noise Parameters**:
  - Master Scale, Flow Field Scale, Flow Strength
  - Octaves, Lacunarity, Persistence
  - Contrast settings and Normal Strength
- **High-Frequency Noise Parameters**:
  - Similar settings for the detail normal map
- **Scene Controls**:
  - Auto-rotation toggle and speed
  - Lighting intensity and ambient light

## Project Structure

- `index.html` - Main HTML file with Three.js setup and visualization code
- `main.vert.glsl` - Vertex shader for the sphere with displacement
- `main.frag.glsl` - Fragment shader for lighting and normal mapping

## Development Notes

- The application requires WebGL support in the browser
- Three.js v0.164.1 is loaded from CDN alongside the lil-gui library for the controls