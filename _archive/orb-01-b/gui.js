// This file contains all the GUI overlay code, which was removed from the HTML.
// It defines functions for setting up the dat.GUI interface and related callbacks.

function updateActivePreset()
{
    // If no active preset is tracked, try to determine one
    if (!currentActivePreset && orbOverlay && orbOverlay.visible)
    {
        // Here we choose preset1 if particles are visible, otherwise preset2.
        // Adjust logic as needed.
        currentActivePreset = "preset1";
        console.log("Detected active preset:", currentActivePreset);
    }
    if (currentActivePreset && orbOverlay && orbOverlay.visible)
    {
        console.log("Applying updates to preset:", currentActivePreset);
        const preset = config.presets[currentActivePreset];
        if (preset.shader)
        {
            if (preset.shader.intensity !== undefined)
                config.shader.intensity = preset.shader.intensity;
            if (preset.shader.noiseScale !== undefined)
                config.shader.noiseScale = preset.shader.noiseScale;
            if (preset.shader.pulseSpeed !== undefined)
                config.shader.pulseSpeed = preset.shader.pulseSpeed;
            if (preset.shader.emberColor !== undefined)
                config.shader.emberColor = preset.shader.emberColor;
            if (preset.shader.glowColor !== undefined)
                config.shader.glowColor = preset.shader.glowColor;
            if (preset.shader.pulseVariation !== undefined)
                config.shader.pulseVariation = preset.shader.pulseVariation;
            if (preset.shader.patternScale !== undefined)
                config.shader.patternScale = preset.shader.patternScale;
            if (preset.shader.displacementAmount !== undefined)
                config.shader.displacementAmount = preset.shader.displacementAmount;
            if (preset.shader.layer2)
            {
                if (preset.shader.layer2.enabled !== undefined)
                    config.shader.layer2.enabled = preset.shader.layer2.enabled;
                if (preset.shader.layer2.intensity !== undefined)
                    config.shader.layer2.intensity = preset.shader.layer2.intensity;
                if (preset.shader.layer2.noiseScale !== undefined)
                    config.shader.layer2.noiseScale = preset.shader.layer2.noiseScale;
                if (preset.shader.layer2.pulseSpeed !== undefined)
                    config.shader.layer2.pulseSpeed = preset.shader.layer2.pulseSpeed;
                if (preset.shader.layer2.emberColor !== undefined)
                    config.shader.layer2.emberColor = preset.shader.layer2.emberColor;
                if (preset.shader.layer2.glowColor !== undefined)
                    config.shader.layer2.glowColor = preset.shader.layer2.glowColor;
                if (preset.shader.layer2.pulseVariation !== undefined)
                    config.shader.layer2.pulseVariation = preset.shader.layer2.pulseVariation;
                if (preset.shader.layer2.patternScale !== undefined)
                    config.shader.layer2.patternScale = preset.shader.layer2.patternScale;
                if (preset.shader.layer2.blendFactor !== undefined)
                    config.shader.layer2.blendFactor = preset.shader.layer2.blendFactor;
            }
        }
        // Update shader uniforms
        if (orbOverlay && orbOverlay.material && orbOverlay.material.uniforms)
        {
            updateShaderUniforms();
        }
        // Update GUI controls
        if (gui)
        {
            gui.__controllers.forEach((controller) => controller.updateDisplay());
        }
    }
}

function setupGUI()
{
    gui = new dat.GUI({ autoPlace: false });
    document.getElementById("controls").appendChild(gui.domElement);

    const presetsFolder = gui.addFolder("Preset Editor");

    // Preset 1 (Red+Blue)
    const effect1Folder = presetsFolder.addFolder("Preset 1 (Red+Blue)");
    const effect1MainFolder = effect1Folder.addFolder("Main Layer");
    effect1MainFolder
        .add(config.presets.preset1.shader, "intensity", 0, 3)
        .name("Intensity")
        .onChange(updateActivePreset);
    effect1MainFolder
        .add(config.presets.preset1.shader, "noiseScale", 0.1, 20)
        .name("Noise Scale")
        .onChange(updateActivePreset);
    effect1MainFolder
        .add(config.presets.preset1.shader, "pulseSpeed", 0, 2)
        .name("Pulse Speed")
        .onChange(updateActivePreset);
    effect1MainFolder
        .add(config.presets.preset1.shader, "pulseVariation", 0, 1)
        .name("Pulse Variation")
        .onChange(updateActivePreset);
    effect1MainFolder
        .add(config.presets.preset1.shader, "patternScale", 0.5, 10)
        .name("Pattern Scale")
        .onChange(updateActivePreset);
    effect1MainFolder
        .add(config.presets.preset1.shader, "displacementAmount", -0.2, 0.2)
        .name("Displacement")
        .onChange(updateActivePreset);
    effect1MainFolder
        .addColor(config.presets.preset1.shader, "emberColor")
        .name("Ember Color")
        .onChange(updateActivePreset);
    effect1MainFolder
        .addColor(config.presets.preset1.shader, "glowColor")
        .name("Glow Color")
        .onChange(updateActivePreset);

    const effect1Layer2Folder = effect1Folder.addFolder("Second Layer");
    effect1Layer2Folder
        .add(config.presets.preset1.shader.layer2, "enabled")
        .name("Enable Layer 2")
        .onChange(updateActivePreset);
    effect1Layer2Folder
        .add(config.presets.preset1.shader.layer2, "intensity", 0, 3)
        .name("Intensity")
        .onChange(updateActivePreset);
    effect1Layer2Folder
        .add(config.presets.preset1.shader.layer2, "noiseScale", 0.1, 20)
        .name("Noise Scale")
        .onChange(updateActivePreset);
    effect1Layer2Folder
        .add(config.presets.preset1.shader.layer2, "pulseSpeed", 0, 2)
        .name("Pulse Speed")
        .onChange(updateActivePreset);
    effect1Layer2Folder
        .add(config.presets.preset1.shader.layer2, "pulseVariation", 0, 1)
        .name("Pulse Variation")
        .onChange(updateActivePreset);
    effect1Layer2Folder
        .add(config.presets.preset1.shader.layer2, "patternScale", 0.5, 10)
        .name("Pattern Scale")
        .onChange(updateActivePreset);
    effect1Layer2Folder
        .add(config.presets.preset1.shader.layer2, "blendFactor", 0, 5)
        .name("Blend Factor")
        .onChange(updateActivePreset);
    effect1Layer2Folder
        .addColor(config.presets.preset1.shader.layer2, "emberColor")
        .name("Ember Color")
        .onChange(updateActivePreset);
    effect1Layer2Folder
        .addColor(config.presets.preset1.shader.layer2, "glowColor")
        .name("Glow Color")
        .onChange(updateActivePreset);
    effect1Folder
        .add(config.presets.preset1.particles, "enabled")
        .name("Enable Particles")
        .onChange(updateActivePreset);

    // Preset 2 (Blue Only)
    const effect2Folder = presetsFolder.addFolder("Preset 2 (Blue Only)");
    const effect2MainFolder = effect2Folder.addFolder("Main Layer");
    effect2MainFolder
        .add(config.presets.preset2.shader, "intensity", 0, 3)
        .name("Intensity")
        .onChange(updateActivePreset);
    effect2MainFolder
        .add(config.presets.preset2.shader, "noiseScale", 0.1, 20)
        .name("Noise Scale")
        .onChange(updateActivePreset);
    effect2MainFolder
        .add(config.presets.preset2.shader, "pulseSpeed", 0, 2)
        .name("Pulse Speed")
        .onChange(updateActivePreset);
    effect2MainFolder
        .add(config.presets.preset2.shader, "pulseVariation", 0, 1)
        .name("Pulse Variation")
        .onChange(updateActivePreset);
    effect2MainFolder
        .add(config.presets.preset2.shader, "patternScale", 0.5, 10)
        .name("Pattern Scale")
        .onChange(updateActivePreset);
    effect2MainFolder
        .add(config.presets.preset2.shader, "displacementAmount", -0.2, 0.2)
        .name("Displacement")
        .onChange(updateActivePreset);
    effect2MainFolder
        .addColor(config.presets.preset2.shader, "emberColor")
        .name("Ember Color")
        .onChange(updateActivePreset);
    effect2MainFolder
        .addColor(config.presets.preset2.shader, "glowColor")
        .name("Glow Color")
        .onChange(updateActivePreset);
    const effect2Layer2Folder = effect2Folder.addFolder("Second Layer");
    effect2Layer2Folder
        .add(config.presets.preset2.shader.layer2, "enabled")
        .name("Enable Layer 2")
        .onChange(updateActivePreset);
    effect2Layer2Folder
        .add(config.presets.preset2.shader.layer2, "intensity", 0, 3)
        .name("Intensity")
        .onChange(updateActivePreset);
    effect2Layer2Folder
        .add(config.presets.preset2.shader.layer2, "noiseScale", 0.1, 20)
        .name("Noise Scale")
        .onChange(updateActivePreset);
    effect2Layer2Folder
        .add(config.presets.preset2.shader.layer2, "pulseSpeed", 0, 2)
        .name("Pulse Speed")
        .onChange(updateActivePreset);
    effect2Layer2Folder
        .add(config.presets.preset2.shader.layer2, "pulseVariation", 0, 1)
        .name("Pulse Variation")
        .onChange(updateActivePreset);
    effect2Layer2Folder
        .add(config.presets.preset2.shader.layer2, "patternScale", 0.5, 10)
        .name("Pattern Scale")
        .onChange(updateActivePreset);
    effect2Layer2Folder
        .add(config.presets.preset2.shader.layer2, "blendFactor", 0, 5)
        .name("Blend Factor")
        .onChange(updateActivePreset);
    effect2Layer2Folder
        .addColor(config.presets.preset2.shader.layer2, "emberColor")
        .name("Ember Color")
        .onChange(updateActivePreset);
    effect2Layer2Folder
        .addColor(config.presets.preset2.shader.layer2, "glowColor")
        .name("Glow Color")
        .onChange(updateActivePreset);
    effect2Folder
        .add(config.presets.preset2.particles, "enabled")
        .name("Enable Particles")
        .onChange(updateActivePreset);

    presetsFolder
        .add(
            {
                saveSettings: function ()
                {
                    if (currentActivePreset)
                    {
                        console.log(
                            "Saving current settings to " +
                            currentActivePreset +
                            " preset"
                        );
                        Object.assign(
                            config.presets[currentActivePreset].shader,
                            JSON.parse(JSON.stringify(config.shader))
                        );
                        Object.assign(config.presets[currentActivePreset].particles, {
                            enabled: config.particles.enabled
                        });
                    } else
                    {
                        console.log("No active preset to save to");
                    }
                }
            },
            "saveSettings"
        )
        .name("Save Current to Active Preset");

    const shaderFolder = gui.addFolder("Current Shader");
    shaderFolder
        .add(config.shader, "enabled")
        .name("Enable Shader")
        .onChange(function (value)
        {
            if (orbOverlay)
            {
                orbOverlay.visible = value;
            }
        });
    shaderFolder.add(config.shader, "intensity", 0, 3).name("Intensity");
    shaderFolder.add(config.shader, "noiseScale", 0.1, 20).name("Noise Scale");
    shaderFolder.add(config.shader, "pulseSpeed", 0, 2).name("Pulse Speed");
    shaderFolder
        .add(config.shader, "pulseVariation", 0, 1)
        .name("Pulse Variation");
    shaderFolder
        .add(config.shader, "patternScale", 0.5, 10)
        .name("Pattern Scale");
    shaderFolder
        .add(config.shader, "displacementAmount", -0.2, 0.2)
        .name("Displacement")
        .onChange(function (value)
        {
            if (orbOverlay && orbOverlay.material.uniforms)
            {
                orbOverlay.material.uniforms.displacementAmount.value = value;
            }
        });
    const emberColorCtrl = shaderFolder
        .addColor(config.shader, "emberColor")
        .name("Ember Color");
    emberColorCtrl.onChange(function (value)
    {
        if (orbOverlay && orbOverlay.material.uniforms)
        {
            orbOverlay.material.uniforms.emberColor.value.setRGB(
                value[0] / 255,
                value[1] / 255,
                value[2] / 255
            );
        }
    });
    const glowColorCtrl = shaderFolder
        .addColor(config.shader, "glowColor")
        .name("Glow Color");
    glowColorCtrl.onChange(function (value)
    {
        if (orbOverlay && orbOverlay.material.uniforms)
        {
            orbOverlay.material.uniforms.glowColor.value.setRGB(
                value[0] / 255,
                value[1] / 255,
                value[2] / 255
            );
        }
    });
    const layer2Folder = shaderFolder.addFolder("Second Layer");
    layer2Folder
        .add(config.shader.layer2, "enabled")
        .name("Enable Layer 2")
        .onChange(function (value)
        {
            if (orbOverlay && orbOverlay.material.uniforms)
            {
                orbOverlay.material.uniforms.layer2Enabled.value = value;
            }
        });
    layer2Folder
        .add(config.shader.layer2, "intensity", 0, 3)
        .name("Layer 2 Intensity")
        .onChange(function (value)
        {
            if (orbOverlay && orbOverlay.material.uniforms)
            {
                orbOverlay.material.uniforms.layer2Intensity.value = value;
            }
        });
    layer2Folder
        .add(config.shader.layer2, "noiseScale", 0.1, 20)
        .name("Layer 2 Noise Scale")
        .onChange(function (value)
        {
            if (orbOverlay && orbOverlay.material.uniforms)
            {
                orbOverlay.material.uniforms.layer2NoiseScale.value = value;
            }
        });
    layer2Folder
        .add(config.shader.layer2, "pulseSpeed", 0, 2)
        .name("Layer 2 Pulse Speed")
        .onChange(function (value)
        {
            if (orbOverlay && orbOverlay.material.uniforms)
            {
                orbOverlay.material.uniforms.layer2PulseSpeed.value = value;
            }
        });
    layer2Folder
        .add(config.shader.layer2, "pulseVariation", 0, 1)
        .name("Layer 2 Pulse Var")
        .onChange(function (value)
        {
            if (orbOverlay && orbOverlay.material.uniforms)
            {
                orbOverlay.material.uniforms.layer2PulseVariation.value = value;
            }
        });
    layer2Folder
        .add(config.shader.layer2, "patternScale", 0.5, 10)
        .name("Layer 2 Pattern Scale")
        .onChange(function (value)
        {
            if (orbOverlay && orbOverlay.material.uniforms)
            {
                orbOverlay.material.uniforms.layer2PatternScale.value = value;
            }
        });
    layer2Folder
        .add(config.shader.layer2, "blendFactor", 0, 5)
        .name("Layer 2 Blend Factor")
        .onChange(function (value)
        {
            if (orbOverlay && orbOverlay.material.uniforms)
            {
                orbOverlay.material.uniforms.layer2BlendFactor.value = value;
            }
        });
    const layer2EmberColorCtrl = layer2Folder
        .addColor(config.shader.layer2, "emberColor")
        .name("Layer 2 Ember Color");
    layer2EmberColorCtrl.onChange(function (value)
    {
        if (orbOverlay && orbOverlay.material.uniforms)
        {
            orbOverlay.material.uniforms.layer2EmberColor.value.setRGB(
                value[0] / 255,
                value[1] / 255,
                value[2] / 255
            );
        }
    });
    const layer2GlowColorCtrl = layer2Folder
        .addColor(config.shader.layer2, "glowColor")
        .name("Layer 2 Glow Color");
    layer2GlowColorCtrl.onChange(function (value)
    {
        if (orbOverlay && orbOverlay.material.uniforms)
        {
            orbOverlay.material.uniforms.layer2GlowColor.value.setRGB(
                value[0] / 255,
                value[1] / 255,
                value[2] / 255
            );
        }
    });
}

function addAnimationControls()
{
    if (!gui) return;
    const animationFolder = gui.addFolder("Animation");
    animationFolder
        .add(config.animation, "enabled")
        .name("Animation Enabled")
        .onChange((value) =>
        {
            animationActions.forEach((action) =>
            {
                if (value)
                {
                    action.play();
                } else
                {
                    action.stop();
                }
            });
        });
    animationFolder
        .add(config.animation, "speed", 0, 3)
        .name("Animation Speed")
        .onChange((value) =>
        {
            if (mixer)
            {
                animationActions.forEach((action) =>
                {
                    action.timeScale = value;
                });
            }
        });
}
