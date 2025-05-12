(function (window, THREE)
{
    // Internal reference to the particle system
    let particleSystem = null;

    // Create a glowing particle texture
    function createParticleTexture()
    {
        const canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 128;
        const context = canvas.getContext('2d');

        // Create radial gradient for halo effect
        const gradient = context.createRadialGradient(
            canvas.width / 2, canvas.height / 2, 0,
            canvas.width / 2, canvas.height / 2, canvas.width / 2
        );
        gradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
        gradient.addColorStop(0.2, 'rgba(255, 220, 150, 0.9)');
        gradient.addColorStop(0.5, 'rgba(255, 180, 100, 0.6)');
        gradient.addColorStop(0.8, 'rgba(255, 100, 50, 0.3)');
        gradient.addColorStop(1, 'rgba(255, 50, 0, 0)');
        context.fillStyle = gradient;
        context.fillRect(0, 0, canvas.width, canvas.height);

        const texture = new THREE.Texture(canvas);
        texture.needsUpdate = true;
        return texture;
    }

    // Create the particle system and add it to the scene.
    // Expects: scene and config object (with config.particles properties)
    function createParticles(scene, config)
    {
        // Temporary geometry for sampling if needed
        const samplingMesh = new THREE.SphereGeometry(1, 64, 64);
        const particleCount = config.particles.count;

        // Create particle texture and material for glow effect
        const particleTexture = createParticleTexture();
        const particleMaterial = new THREE.PointsMaterial({
            size: config.particles.size,
            map: particleTexture,
            blending: THREE.AdditiveBlending,
            transparent: true,
            depthWrite: false,
            vertexColors: true
        });

        // Create geometry with positions and colors
        const particleGeometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);

        const particles = [];
        const baseColor = new THREE.Color(
            config.particles.color[0] / 255,
            config.particles.color[1] / 255,
            config.particles.color[2] / 255
        );

        // Initialize particle data
        for (let i = 0; i < particleCount; i++)
        {
            const particle = {
                position: new THREE.Vector3(),
                velocity: new THREE.Vector3(),
                originalVelocity: new THREE.Vector3(),
                age: config.particles.lifetime, // Start at lifetime so it will respawn immediately
                color: baseColor.clone(),
                initialColor: baseColor.clone(),
                alpha: 0,
                size: 1.0,
                initialSize: 1.0,
                active: false
            };
            particles.push(particle);

            // Set initial invisible positions/colors
            positions[i * 3] = 0;
            positions[i * 3 + 1] = 0;
            positions[i * 3 + 2] = -1000;
            colors[i * 3] = 0;
            colors[i * 3 + 1] = 0;
            colors[i * 3 + 2] = 0;
        }

        particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        particleSystem = new THREE.Points(particleGeometry, particleMaterial);
        particleSystem.userData = {
            particles: particles,
            meshToSample: samplingMesh,
            spawnTimer: 0,
            nextSpawnTime: 0,
            particlesSpawned: 0
        };

        scene.add(particleSystem);
        return particleSystem;
    }

    // Spawns or respawns a single particle.
    function respawnParticle(particle, meshToSample, config)
    {
        particle.age = 0;
        particle.active = true;
        particle.initialColor.set(
            config.particles.color[0] / 255,
            config.particles.color[1] / 255,
            config.particles.color[2] / 255
        );
        particle.initialColor.multiplyScalar(1.5);
        particle.color.copy(particle.initialColor);
        particle.initialSize = config.particles.size;
        particle.size = particle.initialSize;

        if (config.particles.spawnInRing)
        {
            const radius = config.particles.ringRadius + (Math.random() * 0.125 - 0.06);
            const equatorWidth = config.particles.equatorBandWidth;
            const angle = Math.random() * Math.PI * 2;
            const heightVariation = (Math.random() * 2 - 1) * equatorWidth;
            const rotatedAngle = angle + config.particles.ringRotation;

            particle.position.set(
                radius * Math.cos(rotatedAngle),
                heightVariation,
                radius * Math.sin(rotatedAngle)
            );
            const tangentialSpeed = 0.005 + Math.random() * 0.01;
            particle.velocity.set(
                -Math.sin(rotatedAngle) * tangentialSpeed,
                0,
                Math.cos(rotatedAngle) * tangentialSpeed
            );
            particle.velocity.x += (Math.random() * 2 - 1) * 0.005;
            particle.velocity.y += (Math.random() * 2 - 1) * 0.005;
            particle.velocity.z += (Math.random() * 2 - 1) * 0.005;
        } else
        {
            const vertexCount = meshToSample.attributes.position.array.length / 3;
            const vertexIndex = Math.floor(Math.random() * vertexCount) * 3;
            particle.position.set(
                meshToSample.attributes.position.array[vertexIndex],
                meshToSample.attributes.position.array[vertexIndex + 1],
                meshToSample.attributes.position.array[vertexIndex + 2]
            );
            if (meshToSample.attributes.normal)
            {
                particle.velocity.set(
                    meshToSample.attributes.normal.array[vertexIndex] * (0.005 + Math.random() * 0.01),
                    meshToSample.attributes.normal.array[vertexIndex + 1] * (0.005 + Math.random() * 0.01),
                    meshToSample.attributes.normal.array[vertexIndex + 2] * (0.005 + Math.random() * 0.01)
                );
            } else
            {
                const len = particle.position.length();
                particle.velocity.set(
                    (particle.position.x / len) * (0.005 + Math.random() * 0.01),
                    (particle.position.y / len) * (0.005 + Math.random() * 0.01),
                    (particle.position.z / len) * (0.005 + Math.random() * 0.01)
                );
            }
        }
        particle.originalVelocity.copy(particle.velocity);
    }

    // Update the emitter to use a new mesh for particle spawning.
    function updateParticleEmitter(newMesh, config)
    {
        if (!particleSystem) return;
        const particles = particleSystem.userData.particles;
        particleSystem.userData.meshToSample = newMesh.geometry;
        for (let i = 0; i < particles.length; i++)
        {
            particles[i].active = false;
            particles[i].age = config.particles.lifetime;
        }
        particleSystem.userData.spawnTimer = 0;
        particleSystem.userData.nextSpawnTime = 0;
        particleSystem.userData.particlesSpawned = 0;
    }

    // Reset all particles (e.g. when changing presets)
    function resetParticleSystem(config)
    {
        if (!particleSystem) return;
        const particles = particleSystem.userData.particles;
        const positions = particleSystem.geometry.attributes.position.array;
        const colors = particleSystem.geometry.attributes.color.array;
        for (let i = 0; i < particles.length; i++)
        {
            particles[i].active = false;
            particles[i].age = 0;
            positions[i * 3] = 0;
            positions[i * 3 + 1] = 0;
            positions[i * 3 + 2] = -1000;
            colors[i * 3] = 0;
            colors[i * 3 + 1] = 0;
            colors[i * 3 + 2] = 0;
        }
        particleSystem.geometry.attributes.position.needsUpdate = true;
        particleSystem.geometry.attributes.color.needsUpdate = true;
        particleSystem.userData.spawnTimer = 0;
        particleSystem.userData.nextSpawnTime = 0;
        particleSystem.userData.particlesSpawned = 0;
    }

    // Update particle colors (e.g. when changing color settings)
    function updateParticleColors(config)
    {
        if (!particleSystem) return;
        const particles = particleSystem.userData.particles;
        const colors = particleSystem.geometry.attributes.color.array;
        const baseColor = new THREE.Color(
            config.particles.color[0] / 255,
            config.particles.color[1] / 255,
            config.particles.color[2] / 255
        );
        for (let i = 0; i < particles.length; i++)
        {
            particles[i].initialColor.copy(baseColor);
            particles[i].initialColor.multiplyScalar(1.5);
            particles[i].initialColor.r *= (0.9 + Math.random() * 0.2);
            particles[i].initialColor.g *= (0.9 + Math.random() * 0.2);
            particles[i].initialColor.b *= (0.9 + Math.random() * 0.2);
            if (particles[i].active && particles[i].age / config.particles.lifetime < 0.2)
            {
                particles[i].color.copy(particles[i].initialColor);
            }
            if (particles[i].active)
            {
                const alphaMultiplier = particles[i].alpha;
                colors[i * 3] = particles[i].color.r * alphaMultiplier;
                colors[i * 3 + 1] = particles[i].color.g * alphaMultiplier;
                colors[i * 3 + 2] = particles[i].color.b * alphaMultiplier;
            }
        }
        particleSystem.geometry.attributes.color.needsUpdate = true;
    }

    // Update the size of the particle points.
    function updateParticleSize(value)
    {
        if (particleSystem)
        {
            particleSystem.material.size = value;
        }
    }

    // Recreate the particle system with a new count.
    function updateParticleCount(scene, config, newCount)
    {
        config.particles.count = newCount;
        if (particleSystem)
        {
            scene.remove(particleSystem);
            createParticles(scene, config);
        }
    }

    // Update particles on each animation frame.
    // Expects: delta (time elapsed), clock (with elapsedTime), and config
    function updateParticles(delta, clock, config)
    {
        if (!particleSystem) return;
        const particles = particleSystem.userData.particles;
        const positions = particleSystem.geometry.attributes.position.array;
        const colors = particleSystem.geometry.attributes.color.array;
        const time = clock.elapsedTime;
        const meshToSample = particleSystem.userData.meshToSample;

        particleSystem.userData.spawnTimer += delta;
        if (
            config.particles.spawnOverTime &&
            particleSystem.userData.particlesSpawned < particles.length
        )
        {
            if (particleSystem.userData.spawnTimer >= particleSystem.userData.nextSpawnTime)
            {
                const batchSize = Math.min(
                    config.particles.spawnBatchSize,
                    particles.length - particleSystem.userData.particlesSpawned
                );
                for (let i = 0; i < batchSize; i++)
                {
                    const particleIndex = particleSystem.userData.particlesSpawned;
                    if (particleIndex < particles.length)
                    {
                        respawnParticle(particles[particleIndex], meshToSample, config);
                        particleSystem.userData.particlesSpawned++;
                    }
                }
                const totalBatches = Math.ceil(particles.length / config.particles.spawnBatchSize);
                const timePerBatch = config.particles.spawnPeriod / totalBatches;
                particleSystem.userData.nextSpawnTime += timePerBatch;
            }
        }

        const noiseVec = new THREE.Vector3();
        const targetRedColor = new THREE.Color(1, 0, 0);

        for (let i = 0; i < particles.length; i++)
        {
            const particle = particles[i];
            if (!particle.active)
            {
                positions[i * 3] = 0;
                positions[i * 3 + 1] = 0;
                positions[i * 3 + 2] = -1000;
                colors[i * 3] = 0;
                colors[i * 3 + 1] = 0;
                colors[i * 3 + 2] = 0;
                continue;
            }

            particle.age += delta;
            if (particle.age >= config.particles.lifetime)
            {
                if (config.particles.spawnOverTime)
                {
                    particle.active = false;
                    positions[i * 3] = 0;
                    positions[i * 3 + 1] = 0;
                    positions[i * 3 + 2] = -1000;
                    colors[i * 3] = 0;
                    colors[i * 3 + 1] = 0;
                    colors[i * 3 + 2] = 0;
                    continue;
                } else
                {
                    respawnParticle(particle, meshToSample, config);
                }
            } else
            {
                const lifecycle = particle.age / config.particles.lifetime;
                if (lifecycle < 0.05)
                {
                    particle.alpha = lifecycle * 10;
                } else
                {
                    particle.alpha = 1 - lifecycle;
                    const flicker = Math.sin(particle.age * 10 + i) * 0.1;
                    particle.alpha *= particle.alpha + flicker;
                }
                particle.size = particle.initialSize * (1 + (lifecycle * config.particles.growthFactor));
                particleSystem.material.size = particle.size;
                particle.color.copy(particle.initialColor).lerp(targetRedColor, lifecycle);

                const noiseScale = config.particles.meanderNoiseScale;
                const meanderStrength = config.particles.meanderStrength;
                noiseVec.set(
                    Math.sin(particle.position.x * noiseScale + time * 0.5) * 0.02 * meanderStrength,
                    Math.cos(particle.position.y * noiseScale + time * 0.4) * 0.02 * meanderStrength,
                    Math.sin(particle.position.z * noiseScale + time * 0.6) * 0.02 * meanderStrength
                );

                const currentVelocity = particle.originalVelocity.clone().add(noiseVec)
                    .multiplyScalar(config.particles.speed);
                particle.position.add(currentVelocity);

                const distanceFromCenter = particle.position.length();
                if (distanceFromCenter > 3.0)
                {
                    const pullBackForce = particle.position.clone().normalize()
                        .multiplyScalar(-0.005 * (distanceFromCenter - 3.0));
                    particle.position.add(pullBackForce);
                }
            }
            positions[i * 3] = particle.position.x;
            positions[i * 3 + 1] = particle.position.y;
            positions[i * 3 + 2] = particle.position.z;
            const alphaMultiplier = particle.alpha;
            colors[i * 3] = particle.color.r * alphaMultiplier;
            colors[i * 3 + 1] = particle.color.g * alphaMultiplier;
            colors[i * 3 + 2] = particle.color.b * alphaMultiplier;
        }
        particleSystem.geometry.attributes.position.needsUpdate = true;
        particleSystem.geometry.attributes.color.needsUpdate = true;
    }

    // Expose the particle system API as a global object.
    window.ParticleSys = {
        createParticles: createParticles,
        respawnParticle: respawnParticle,
        updateParticleEmitter: updateParticleEmitter,
        resetParticleSystem: resetParticleSystem,
        updateParticleColors: updateParticleColors,
        updateParticleSize: updateParticleSize,
        updateParticleCount: updateParticleCount,
        updateParticles: updateParticles
    };
})(window, THREE);
