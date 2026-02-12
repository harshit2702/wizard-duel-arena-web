/* ======================================================
   DYNAMIC BACKGROUNDS — backgrounds.js
   Parallax layers, fog, rain, lightning, ambient particles
   ====================================================== */

class DynamicBackground {
    constructor(canvasId) {
        this.canvas = null;
        this.ctx = null;
        this.canvasId = canvasId;
        this.layers = [];
        this.raindrops = [];
        this.fogClouds = [];
        this.ambientParticles = [];
        this.lightning = null;
        this.animFrame = null;
        this.weather = 'clear'; // 'clear', 'fog', 'rain', 'storm', 'embers'
        this.intensity = 1.0;
        this.time = 0;
        this.mouseX = 0;
        this.mouseY = 0;
        this.isActive = false;
    }

    init(canvasElement) {
        this.canvas = canvasElement || document.getElementById(this.canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.resize();
        window.addEventListener('resize', () => this.resize());
        document.addEventListener('mousemove', (e) => {
            this.mouseX = e.clientX;
            this.mouseY = e.clientY;
        });
        this.isActive = true;
    }

    resize() {
        if (!this.canvas) return;
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    /** Set weather mode for the arena */
    setWeather(type, intensity = 1.0) {
        this.weather = type;
        this.intensity = intensity;
        this.raindrops = [];
        this.fogClouds = [];
        this.ambientParticles = [];

        switch (type) {
            case 'fog':
                this._initFog();
                break;
            case 'rain':
                this._initRain();
                break;
            case 'storm':
                this._initRain();
                this._initFog();
                break;
            case 'embers':
                this._initEmbers();
                break;
            case 'snow':
                this._initSnow();
                break;
            case 'dark_mist':
                this._initDarkMist();
                break;
        }
    }

    /** Start the animation loop */
    start() {
        if (this.animFrame) return;
        this._animate();
    }

    /** Stop the animation loop */
    stop() {
        if (this.animFrame) {
            cancelAnimationFrame(this.animFrame);
            this.animFrame = null;
        }
        this.isActive = false;
    }

    // ─── ANIMATION LOOP ───

    _animate() {
        if (!this.ctx || !this.isActive) return;
        const { ctx, canvas } = this;
        this.time += 0.016; // ~60fps tick

        // Clear with very low alpha to create motion trails
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw parallax gradient layers
        this._drawParallaxBackground();

        // Weather effects
        switch (this.weather) {
            case 'fog':
                this._updateFog();
                this._drawFog();
                break;
            case 'rain':
                this._updateRain();
                this._drawRain();
                break;
            case 'storm':
                this._updateRain();
                this._drawRain();
                this._updateFog();
                this._drawFog();
                this._maybeFlashLightning();
                break;
            case 'embers':
                this._updateEmbers();
                this._drawEmbers();
                break;
            case 'snow':
                this._updateSnow();
                this._drawSnow();
                break;
            case 'dark_mist':
                this._updateDarkMist();
                this._drawDarkMist();
                break;
        }

        // Always draw ambient dust
        this._updateAmbientDust();
        this._drawAmbientDust();

        this.animFrame = requestAnimationFrame(() => this._animate());
    }

    // ─── PARALLAX BACKGROUND ───

    _drawParallaxBackground() {
        const { ctx, canvas, mouseX, mouseY, time } = this;
        const w = canvas.width;
        const h = canvas.height;

        // Parallax offset based on mouse position
        const px = (mouseX / w - 0.5) * 20;
        const py = (mouseY / h - 0.5) * 10;

        // Layer 1: Deep background nebula (slowest parallax)
        const grd1 = ctx.createRadialGradient(
            w * 0.3 + px * 0.2, h * 0.3 + py * 0.2, 0,
            w * 0.3 + px * 0.2, h * 0.3 + py * 0.2, w * 0.5
        );
        grd1.addColorStop(0, 'rgba(26, 16, 64, 0.15)');
        grd1.addColorStop(1, 'transparent');
        ctx.fillStyle = grd1;
        ctx.fillRect(0, 0, w, h);

        // Layer 2: Mid nebula (medium parallax)
        const grd2 = ctx.createRadialGradient(
            w * 0.7 + px * 0.5, h * 0.6 + py * 0.5, 0,
            w * 0.7 + px * 0.5, h * 0.6 + py * 0.5, w * 0.4
        );
        grd2.addColorStop(0, 'rgba(15, 26, 48, 0.12)');
        grd2.addColorStop(1, 'transparent');
        ctx.fillStyle = grd2;
        ctx.fillRect(0, 0, w, h);

        // Layer 3: Subtle pulsing glow at center (fastest parallax)
        const pulse = Math.sin(time * 0.5) * 0.03 + 0.05;
        const grd3 = ctx.createRadialGradient(
            w * 0.5 + px, h * 0.5 + py, 0,
            w * 0.5 + px, h * 0.5 + py, w * 0.3
        );
        grd3.addColorStop(0, `rgba(99, 102, 241, ${pulse})`);
        grd3.addColorStop(1, 'transparent');
        ctx.fillStyle = grd3;
        ctx.fillRect(0, 0, w, h);
    }

    // ─── FOG ───

    _initFog() {
        this.fogClouds = [];
        const count = Math.floor(8 * this.intensity);
        for (let i = 0; i < count; i++) {
            this.fogClouds.push({
                x: Math.random() * this.canvas.width,
                y: this.canvas.height * (0.3 + Math.random() * 0.5),
                w: 200 + Math.random() * 400,
                h: 60 + Math.random() * 120,
                speed: 0.2 + Math.random() * 0.5,
                opacity: 0.03 + Math.random() * 0.06,
                phase: Math.random() * Math.PI * 2,
            });
        }
    }

    _updateFog() {
        for (const cloud of this.fogClouds) {
            cloud.x += cloud.speed;
            if (cloud.x > this.canvas.width + cloud.w) {
                cloud.x = -cloud.w;
            }
            cloud.y += Math.sin(this.time + cloud.phase) * 0.3;
        }
    }

    _drawFog() {
        const { ctx } = this;
        for (const cloud of this.fogClouds) {
            const alpha = cloud.opacity * (0.8 + Math.sin(this.time * 0.3 + cloud.phase) * 0.2);
            const grad = ctx.createRadialGradient(
                cloud.x, cloud.y, 0,
                cloud.x, cloud.y, cloud.w * 0.5
            );
            grad.addColorStop(0, `rgba(180, 200, 220, ${alpha})`);
            grad.addColorStop(0.5, `rgba(150, 170, 200, ${alpha * 0.5})`);
            grad.addColorStop(1, 'transparent');
            ctx.fillStyle = grad;
            ctx.fillRect(cloud.x - cloud.w, cloud.y - cloud.h, cloud.w * 2, cloud.h * 2);
        }
    }

    // ─── RAIN ───

    _initRain() {
        this.raindrops = [];
        const count = Math.floor(200 * this.intensity);
        for (let i = 0; i < count; i++) {
            this.raindrops.push(this._newRaindrop());
        }
    }

    _newRaindrop() {
        return {
            x: Math.random() * (this.canvas.width + 200) - 100,
            y: Math.random() * -this.canvas.height,
            len: 10 + Math.random() * 20,
            speed: 8 + Math.random() * 12,
            wind: 2 + Math.random() * 3,
            opacity: 0.1 + Math.random() * 0.2,
        };
    }

    _updateRain() {
        for (const drop of this.raindrops) {
            drop.x += drop.wind;
            drop.y += drop.speed;
            if (drop.y > this.canvas.height) {
                Object.assign(drop, this._newRaindrop());
                drop.y = -drop.len;
            }
        }
    }

    _drawRain() {
        const { ctx } = this;
        ctx.strokeStyle = 'rgba(174, 194, 224, 0.3)';
        ctx.lineWidth = 1;
        for (const drop of this.raindrops) {
            ctx.globalAlpha = drop.opacity;
            ctx.beginPath();
            ctx.moveTo(drop.x, drop.y);
            ctx.lineTo(drop.x + drop.wind * 0.4, drop.y + drop.len);
            ctx.stroke();
        }
        ctx.globalAlpha = 1;
    }

    // ─── LIGHTNING ───

    _maybeFlashLightning() {
        if (Math.random() < 0.002 * this.intensity) {
            this.lightning = { timer: 3, intensity: 0.2 + Math.random() * 0.3 };
        }
        if (this.lightning) {
            const { ctx, canvas } = this;
            ctx.fillStyle = `rgba(200, 200, 255, ${this.lightning.intensity})`;
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            this.lightning.timer--;
            if (this.lightning.timer <= 0) {
                // Double flash
                if (Math.random() < 0.4 && this.lightning.intensity > 0.1) {
                    this.lightning = { timer: 2, intensity: this.lightning.intensity * 0.6 };
                } else {
                    this.lightning = null;
                }
            }
        }
    }

    // ─── EMBERS ───

    _initEmbers() {
        this.ambientParticles = [];
        const count = Math.floor(60 * this.intensity);
        for (let i = 0; i < count; i++) {
            this.ambientParticles.push(this._newEmber());
        }
    }

    _newEmber() {
        return {
            x: Math.random() * this.canvas.width,
            y: this.canvas.height + Math.random() * 50,
            vx: (Math.random() - 0.5) * 1.5,
            vy: -1 - Math.random() * 3,
            size: 1 + Math.random() * 3,
            life: 80 + Math.random() * 120,
            maxLife: 200,
            color: ['#ff4500', '#ff6600', '#ff8800', '#ffaa00', '#ffdd00'][Math.floor(Math.random() * 5)],
            flicker: Math.random() * Math.PI * 2,
        };
    }

    _updateEmbers() {
        for (let i = this.ambientParticles.length - 1; i >= 0; i--) {
            const p = this.ambientParticles[i];
            p.x += p.vx + Math.sin(this.time * 2 + p.flicker) * 0.5;
            p.y += p.vy;
            p.life--;
            if (p.life <= 0 || p.y < -20) {
                this.ambientParticles[i] = this._newEmber();
            }
        }
    }

    _drawEmbers() {
        const { ctx } = this;
        for (const p of this.ambientParticles) {
            const alpha = Math.min(1, p.life / 40) * (0.5 + Math.sin(this.time * 5 + p.flicker) * 0.3);
            ctx.save();
            ctx.globalAlpha = alpha;
            ctx.shadowColor = p.color;
            ctx.shadowBlur = p.size * 4;
            ctx.fillStyle = p.color;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        }
    }

    // ─── SNOW ───

    _initSnow() {
        this.ambientParticles = [];
        const count = Math.floor(100 * this.intensity);
        for (let i = 0; i < count; i++) {
            this.ambientParticles.push(this._newSnowflake());
        }
    }

    _newSnowflake() {
        return {
            x: Math.random() * this.canvas.width,
            y: -10 - Math.random() * this.canvas.height,
            size: 1 + Math.random() * 3,
            speed: 0.5 + Math.random() * 1.5,
            drift: Math.random() * Math.PI * 2,
            opacity: 0.3 + Math.random() * 0.5,
        };
    }

    _updateSnow() {
        for (const p of this.ambientParticles) {
            p.x += Math.sin(this.time + p.drift) * 0.5;
            p.y += p.speed;
            if (p.y > this.canvas.height + 10) {
                Object.assign(p, this._newSnowflake());
                p.y = -10;
            }
        }
    }

    _drawSnow() {
        const { ctx } = this;
        for (const p of this.ambientParticles) {
            ctx.save();
            ctx.globalAlpha = p.opacity;
            ctx.fillStyle = '#e8f0ff';
            ctx.shadowColor = '#aaccff';
            ctx.shadowBlur = p.size * 2;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        }
    }

    // ─── DARK MIST ───

    _initDarkMist() {
        this.fogClouds = [];
        const count = Math.floor(12 * this.intensity);
        for (let i = 0; i < count; i++) {
            this.fogClouds.push({
                x: Math.random() * this.canvas.width,
                y: this.canvas.height * (0.4 + Math.random() * 0.5),
                w: 300 + Math.random() * 500,
                h: 80 + Math.random() * 150,
                speed: 0.1 + Math.random() * 0.3,
                opacity: 0.04 + Math.random() * 0.07,
                phase: Math.random() * Math.PI * 2,
            });
        }
    }

    _updateDarkMist() {
        for (const cloud of this.fogClouds) {
            cloud.x += cloud.speed;
            if (cloud.x > this.canvas.width + cloud.w) {
                cloud.x = -cloud.w;
            }
            cloud.y += Math.sin(this.time * 0.5 + cloud.phase) * 0.4;
        }
    }

    _drawDarkMist() {
        const { ctx } = this;
        for (const cloud of this.fogClouds) {
            const alpha = cloud.opacity * (0.7 + Math.sin(this.time * 0.2 + cloud.phase) * 0.3);
            const grad = ctx.createRadialGradient(
                cloud.x, cloud.y, 0,
                cloud.x, cloud.y, cloud.w * 0.5
            );
            grad.addColorStop(0, `rgba(30, 0, 50, ${alpha})`);
            grad.addColorStop(0.4, `rgba(20, 0, 30, ${alpha * 0.6})`);
            grad.addColorStop(1, 'transparent');
            ctx.fillStyle = grad;
            ctx.fillRect(cloud.x - cloud.w, cloud.y - cloud.h, cloud.w * 2, cloud.h * 2);
        }
    }

    // ─── AMBIENT DUST (always active) ───

    _ambientDust = [];

    _updateAmbientDust() {
        // Keep a small number of floating particles
        while (this._ambientDust.length < 25) {
            this._ambientDust.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                size: 0.5 + Math.random() * 1.5,
                speedX: (Math.random() - 0.5) * 0.3,
                speedY: (Math.random() - 0.5) * 0.3,
                opacity: 0.1 + Math.random() * 0.2,
                phase: Math.random() * Math.PI * 2,
            });
        }
        for (let i = this._ambientDust.length - 1; i >= 0; i--) {
            const d = this._ambientDust[i];
            d.x += d.speedX + Math.sin(this.time * 0.5 + d.phase) * 0.2;
            d.y += d.speedY + Math.cos(this.time * 0.3 + d.phase) * 0.15;
            if (d.x < -10 || d.x > this.canvas.width + 10 ||
                d.y < -10 || d.y > this.canvas.height + 10) {
                this._ambientDust.splice(i, 1);
            }
        }
    }

    _drawAmbientDust() {
        const { ctx } = this;
        for (const d of this._ambientDust) {
            ctx.save();
            ctx.globalAlpha = d.opacity * (0.5 + Math.sin(this.time + d.phase) * 0.5);
            ctx.fillStyle = '#8888cc';
            ctx.beginPath();
            ctx.arc(d.x, d.y, d.size, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        }
    }

    /** Random weather based on match context */
    randomizeWeather() {
        const weathers = ['clear', 'fog', 'rain', 'storm', 'embers', 'snow', 'dark_mist'];
        const weights =  [   2,     3,     3,      1,       2,       2,       2       ];
        const total = weights.reduce((a, b) => a + b, 0);
        let r = Math.random() * total;
        for (let i = 0; i < weathers.length; i++) {
            r -= weights[i];
            if (r <= 0) {
                this.setWeather(weathers[i], 0.5 + Math.random() * 0.5);
                return weathers[i];
            }
        }
        this.setWeather('fog');
        return 'fog';
    }
}

// Global instance
const dynamicBG = new DynamicBackground('bg-canvas');
