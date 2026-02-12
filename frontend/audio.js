/* ======================================================
   AUDIO MANAGER — audio.js
   Handles BGM, SFX, and volume control using Web Audio API
   ====================================================== */

class AudioManager {
    constructor() {
        this.enabled = true;
        this.bgmVolume = 0.3;
        this.sfxVolume = 0.6;
        this.currentBGM = null;
        this.bgmAudio = null;
        this.loaded = {};
        this.muted = false;

        // Audio file mapping
        this.tracks = {
            // BGM
            bgm_lobby: 'assets/dark_theme_1.mp3',
            bgm_battle: 'assets/duel_theme_1.mp3',
            bgm_battle_alt: 'assets/duel_theme_2.mp3',
            bgm_dark: 'assets/dark_theme_2.mp3',
            // SFX
            sfx_cast_attack: 'assets/cast_attack.mp3',
            sfx_cast_curse: 'assets/cast_curse.mp3',
            sfx_cast_defend: 'assets/cast_defend.mp3',
            sfx_hit: 'assets/hit_damage.mp3',
            sfx_victory: 'assets/victory_string_1.mp3',
            sfx_victory_alt: 'assets/victory_string_2.mp3',
            sfx_defeat: 'assets/defeat_string_1.mp3',
            sfx_defeat_alt: 'assets/defeat_string_2.mp3',
        };

        // Preload all SFX
        this._preloadAll();
    }

    _preloadAll() {
        Object.entries(this.tracks).forEach(([key, src]) => {
            if (key.startsWith('sfx_')) {
                const audio = new Audio(src);
                audio.preload = 'auto';
                audio.volume = this.sfxVolume;
                this.loaded[key] = audio;
            }
        });
    }

    // ── BGM ──

    playBGM(trackKey, loop = true) {
        if (this.muted) return;
        const src = this.tracks[trackKey];
        if (!src) return;

        // Don't restart same track
        if (this.currentBGM === trackKey && this.bgmAudio && !this.bgmAudio.paused) return;

        this.stopBGM();

        this.bgmAudio = new Audio(src);
        this.bgmAudio.loop = loop;
        this.bgmAudio.volume = this.bgmVolume;
        this.currentBGM = trackKey;

        // Fade in
        this.bgmAudio.volume = 0;
        this.bgmAudio.play().catch(() => {});
        this._fadeIn(this.bgmAudio, this.bgmVolume, 1500);
    }

    stopBGM(fadeMs = 800) {
        if (this.bgmAudio) {
            const audio = this.bgmAudio;
            this._fadeOut(audio, fadeMs, () => {
                audio.pause();
                audio.currentTime = 0;
            });
            this.bgmAudio = null;
            this.currentBGM = null;
        }
    }

    // ── SFX ──

    playSFX(trackKey) {
        if (this.muted) return;
        const cached = this.loaded[trackKey];
        if (cached) {
            // Clone to allow overlapping playback
            const clone = cached.cloneNode();
            clone.volume = this.sfxVolume;
            clone.play().catch(() => {});
            return;
        }
        // Fallback: load and play
        const src = this.tracks[trackKey];
        if (src) {
            const audio = new Audio(src);
            audio.volume = this.sfxVolume;
            audio.play().catch(() => {});
        }
    }

    /** Play the appropriate SFX for a spell cast */
    playSpellSFX(spellName) {
        if (!spellName) return;
        const s = spellName.toLowerCase();
        if (s.includes('crucio') || s.includes('avada')) {
            this.playSFX('sfx_cast_curse');
        } else if (s.includes('protego')) {
            this.playSFX('sfx_cast_defend');
        } else {
            this.playSFX('sfx_cast_attack');
        }
    }

    /** Play hit SFX */
    playHitSFX() {
        this.playSFX('sfx_hit');
    }

    /** Play victory/defeat sting */
    playResultSFX(isVictory) {
        this.stopBGM(400);
        setTimeout(() => {
            if (isVictory) {
                this.playSFX(Math.random() > 0.5 ? 'sfx_victory' : 'sfx_victory_alt');
            } else {
                this.playSFX(Math.random() > 0.5 ? 'sfx_defeat' : 'sfx_defeat_alt');
            }
        }, 500);
    }

    // ── Volume Controls ──

    setBGMVolume(vol) {
        this.bgmVolume = Math.max(0, Math.min(1, vol));
        if (this.bgmAudio) this.bgmAudio.volume = this.bgmVolume;
    }

    setSFXVolume(vol) {
        this.sfxVolume = Math.max(0, Math.min(1, vol));
    }

    toggleMute() {
        this.muted = !this.muted;
        if (this.muted) {
            this.stopBGM(200);
        }
        return this.muted;
    }

    // ── Fade Helpers ──

    _fadeIn(audio, targetVol, durationMs) {
        const steps = 20;
        const stepTime = durationMs / steps;
        const volStep = targetVol / steps;
        let current = 0;
        const interval = setInterval(() => {
            current += volStep;
            if (current >= targetVol) {
                audio.volume = targetVol;
                clearInterval(interval);
            } else {
                audio.volume = current;
            }
        }, stepTime);
    }

    _fadeOut(audio, durationMs, callback) {
        if (!audio) { if (callback) callback(); return; }
        const steps = 15;
        const stepTime = durationMs / steps;
        const volStep = audio.volume / steps;
        const interval = setInterval(() => {
            const newVol = audio.volume - volStep;
            if (newVol <= 0.01) {
                audio.volume = 0;
                clearInterval(interval);
                if (callback) callback();
            } else {
                audio.volume = newVol;
            }
        }, stepTime);
    }
}

// Global instance
const audioManager = new AudioManager();
