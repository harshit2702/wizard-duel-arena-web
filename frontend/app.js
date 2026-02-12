/* ======================================================
   WIZARD DUEL FRONTEND — app.js v12
   Full rewrite: particles, all 16 spells, floating dmg,
   figures, fog-of-war, win screen, model selection,
   audio, tutorial, draft, dynamic backgrounds
   ====================================================== */

const API = '/api';

// ─── GAME MODES ───
let currentMode = 'classic'; // 'classic', 'tutorial', 'grimoire'
let draftedDeck = null; // If playing with a drafted deck

// ─── ALL 16 SPELLS (matching duel_engine SPELL_BOOK) ───
const SPELLS = [
    // Defense
    { idx:0,  name:"Protego",           type:"Defense", cost:0,   icon:"🛡️", color:"#22c55e",
      desc:"Block incoming attack. Reflects Control spells." },
    { idx:1,  name:"Protego Maximus",   type:"Defense", cost:30,  icon:"🛡️", color:"#15803d",
      desc:"Absorbs damage as Focus instead of HP." },
    // Info
    { idx:2,  name:"Revelio",           type:"Info",    cost:10,  icon:"👁️", color:"#3b82f6",
      desc:"Reveals all enemy stats for 3 rounds." },
    { idx:3,  name:"Legilimens",        type:"Info",    cost:20,  icon:"🧠", color:"#2563eb",
      desc:"Mind-read target: shows HP/Posture for 5 rounds." },
    // Control
    { idx:4,  name:"Levioso",           type:"Control", cost:15,  icon:"🪶", color:"#a855f7",
      desc:"Levitates target (Airborne). 20 posture dmg." },
    { idx:5,  name:"Glacius",           type:"Control", cost:20,  icon:"❄️", color:"#06b6d4",
      desc:"Freezes target (skip action) + Brittle (2x next hit)." },
    { idx:6,  name:"Arresto Momentum",  type:"Control", cost:25,  icon:"⏸️", color:"#8b5cf6",
      desc:"Slows target (no posture regen)." },
    // Force
    { idx:7,  name:"Accio",             type:"Force",   cost:15,  icon:"🧲", color:"#eab308",
      desc:"Pulls target to CLOSE range. 15 pos, 5 HP dmg." },
    { idx:8,  name:"Depulso",           type:"Force",   cost:15,  icon:"⚡", color:"#ca8a04",
      desc:"Pushes target back. 25 pos, 5 HP dmg." },
    { idx:9,  name:"Descendo",          type:"Force",   cost:20,  icon:"⬇️", color:"#854d0e",
      desc:"Slam. Bonus on Airborne. Save airborne ally." },
    // Damage
    { idx:10, name:"Basic Cast",        type:"Damage",  cost:0,   icon:"✨", color:"#94a3b8",
      desc:"Weak attack (5 HP, 5 pos). +5 posture regen." },
    { idx:11, name:"Incendio",          type:"Damage",  cost:25,  icon:"🔥", color:"#dc2626",
      desc:"Fire. AoE 30 at CLOSE. 20 MID, 15 FAR." },
    { idx:12, name:"Confringo",         type:"Damage",  cost:30,  icon:"💥", color:"#b91c1c",
      desc:"Explosion. 25 HP + bonus +10 at FAR range." },
    { idx:13, name:"Diffindo",          type:"Damage",  cost:35,  icon:"🗡️", color:"#991b1b",
      desc:"High damage single-target: 45 HP." },
    // Curses
    { idx:14, name:"Crucio",            type:"Curse",   cost:50,  icon:"💀", color:"#7f1d1d",
      desc:"Stun 2 turns + Cursed Pain (DoT 15/10/7)." },
    { idx:15, name:"Avada Kedavra",     type:"Curse",   cost:100, icon:"☠️", color:"#020617",
      desc:"INSTANT KILL if posture broken. Else 40 HP. Chains to CursedPain targets." },
];

// ─── STATE ───
let sessionId = null;
let selectedSpellIdx = null;
let currentState = null;
let isProcessing = false;
let animating = false;
let pendingAnimations = [];
let animResolve = null;
let playerWizardId = null;   // the human-controlled wizard ID

// ─── SETTINGS STATE ───
let animSpeed = 1.0;         // multiplier: higher = slower
let clickMode = 'always';    // 'always' | 'next-turn' | 'never'
let clickNextTurnOnly = false; // when true, next turn uses click, then reverts
let showDistLabels = true;
let lastRoundEvents = [];    // store last round's animation events for replay
let isReplaying = false;     // true during replay (visual-only, no state change)

const DIST_NAMES = { 0: 'CLOSE', 1: 'MID', 2: 'FAR' };

// ─── PARTICLE SYSTEM ───
let canvas, ctx;
let particles = [];
let particleAnimFrame = null;

class Particle {
    constructor(x, y, opts = {}) {
        this.x = x;
        this.y = y;
        this.vx = opts.vx ?? (Math.random() - 0.5) * 3;
        this.vy = opts.vy ?? (Math.random() - 0.5) * 3;
        this.life = opts.life ?? 60;
        this.maxLife = this.life;
        this.size = opts.size ?? 3;
        this.color = opts.color ?? '#fff';
        this.glow = opts.glow ?? false;
        this.gravity = opts.gravity ?? 0;
        this.friction = opts.friction ?? 0.98;
        this.type = opts.type ?? 'default'; // fire, ice, electric, magic, dark
    }
    update() {
        this.vx *= this.friction;
        this.vy *= this.friction;
        this.vy += this.gravity;
        this.x += this.vx;
        this.y += this.vy;
        this.life--;
    }
    draw(ctx) {
        const alpha = Math.max(0, this.life / this.maxLife);
        ctx.save();
        ctx.globalAlpha = alpha;
        if (this.glow) {
            ctx.shadowColor = this.color;
            ctx.shadowBlur = this.size * 3;
        }
        ctx.fillStyle = this.color;
        if (this.type === 'fire') {
            // Flickering fire particle
            const s = this.size * (0.5 + alpha * 0.5);
            ctx.beginPath();
            ctx.arc(this.x, this.y, s, 0, Math.PI * 2);
            ctx.fill();
            // Inner bright core
            ctx.fillStyle = '#ffdd44';
            ctx.beginPath();
            ctx.arc(this.x, this.y, s * 0.4, 0, Math.PI * 2);
            ctx.fill();
        } else if (this.type === 'electric') {
            // Jagged electric arc dot
            ctx.strokeStyle = this.color;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(this.x, this.y);
            ctx.lineTo(this.x + (Math.random()-0.5)*8, this.y + (Math.random()-0.5)*8);
            ctx.stroke();
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size * 0.6, 0, Math.PI*2);
            ctx.fill();
        } else if (this.type === 'magic') {
            // Sparkle
            const s = this.size;
            ctx.beginPath();
            for (let i = 0; i < 4; i++) {
                const angle = (i / 4) * Math.PI * 2 + (this.maxLife - this.life) * 0.05;
                const ox = Math.cos(angle) * s;
                const oy = Math.sin(angle) * s;
                ctx.moveTo(this.x, this.y);
                ctx.lineTo(this.x + ox, this.y + oy);
            }
            ctx.strokeStyle = this.color;
            ctx.lineWidth = 1;
            ctx.stroke();
        } else if (this.type === 'dark') {
            // Dark curse swirl
            ctx.beginPath();
            const a = (this.maxLife - this.life)*0.1;
            ctx.arc(this.x + Math.sin(a)*4, this.y + Math.cos(a)*4, this.size, 0, Math.PI*2);
            ctx.fill();
        } else {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.restore();
    }
}

function initCanvas() {
    canvas = document.getElementById('particle-canvas');
    if (!canvas) return;
    ctx = canvas.getContext('2d');
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    animateParticles();
}

function resizeCanvas() {
    if (!canvas) return;
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}

function animateParticles() {
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let i = particles.length - 1; i >= 0; i--) {
        particles[i].update();
        particles[i].draw(ctx);
        if (particles[i].life <= 0) particles.splice(i, 1);
    }
    particleAnimFrame = requestAnimationFrame(animateParticles);
}

// ─── SPELL-SPECIFIC PARTICLE EMITTERS ───

function getCardCenter(wizId) {
    const el = document.querySelector(`.wizard-card[data-id="${wizId}"]`);
    if (!el) return { x: canvas.width / 2, y: canvas.height / 2 };
    const r = el.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
}

function emitBeam(from, to, color, count=30, type='default') {
    for (let i = 0; i < count; i++) {
        const t = Math.random();
        const x = from.x + (to.x - from.x) * t;
        const y = from.y + (to.y - from.y) * t;
        particles.push(new Particle(x, y, {
            vx: (Math.random()-0.5)*2,
            vy: (Math.random()-0.5)*2,
            life: 30 + Math.random()*30,
            size: 2 + Math.random()*2,
            color, glow: true, type
        }));
    }
}

function emitFireSpell(from, to) {
    const colors = ['#ff4500','#ff6600','#ff8800','#ffaa00','#ffdd00'];
    for (let i = 0; i < 60; i++) {
        const t = Math.random();
        const x = from.x + (to.x - from.x) * t + (Math.random()-0.5)*20;
        const y = from.y + (to.y - from.y) * t + (Math.random()-0.5)*20;
        particles.push(new Particle(x, y, {
            vx: (to.x - from.x) * 0.02 + (Math.random()-0.5)*3,
            vy: -Math.random()*3 - 1,
            life: 40 + Math.random()*40,
            size: 2 + Math.random()*4,
            color: colors[Math.floor(Math.random()*colors.length)],
            glow: true, type: 'fire', gravity: -0.05
        }));
    }
    // Burst at target
    setTimeout(() => {
        for (let i = 0; i < 40; i++) {
            particles.push(new Particle(to.x, to.y, {
                vx: (Math.random()-0.5)*8,
                vy: (Math.random()-0.5)*8,
                life: 25 + Math.random()*25,
                size: 3 + Math.random()*5,
                color: colors[Math.floor(Math.random()*colors.length)],
                glow: true, type: 'fire', gravity: 0.1
            }));
        }
    }, 200);
}

function emitElectricSpell(from, to) {
    const colors = ['#00ccff','#66ddff','#ffffff','#88eeff'];
    // Electric arcs along path
    for (let i = 0; i < 50; i++) {
        const t = Math.random();
        const x = from.x + (to.x - from.x) * t;
        const y = from.y + (to.y - from.y) * t;
        particles.push(new Particle(x + (Math.random()-0.5)*30, y + (Math.random()-0.5)*30, {
            vx: (Math.random()-0.5)*6,
            vy: (Math.random()-0.5)*6,
            life: 15 + Math.random()*20,
            size: 1.5 + Math.random()*2,
            color: colors[Math.floor(Math.random()*colors.length)],
            glow: true, type: 'electric', friction: 0.9
        }));
    }
}

function emitMagicDust(from, to) {
    const colors = ['#a78bfa','#c084fc','#e879f9','#f0abfc','#ffffff'];
    for (let i = 0; i < 45; i++) {
        const t = Math.random();
        const x = from.x + (to.x - from.x)*t;
        const y = from.y + (to.y - from.y)*t;
        particles.push(new Particle(x, y, {
            vx: (Math.random()-0.5)*2,
            vy: (Math.random()-0.5)*2 - 0.5,
            life: 40 + Math.random()*40,
            size: 1 + Math.random()*3,
            color: colors[Math.floor(Math.random()*colors.length)],
            glow: true, type: 'magic'
        }));
    }
}

function emitDarkCurse(from, to) {
    const colors = ['#1a0a2e','#3b0764','#581c87','#000000','#22002e'];
    for (let i = 0; i < 70; i++) {
        const t = Math.random();
        const x = from.x + (to.x - from.x)*t;
        const y = from.y + (to.y - from.y)*t;
        particles.push(new Particle(x, y, {
            vx: (Math.random()-0.5)*4,
            vy: (Math.random()-0.5)*4,
            life: 50 + Math.random()*40,
            size: 2 + Math.random()*4,
            color: colors[Math.floor(Math.random()*colors.length)],
            glow: true, type: 'dark'
        }));
    }
    // Green death flash at target (AK only)
    setTimeout(() => {
        for (let i = 0; i < 50; i++) {
            particles.push(new Particle(to.x, to.y, {
                vx: (Math.random()-0.5)*10,
                vy: (Math.random()-0.5)*10,
                life: 20 + Math.random()*20,
                size: 3 + Math.random()*5,
                color: '#00ff00',
                glow: true, type: 'default', friction: 0.92
            }));
        }
    }, 250);
}

function emitIceSpell(from, to) {
    const colors = ['#00e5ff','#80deea','#b2ebf2','#e0f7fa','#ffffff'];
    for (let i = 0; i < 50; i++) {
        const t = Math.random();
        const x = from.x + (to.x - from.x)*t;
        const y = from.y + (to.y - from.y)*t;
        particles.push(new Particle(x, y, {
            vx: (Math.random()-0.5)*2,
            vy: (Math.random()-0.5)*2,
            life: 40 + Math.random()*30,
            size: 2 + Math.random()*3,
            color: colors[Math.floor(Math.random()*colors.length)],
            glow: true, type: 'magic', friction: 0.95
        }));
    }
    // Freeze burst at target
    setTimeout(() => {
        for (let i = 0; i < 30; i++) {
            particles.push(new Particle(to.x, to.y, {
                vx: (Math.random()-0.5)*5,
                vy: (Math.random()-0.5)*5,
                life: 30 + Math.random()*20,
                size: 3 + Math.random()*3,
                color: '#80deea',
                glow: true, type: 'default'
            }));
        }
    }, 200);
}

function emitShieldParticles(pos) {
    const colors = ['#22c55e','#4ade80','#86efac','#ffffff'];
    for (let i = 0; i < 40; i++) {
        const angle = (i / 40) * Math.PI * 2;
        const r = 25 + Math.random() * 15;
        particles.push(new Particle(pos.x + Math.cos(angle)*r, pos.y + Math.sin(angle)*r, {
            vx: Math.cos(angle)*0.5,
            vy: Math.sin(angle)*0.5,
            life: 40 + Math.random()*30,
            size: 1.5 + Math.random()*2,
            color: colors[Math.floor(Math.random()*colors.length)],
            glow: true, type: 'magic'
        }));
    }
}

function emitRevealParticles(pos) {
    const colors = ['#3b82f6','#60a5fa','#93c5fd','#ffffff'];
    for (let i = 0; i < 50; i++) {
        const angle = Math.random() * Math.PI * 2;
        const speed = 1 + Math.random()*3;
        particles.push(new Particle(pos.x, pos.y, {
            vx: Math.cos(angle)*speed,
            vy: Math.sin(angle)*speed,
            life: 50 + Math.random()*30,
            size: 1 + Math.random()*2.5,
            color: colors[Math.floor(Math.random()*colors.length)],
            glow: true, type: 'magic'
        }));
    }
}

function emitForceSpell(from, to) {
    const colors = ['#eab308','#facc15','#fde047','#fef08a'];
    for (let i = 0; i < 40; i++) {
        const t = Math.random();
        const x = from.x + (to.x - from.x)*t;
        const y = from.y + (to.y - from.y)*t;
        particles.push(new Particle(x, y, {
            vx: (to.x - from.x)*0.03 + (Math.random()-0.5)*3,
            vy: (to.y - from.y)*0.03 + (Math.random()-0.5)*3,
            life: 25 + Math.random()*20,
            size: 2 + Math.random()*3,
            color: colors[Math.floor(Math.random()*colors.length)],
            glow: true, type: 'default'
        }));
    }
}

function playSpellParticles(casterId, targetId, spellName) {
    const from = getCardCenter(casterId);
    const to = targetId != null ? getCardCenter(targetId) : from;
    const spell = spellName.toLowerCase();

    if (spell.includes('incendio') || spell.includes('confringo')) {
        emitFireSpell(from, to);
    } else if (spell.includes('depulso')) {
        emitElectricSpell(from, to);
    } else if (spell.includes('glacius')) {
        emitIceSpell(from, to);
    } else if (spell.includes('revelio')) {
        emitRevealParticles(from);
    } else if (spell.includes('legilimens')) {
        emitMagicDust(from, to);
    } else if (spell.includes('protego')) {
        emitShieldParticles(from);
        // Flash the caster card green during animation
        const casterCard = document.querySelector(`.wizard-card[data-id="${casterId}"]`);
        if (casterCard) {
            casterCard.classList.add('shield-glow');
        }
    } else if (spell.includes('levioso') || spell.includes('arresto')) {
        emitMagicDust(from, to);
    } else if (spell.includes('crucio') || spell.includes('avada')) {
        emitDarkCurse(from, to);
    } else if (spell.includes('diffindo')) {
        // Slash effect — fast red beam
        emitBeam(from, to, '#ef4444', 40, 'default');
    } else if (spell.includes('accio') || spell.includes('descendo')) {
        emitForceSpell(from, to);
    } else {
        // Basic Cast / default
        emitBeam(from, to, '#94a3b8', 20, 'default');
    }
}


// ─── FLOATING DAMAGE NUMBERS ───

function showFloatingNumber(wizId, text, type='damage') {
    const card = document.querySelector(`.wizard-card[data-id="${wizId}"]`);
    if (!card) return;
    const container = card.querySelector('.floating-numbers');
    if (!container) return;

    const el = document.createElement('div');
    el.className = `float-number ${type}`;
    el.textContent = text;
    // Random horizontal offset
    el.style.left = (30 + Math.random()*40) + '%';
    el.style.top = (10 + Math.random()*30) + '%';
    container.appendChild(el);

    el.addEventListener('animationend', () => el.remove());
}

function showDamageNumbers(ev) {
    if (!ev) return;
    const tid = ev.target_id;
    if (tid == null) return;

    const hpDmg = ev.hp_dmg || 0;
    const posDmg = ev.pos_dmg || 0;
    const effect = (ev.effect || '').toLowerCase();

    if (effect.includes('block') || effect.includes('miss')) {
        showFloatingNumber(tid, 'BLOCKED', 'shield');
        return;
    }
    if (effect.includes('absorb')) {
        showFloatingNumber(tid, `+${Math.round(hpDmg)} FOC`, 'shield');
        return;
    }
    if (effect.includes('reflect')) {
        showFloatingNumber(ev.caster_id, 'REFLECTED', 'shield');
        return;
    }

    if (hpDmg > 0) {
        const isCrit = hpDmg >= 40 || effect.includes('kill') || effect.includes('brittle');
        showFloatingNumber(tid, `-${Math.round(hpDmg)} HP`, isCrit ? 'critical' : 'damage');
    }
    if (posDmg > 0) {
        showFloatingNumber(tid, `-${Math.round(posDmg)} POS`, 'pos-damage');
    }
    if (effect.includes('saved') || effect.includes('save')) {
        showFloatingNumber(tid, 'SAVED!', 'heal');
    }

    // Flash the card
    const card = document.querySelector(`.wizard-card[data-id="${tid}"]`);
    if (card && (hpDmg > 0 || posDmg > 0)) {
        card.classList.add('hit-flash');
        setTimeout(() => card.classList.remove('hit-flash'), 500);
    }
}

// ─── ANIMATION SEQUENCER ───
// Plays spell animations sequentially, respects settings for click/speed

function shouldWaitForClick() {
    if (clickMode === 'always') return true;
    if (clickMode === 'never') return false;
    if (clickMode === 'next-turn') {
        // Only wait if the flag was set
        if (clickNextTurnOnly) {
            clickNextTurnOnly = false; // reset after use
            return true;
        }
        return false;
    }
    return true;
}

async function playAnimationSequence(events, visualOnly = false) {
    if (!events || events.length === 0) return;
    animating = true;

    const waitClick = shouldWaitForClick();

    for (const ev of events) {
        // Play particles
        playSpellParticles(ev.caster_id, ev.target_id, ev.spell || '');
        showDamageNumbers(ev);

        // ── AUDIO: spell cast SFX ──
        if (typeof audioManager !== 'undefined') {
            audioManager.playSpellSFX(ev.spell);
            // Hit SFX for damage events
            const hpDmg = ev.hp_dmg || 0;
            if (hpDmg > 0) {
                setTimeout(() => audioManager.playHitSFX(), 200);
            }
        }

        // Add log entry for this event (skip during replay to avoid duplicates)
        if (!visualOnly) {
            addLogEntry(ev);
        }

        if (waitClick) {
            // Wait for click to continue
            await waitForClick();
        } else {
            // Auto-advance with speed-controlled delay
            const delay = 800 * animSpeed;
            await new Promise(r => setTimeout(r, delay));
        }
    }
    animating = false;
}

async function replayLastRound() {
    if (!lastRoundEvents || lastRoundEvents.length === 0) return;
    if (animating || isReplaying) return;
    isReplaying = true;
    await playAnimationSequence(lastRoundEvents, true);
    isReplaying = false;
}

function waitForClick() {
    return new Promise(resolve => {
        // Show "click to continue" indicator
        const indicator = document.createElement('div');
        indicator.className = 'click-to-continue';
        indicator.textContent = 'Click to continue...';
        document.body.appendChild(indicator);

        // Overlay to catch clicks
        const overlay = document.createElement('div');
        overlay.className = 'spell-casting-overlay';
        document.body.appendChild(overlay);

        function onClick() {
            overlay.removeEventListener('click', onClick);
            overlay.remove();
            indicator.remove();
            resolve();
        }
        overlay.addEventListener('click', onClick);
    });
}

function addLogEntry(ev) {
    const logBox = document.getElementById('round-log');
    if (!logBox) return;

    const casterName = ev.caster_name || `#${ev.caster_id}`;
    const targetName = ev.target_name || (ev.target_id != null ? `#${ev.target_id}` : '');
    const spell = ev.spell || '?';
    const effect = ev.effect || '';

    let cls = '';
    const spellData = SPELLS.find(s => s.name === spell);
    if (spellData) {
        if (spellData.type === 'Damage' || spellData.type === 'Curse') cls = 'damage-log';
        else if (spellData.type === 'Info') cls = 'info-log';
        else if (spellData.type === 'Control') cls = 'control-log';
        else if (spellData.type === 'Defense') cls = 'heal-log';
    }

    const entry = document.createElement('div');
    entry.className = `log-entry ${cls}`;
    entry.textContent = `${casterName} > ${spell}${targetName ? ' on ' + targetName : ''}: ${effect}`;
    logBox.appendChild(entry);
    logBox.scrollTop = logBox.scrollHeight;
}


// ─── SPELL BAR ───

function renderSpellBar() {
    const bar = document.getElementById('spell-bar');
    if (!bar) return;
    bar.innerHTML = '';

    // If drafted deck exists, only show those spells
    const spellsToShow = draftedDeck && draftedDeck.length > 0
        ? draftedDeck
        : SPELLS;

    spellsToShow.forEach((spell, idx) => {
        const spellIdx = draftedDeck ? SPELLS.findIndex(s => s.name === spell.name) : idx;
        const btn = document.createElement('div');
        btn.className = 'spell-btn';
        btn.dataset.idx = spellIdx;
        btn.style.borderColor = spell.color;

        // Type indicator dot
        const typeDot = document.createElement('div');
        typeDot.className = 'spell-type-dot';
        typeDot.style.background = spell.color;
        btn.appendChild(typeDot);

        btn.innerHTML += `
            <div class="spell-icon">${spell.icon}</div>
            <div class="spell-name">${spell.name}</div>
            <div class="spell-cost">${spell.cost > 0 ? spell.cost + ' FOC' : 'FREE'}</div>
        `;

        btn.addEventListener('click', () => onSpellClick(spellIdx, btn));
        btn.addEventListener('mouseenter', (e) => showTooltip(spell, e));
        btn.addEventListener('mouseleave', hideTooltip);
        bar.appendChild(btn);
    });
}

function updateSpellBarState() {
    if (!currentState) return;
    // Find player wizard focus
    const pw = currentState.team_a.find(w => w.id === playerWizardId);
    if (!pw) return;
    const focus = pw.focus ?? 0;

    document.querySelectorAll('.spell-btn').forEach(btn => {
        const idx = parseInt(btn.dataset.idx);
        const spell = SPELLS[idx];
        if (focus < spell.cost || currentState.game_over) {
            btn.classList.add('disabled');
        } else {
            btn.classList.remove('disabled');
        }
    });
}

function showTooltip(spell, e) {
    const tt = document.getElementById('spell-tooltip');
    if (!tt) return;
    tt.innerHTML = `
        <div class="tt-name" style="color:${spell.color}">${spell.icon} ${spell.name}</div>
        <div class="tt-type">${spell.type} — ${spell.cost > 0 ? spell.cost + ' Focus' : 'Free'}</div>
        <div class="tt-desc">${spell.desc}</div>
    `;
    tt.classList.remove('hidden');
    // Position near the button
    const btn = e.currentTarget;
    const r = btn.getBoundingClientRect();
    const bar = document.getElementById('controls').getBoundingClientRect();
    tt.style.left = (r.left - bar.left + r.width/2 - 125) + 'px';
}

function hideTooltip() {
    const tt = document.getElementById('spell-tooltip');
    if (tt) tt.classList.add('hidden');
}

function onSpellClick(idx, btn) {
    if (isProcessing || animating) return;

    document.querySelectorAll('.spell-btn').forEach(b => b.classList.remove('selected'));
    selectedSpellIdx = idx;
    btn.classList.add('selected');

    // Enable targeting on enemies (and allies for Descendo)
    enableTargeting();
}

function enableTargeting() {
    const spell = SPELLS[selectedSpellIdx];
    if (!spell) return;

    // Clear any existing targetable states
    document.querySelectorAll('.wizard-card').forEach(c => {
        c.classList.remove('targetable');
        const ret = c.querySelector('.target-reticle');
        if (ret) ret.classList.add('hidden');
    });

    // Revelio: auto-cast (no target needed)
    if (spell.name === 'Revelio') {
        submitAction(playerWizardId);
        return;
    }

    // Protego Maximus: self-cast
    if (spell.name === 'Protego Maximus') {
        submitAction(playerWizardId);
        return;
    }

    // Protego: target an enemy to shield AGAINST (directional)
    // Descendo: can target allies (to save) or enemies
    // Everything else: just enemies
    const canTargetAllies = (spell.name === 'Descendo');
    const zones = canTargetAllies ? ['team-a', 'team-b'] : ['team-b'];

    zones.forEach(zoneId => {
        const zone = document.getElementById(zoneId);
        if (!zone) return;
        zone.querySelectorAll('.wizard-card:not(.defeated)').forEach(card => {
            card.classList.add('targetable');
            const ret = card.querySelector('.target-reticle');
            if (ret) ret.classList.remove('hidden');
        });
    });
}

function clearTargeting() {
    selectedSpellIdx = null;
    document.querySelectorAll('.spell-btn').forEach(b => b.classList.remove('selected'));
    document.querySelectorAll('.wizard-card').forEach(c => {
        c.classList.remove('targetable');
        const ret = c.querySelector('.target-reticle');
        if (ret) ret.classList.add('hidden');
    });
}


// ─── GAME LOGIC ───

async function createGame() {
    const avatar = document.querySelector('#ally-avatar-picker .avatar-option.selected')?.dataset.avatar || 'harry';
    const archetype = document.getElementById('p-archetype').value;
    const teamASize = parseInt(document.getElementById('team-a-size').value);
    const enemyCount = parseInt(document.getElementById('enemy-count').value);
    const allyModel = document.getElementById('ally-model').value;
    const enemyModel = document.getElementById('enemy-model').value;
    const enemyAvatar = document.querySelector('#enemy-avatar-picker .avatar-option.selected')?.dataset.avatar || 'voldemort';

    // Build model variant info
    function modelToConfig(modelStr) {
        switch(modelStr) {
            case 'unified_evo':
                return { control: 'unified', variant: { file_pattern: 'unified_best_{i}.pth', checkpoint_dir: 'checkpoints_evo_gpu' }};
            case 'unified_mappo':
                return { control: 'unified', variant: { file_pattern: 'mappo_best.pth', single_file: true, checkpoint_dir: 'checkpoints_mappo_gpu' }};
            case 'unified_pbt':
                return { control: 'unified', variant: { file_pattern: 'pbt_best.pth', single_file: true, checkpoint_dir: 'checkpoints_pbt_gpu' }};
            case 'unified_dqn':
                return { control: 'unified', variant: { file_pattern: 'dqn_best.pth', single_file: true, checkpoint_dir: 'checkpoints_dqn_gpu' }};
            case 'unified_imitation':
                return { control: 'unified', variant: { file_pattern: 'student_best.pth', single_file: true, checkpoint_dir: 'checkpoints_imitation_gpu' }};
            case 'legacy':
                return { control: 'legacy', variant: null };
            default:
                return { control: 'random', variant: null };
        }
    }

    const allyConf = modelToConfig(allyModel);
    const enemyConf = modelToConfig(enemyModel);

    const payload = {
        team_a: {
            size: teamASize,
            control: 'player',
            ally_ai: allyConf,
            avatars: [{ name: 'Hero', archetype, avatar_id: avatar }],
            variant: { player_position: 1 }
        },
        team_b: {
            size: enemyCount,
            control: enemyConf.control,
            variant: enemyConf.variant,
            avatars: Array(enemyCount).fill({ name: 'Enemy', archetype: 'default', avatar_id: enemyAvatar })
        }
    };

    try {
        // Determine endpoint based on mode
        let endpoint = `${API}/create`;
        const body = { ...payload };

        if (draftedDeck && draftedDeck.length > 0) {
            endpoint = `${API}/create_draft`;
            body.drafted_deck = draftedDeck.map(s => s.name);
        }

        const res = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        if (!res.ok) throw new Error(`Server ${res.status}: ${await res.text()}`);
        const data = await res.json();
        sessionId = data.session_id;
        playerWizardId = data.player_id ?? 0;

        switchScreen('battle');
        initCanvas();
        renderSpellBar(); // Re-render to reflect drafted deck

        // ── AUDIO: start battle BGM ──
        if (typeof audioManager !== 'undefined') {
            audioManager.playBGM(Math.random() > 0.5 ? 'bgm_battle' : 'bgm_battle_alt');
        }

        // ── DYNAMIC BACKGROUND: set weather ──
        if (typeof dynamicBG !== 'undefined' && dynamicBG.isActive) {
            const weatherSel = document.getElementById('weather-select');
            const weatherChoice = weatherSel ? weatherSel.value : 'random';
            const weatherName = weatherChoice === 'random' ? dynamicBG.randomizeWeather() : (dynamicBG.setWeather(weatherChoice), weatherChoice);
            const weatherIndicator = document.getElementById('weather-indicator');
            if (weatherIndicator) {
                const weatherIcons = { clear: '☀️', fog: '🌫️', rain: '🌧️', storm: '⛈️', embers: '🔥', snow: '❄️', dark_mist: '🌑' };
                weatherIndicator.textContent = `${weatherIcons[weatherName] || '🌤️'} ${weatherName.charAt(0).toUpperCase() + weatherName.slice(1).replace('_', ' ')}`;
            }
        }

        await pollState();
    } catch (e) {
        console.error('Failed to create game:', e);
        alert('Failed to start game: ' + e.message);
    }
}

async function pollState() {
    if (!sessionId) return;
    try {
        const res = await fetch(`${API}/state/${sessionId}`);
        if (!res.ok) throw new Error(`Poll ${res.status}`);
        const state = await res.json();
        await renderState(state);
    } catch (e) {
        console.error('Poll error:', e);
    }
}

async function submitAction(targetId) {
    if (selectedSpellIdx === null || isProcessing || animating) return;
    isProcessing = true;

    const spell = SPELLS[selectedSpellIdx];
    clearTargeting();

    // Arm click-to-continue for next-turn mode
    if (clickMode === 'next-turn') {
        clickNextTurnOnly = true;
    }

    try {
        const res = await fetch(`${API}/action`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                spell_name: spell.name,
                target_id: targetId
            })
        });
        if (!res.ok) throw new Error(`Action ${res.status}`);
        const state = await res.json();
        await renderState(state);
    } catch (e) {
        console.error('Action failed:', e);
    } finally {
        isProcessing = false;
    }
}

// ─── RENDERING ───

async function renderState(state) {
    currentState = state;

    // Turn indicator
    const ti = document.getElementById('turn-indicator');
    if (ti) ti.textContent = `Turn ${state.turn} / ${state.max_turns}`;

    // Render teams
    renderTeam('team-a', state.team_a, true);
    renderTeam('team-b', state.team_b, false);

    // Play animations (sequential with click-to-continue or auto-advance)
    if (state.animation_events && state.animation_events.length > 0) {
        lastRoundEvents = state.animation_events;
        // Show replay button
        const replayBtn = document.getElementById('replay-btn');
        if (replayBtn) replayBtn.classList.remove('hidden');

        await playAnimationSequence(state.animation_events, false);
    }

    // Add any text-only logs
    if (state.text_logs) {
        const logBox = document.getElementById('round-log');
        state.text_logs.forEach(txt => {
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.textContent = txt;
            logBox.appendChild(entry);
        });
        if (logBox) logBox.scrollTop = logBox.scrollHeight;
    }

    // Update spell bar availability
    updateSpellBarState();

    // Check game over
    if (state.game_over) {
        showWinScreen(state);
        return;
    }

    // If player is dead but game is not over, auto-advance turns
    const playerWiz = (state.team_a || []).find(w => w.id === playerWizardId);
    if (playerWiz && typeof playerWiz.hp === 'number' && playerWiz.hp <= 0) {
        // Player is dead — auto-submit to keep game moving
        setTimeout(async () => {
            isProcessing = true;
            try {
                const res = await fetch(`${API}/action`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        spell_name: 'Basic Cast',
                        target_id: playerWizardId // dummy target, player is dead anyway
                    })
                });
                const s = await res.json();
                await renderState(s);
            } catch(e) { console.error(e); }
            finally { isProcessing = false; }
        }, 1500);
    }
}

function renderTeam(zoneId, wizards, isAlly) {
    const zone = document.getElementById(zoneId);
    if (!zone || !wizards) return;

    wizards.forEach(wiz => {
        let card = zone.querySelector(`.wizard-card[data-id="${wiz.id}"]`);
        if (!card) {
            card = createWizardCard(wiz, isAlly);
            zone.appendChild(card);
        }
        updateWizardCard(card, wiz, isAlly);
    });
}

function createWizardCard(wiz, isAlly) {
    const tmpl = document.getElementById('wizard-card-template');
    const clone = tmpl.content.cloneNode(true);
    const card = clone.querySelector('.wizard-card');
    card.dataset.id = wiz.id;

    if (isAlly) {
        card.classList.add('ally-card');
        if (wiz.id === playerWizardId) card.classList.add('player-card');
    } else {
        card.classList.add('enemy-card');
    }

    // Click to target
    card.addEventListener('click', () => {
        if (selectedSpellIdx !== null && card.classList.contains('targetable')) {
            submitAction(wiz.id);
        }
    });

    return card;
}

function updateWizardCard(card, wiz, isAlly) {
    // Name
    const nameEl = card.querySelector('.wizard-name');
    let nameText = wiz.name;
    if (wiz.id === playerWizardId) nameText += ' (YOU)';
    nameEl.textContent = nameText;
    nameEl.style.color = isAlly ? '#4ade80' : '#f87171';

    // Distance positioning
    const dist = typeof wiz.dist === 'number' ? wiz.dist : 1;
    card.dataset.dist = dist;
    const distLabel = card.querySelector('.dist-label');
    if (distLabel) {
        const distName = DIST_NAMES[dist] || 'MID';
        distLabel.textContent = distName;
        distLabel.className = 'dist-label';
        if (dist === 0) distLabel.classList.add('dist-close');
        else if (dist === 1) distLabel.classList.add('dist-mid');
        else distLabel.classList.add('dist-far');
        if (!showDistLabels) distLabel.classList.add('dist-hidden');
    }

    // ASCII Avatar
    const avatarEl = card.querySelector('.ascii-avatar');
    if (wiz.ascii_art) {
        avatarEl.textContent = wiz.ascii_art;
    } else {
        avatarEl.textContent = wiz.avatar_id || '?';
    }

    // Shield glow — green background when Protego/MaxShield active
    const hasShield = wiz.status && (wiz.status.Shield > 0 || wiz.status.MaxShield > 0);
    if (hasShield) {
        card.classList.add('shield-glow');
    } else {
        card.classList.remove('shield-glow');
    }

    // Fog of war for enemies
    const isRevealed = wiz.is_revealed !== false;
    if (!isAlly && !isRevealed) {
        card.classList.add('fog-of-war');
    } else {
        card.classList.remove('fog-of-war');
    }

    // Stats (only if revealed or ally)
    if (isAlly || isRevealed) {
        const hp = typeof wiz.hp === 'number' ? wiz.hp : 0;
        const maxHp = wiz.max_hp || 100;
        const pos = typeof wiz.posture === 'number' ? wiz.posture : 0;
        const maxPos = wiz.max_posture || 50;
        const foc = typeof wiz.focus === 'number' ? wiz.focus : 0;
        const maxFoc = wiz.max_focus || 150;

        const hpPct = Math.max(0, (hp / maxHp) * 100);
        const posPct = Math.max(0, (pos / maxPos) * 100);
        const focPct = Math.max(0, (foc / maxFoc) * 100);

        const hpFill = card.querySelector('.hp-fill');
        const posFill = card.querySelector('.pos-fill');
        const focFill = card.querySelector('.foc-fill');

        if (hpFill) {
            hpFill.style.width = hpPct + '%';
            hpFill.classList.toggle('low', hpPct < 30);
        }
        if (posFill) {
            posFill.style.width = Math.max(0, posPct) + '%';
            posFill.classList.toggle('broken', pos <= 0);
        }
        if (focFill) focFill.style.width = focPct + '%';

        // Values
        const hpVal = card.querySelector('.hp-value');
        const posVal = card.querySelector('.pos-value');
        const focVal = card.querySelector('.foc-value');
        if (hpVal) hpVal.textContent = Math.round(hp);
        if (posVal) posVal.textContent = Math.round(pos);
        if (focVal) focVal.textContent = Math.round(foc);

        // Show/hide focus bar for enemies (hidden unless revealed)
        const focRow = card.querySelector('.foc-row');
        if (focRow) focRow.style.display = isAlly ? '' : (isRevealed ? '' : 'none');
    }

    // Status badges
    const badgeContainer = card.querySelector('.status-badges');
    if (badgeContainer) {
        badgeContainer.innerHTML = '';
        if (wiz.status) {
            const statusMap = {
                'Airborne': 'badge-airborne',
                'Frozen': 'badge-frozen',
                'Slowed': 'badge-slowed',
                'Brittle': 'badge-brittle',
                'Stunned': 'badge-stunned',
                'CursedPain': 'badge-cursedpain',
                'Shield': 'badge-shield',
                'MaxShield': 'badge-maxshield',
            };
            Object.entries(wiz.status).forEach(([key, val]) => {
                if (val > 0 || val === true) {
                    const badge = document.createElement('span');
                    badge.className = `status-badge ${statusMap[key] || ''}`;
                    badge.textContent = key === 'CursedPain' ? `DoT${val}` : (typeof val === 'number' && val > 1 ? `${key}${val}` : key);
                    badge.title = `${key}: ${val}`;
                    badgeContainer.appendChild(badge);
                }
            });
        }
        // Show if revealed by info spell
        if (!isAlly && wiz.is_revealed && wiz.scan_timer > 0) {
            const badge = document.createElement('span');
            badge.className = 'status-badge badge-revealed';
            badge.textContent = `Seen(${wiz.scan_timer})`;
            badgeContainer.appendChild(badge);
        }
    }

    // Defeated state
    if (typeof wiz.hp === 'number' && wiz.hp <= 0) {
        card.classList.add('defeated');
    } else {
        card.classList.remove('defeated');
    }
}


// ─── WIN SCREEN ───

function showWinScreen(state) {
    const winScreen = document.getElementById('win-screen');
    if (!winScreen) return;

    const icon = document.getElementById('win-icon');
    const title = document.getElementById('win-title');
    const subtitle = document.getElementById('win-subtitle');
    const stats = document.getElementById('win-stats');

    const winner = state.winner || 'Draw';
    let isVictory = false;

    if (winner.includes('A') || winner.includes('ally') || winner.toLowerCase().includes('ally')) {
        icon.textContent = '🏆';
        title.textContent = 'VICTORY!';
        title.className = 'victory';
        subtitle.textContent = 'Your team has won the duel!';
        isVictory = true;
    } else if (winner.includes('Draw') || winner.includes('Max')) {
        icon.textContent = '⚖️';
        title.textContent = 'DRAW';
        title.className = 'draw';
        subtitle.textContent = 'The duel ended in a draw.';
    } else {
        icon.textContent = '💀';
        title.textContent = 'DEFEAT';
        title.className = 'defeat';
        subtitle.textContent = 'The enemy team has won...';
    }

    // ── AUDIO: victory/defeat sting ──
    if (typeof audioManager !== 'undefined') {
        audioManager.playResultSFX(isVictory);
    }

    // Stats summary
    const aliveA = (state.team_a || []).filter(w => w.hp > 0).length;
    const aliveB = (state.team_b || []).filter(w => typeof w.hp === 'number' && w.hp > 0).length;

    stats.innerHTML = `
        <div class="win-stat-box">
            <div class="stat-num" style="color:#4ade80">${aliveA}</div>
            <div class="stat-lbl">Allies Alive</div>
        </div>
        <div class="win-stat-box">
            <div class="stat-num" style="color:#f87171">${aliveB}</div>
            <div class="stat-lbl">Enemies Alive</div>
        </div>
        <div class="win-stat-box">
            <div class="stat-num" style="color:#a78bfa">${state.turn || 0}</div>
            <div class="stat-lbl">Rounds</div>
        </div>
    `;

    winScreen.classList.add('active');
}

function hideWinScreen() {
    const ws = document.getElementById('win-screen');
    if (ws) ws.classList.remove('active');
}


// ─── SCREEN MANAGEMENT ───

function switchScreen(target) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    const el = document.getElementById(target + '-screen');
    if (el) el.classList.add('active');
}

function resetToLobby() {
    sessionId = null;
    selectedSpellIdx = null;
    currentState = null;
    isProcessing = false;
    animating = false;
    isReplaying = false;
    playerWizardId = null;
    particles = [];
    lastRoundEvents = [];
    draftedDeck = null;

    // Hide replay button
    const replayBtn = document.getElementById('replay-btn');
    if (replayBtn) replayBtn.classList.add('hidden');

    // Clear battle zones
    const ta = document.getElementById('team-a');
    const tb = document.getElementById('team-b');
    if (ta) ta.innerHTML = '';
    if (tb) tb.innerHTML = '';

    const log = document.getElementById('round-log');
    if (log) log.innerHTML = '';

    if (particleAnimFrame) {
        cancelAnimationFrame(particleAnimFrame);
        particleAnimFrame = null;
    }

    // ── AUDIO: lobby BGM ──
    if (typeof audioManager !== 'undefined') {
        audioManager.playBGM('bgm_lobby');
    }

    // ── DYNAMIC BG: calm weather ──
    if (typeof dynamicBG !== 'undefined' && dynamicBG.isActive) {
        dynamicBG.setWeather('clear');
    }

    hideWinScreen();
    switchScreen('lobby');
}


// ─── AVATAR PICKER ───

function initAvatarPickers() {
    document.querySelectorAll('.avatar-picker').forEach(picker => {
        picker.querySelectorAll('.avatar-option').forEach(opt => {
            opt.addEventListener('click', () => {
                picker.querySelectorAll('.avatar-option').forEach(o => o.classList.remove('selected'));
                opt.classList.add('selected');
            });
        });
    });
}


// ─── SETTINGS ───

function openSettings() {
    const overlay = document.getElementById('settings-overlay');
    if (overlay) overlay.classList.add('active');
}

function closeSettings() {
    const overlay = document.getElementById('settings-overlay');
    if (overlay) overlay.classList.remove('active');
}

function initSettings() {
    // Open/Close
    document.getElementById('settings-btn')?.addEventListener('click', openSettings);
    document.getElementById('settings-close-btn')?.addEventListener('click', closeSettings);

    // Speed slider
    const slider = document.getElementById('anim-speed-slider');
    const sliderVal = document.getElementById('anim-speed-value');
    if (slider) {
        slider.addEventListener('input', () => {
            animSpeed = parseFloat(slider.value);
            if (sliderVal) sliderVal.textContent = animSpeed.toFixed(1) + 'x';
        });
    }

    // Click mode radio buttons
    document.querySelectorAll('input[name="click-mode"]').forEach(radio => {
        radio.addEventListener('change', () => {
            clickMode = radio.value;
            // If switching to "next-turn", immediately arm it
            if (clickMode === 'next-turn') {
                clickNextTurnOnly = true;
            }
        });
    });

    // Distance labels toggle
    const distToggle = document.getElementById('show-dist-labels');
    if (distToggle) {
        distToggle.addEventListener('change', () => {
            showDistLabels = distToggle.checked;
            // Update all visible labels
            document.querySelectorAll('.dist-label').forEach(el => {
                el.classList.toggle('dist-hidden', !showDistLabels);
            });
        });
    }

    // Replay button
    document.getElementById('replay-btn')?.addEventListener('click', replayLastRound);
}


// ─── INIT ───

function init() {
    renderSpellBar();
    initAvatarPickers();
    initSettings();
    initModeTabs();
    initAudioControls();
    initTutorialUI();
    initDraftUI();

    // ── DYNAMIC BACKGROUND ──
    if (typeof dynamicBG !== 'undefined') {
        const bgCanvas = document.getElementById('bg-canvas');
        if (bgCanvas) {
            dynamicBG.init(bgCanvas);
            dynamicBG.setWeather('clear');
            dynamicBG.start();
        }
    }

    // ── AUDIO: lobby music ──
    if (typeof audioManager !== 'undefined') {
        // Start lobby music on first user interaction (browser autoplay policy)
        document.addEventListener('click', function startLobbyMusic() {
            audioManager.playBGM('bgm_lobby');
            document.removeEventListener('click', startLobbyMusic);
        }, { once: true });
    }

    document.getElementById('start-btn')?.addEventListener('click', () => {
        if (currentMode === 'grimoire') {
            // Start draft first
            switchScreen('draft');
            startDraftUI();
        } else {
            createGame();
        }
    });
    document.getElementById('back-to-lobby-btn')?.addEventListener('click', resetToLobby);
}

// ─── MODE TABS ───

function initModeTabs() {
    document.querySelectorAll('.mode-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            currentMode = tab.dataset.mode;

            const startBtn = document.getElementById('start-btn');
            const setupGrid = document.querySelector('.setup-grid');

            if (currentMode === 'tutorial') {
                switchScreen('tutorial');
                renderTutorialLessons();
            } else if (currentMode === 'grimoire') {
                // Show setup grid but change button text
                if (startBtn) {
                    startBtn.innerHTML = '<span class="btn-icon">📖</span> Draft & Fight';
                }
                if (setupGrid) setupGrid.style.display = '';
            } else {
                // Classic mode
                if (startBtn) {
                    startBtn.innerHTML = '<span class="btn-icon">⚔️</span> Begin Duel';
                }
                if (setupGrid) setupGrid.style.display = '';
            }
        });
    });
}

// ─── AUDIO CONTROLS ───

function initAudioControls() {
    if (typeof audioManager === 'undefined') return;

    // Mute buttons
    const muteBtn = document.getElementById('mute-btn');
    const battleMuteBtn = document.getElementById('battle-mute-btn');

    function toggleMute(btn) {
        const muted = audioManager.toggleMute();
        if (btn) {
            btn.textContent = muted ? '🔇' : '🔊';
            btn.classList.toggle('muted', muted);
        }
        // Sync the other button
        [muteBtn, battleMuteBtn].forEach(b => {
            if (b) {
                b.textContent = muted ? '🔇' : '🔊';
                b.classList.toggle('muted', muted);
            }
        });
        if (!muted) audioManager.playBGM('bgm_lobby');
    }

    if (muteBtn) muteBtn.addEventListener('click', () => toggleMute(muteBtn));
    if (battleMuteBtn) battleMuteBtn.addEventListener('click', () => toggleMute(battleMuteBtn));

    // Volume sliders (lobby)
    const bgmSlider = document.getElementById('bgm-vol');
    const sfxSlider = document.getElementById('sfx-vol');
    if (bgmSlider) bgmSlider.addEventListener('input', () => audioManager.setBGMVolume(bgmSlider.value / 100));
    if (sfxSlider) sfxSlider.addEventListener('input', () => audioManager.setSFXVolume(sfxSlider.value / 100));

    // Volume sliders (settings)
    const sBgm = document.getElementById('settings-bgm-vol');
    const sSfx = document.getElementById('settings-sfx-vol');
    if (sBgm) sBgm.addEventListener('input', () => {
        audioManager.setBGMVolume(sBgm.value / 100);
        if (bgmSlider) bgmSlider.value = sBgm.value;
    });
    if (sSfx) sSfx.addEventListener('input', () => {
        audioManager.setSFXVolume(sSfx.value / 100);
        if (sfxSlider) sfxSlider.value = sSfx.value;
    });
}

// ─── TUTORIAL UI ───

function initTutorialUI() {
    document.getElementById('tutorial-back-btn')?.addEventListener('click', () => {
        switchScreen('lobby');
        document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
        document.querySelector('.mode-tab[data-mode="classic"]')?.classList.add('active');
        currentMode = 'classic';
    });
    document.getElementById('tut-reset-btn')?.addEventListener('click', () => {
        if (tutorialManager.lessonIndex >= 0) {
            tutorialManager.resetLesson();
            renderTutorialArena();
        }
    });
    document.getElementById('tut-back-to-list-btn')?.addEventListener('click', () => {
        document.getElementById('tutorial-arena').classList.add('hidden');
        document.getElementById('tutorial-lesson-list').style.display = '';
        renderTutorialLessons();
    });
    document.getElementById('tut-next-btn')?.addEventListener('click', () => {
        const next = tutorialManager.lessonIndex + 1;
        if (next < TUTORIAL_LESSONS.length) {
            startTutorialLesson(next);
        }
    });
}

function renderTutorialLessons() {
    const list = document.getElementById('tutorial-lesson-list');
    if (!list) return;
    list.style.display = '';
    document.getElementById('tutorial-arena')?.classList.add('hidden');

    const progress = tutorialManager.getProgress();
    list.innerHTML = '';

    progress.lessons.forEach((lesson, idx) => {
        const card = document.createElement('div');
        card.className = `tutorial-lesson-card${lesson.completed ? ' completed' : ''}`;
        card.innerHTML = `
            <div class="lesson-card-icon">${lesson.icon}</div>
            <div class="lesson-card-title">${lesson.title}</div>
            <div class="lesson-card-desc">${lesson.description.slice(0, 80)}...</div>
        `;
        card.addEventListener('click', () => startTutorialLesson(idx));
        list.appendChild(card);
    });
}

function startTutorialLesson(index) {
    const lesson = tutorialManager.startLesson(index);
    if (!lesson) return;

    document.getElementById('tutorial-lesson-list').style.display = 'none';
    document.getElementById('tutorial-arena').classList.remove('hidden');
    document.getElementById('tut-next-btn')?.classList.add('hidden');

    renderTutorialArena();
}

function renderTutorialArena() {
    const lesson = tutorialManager.currentLesson;
    if (!lesson) return;

    document.getElementById('tutorial-title').textContent = `${lesson.icon} ${lesson.title}`;
    document.getElementById('tutorial-desc').textContent = lesson.description;
    document.getElementById('tutorial-tip').textContent = `💡 Tip: ${lesson.tip}`;
    document.getElementById('tutorial-objective').textContent = `🎯 Objective: ${lesson.objective}`;
    document.getElementById('tutorial-message').textContent = '';
    document.getElementById('tutorial-message').className = 'tutorial-message';

    updateTutorialStats();
    renderTutorialSpells();
}

function updateTutorialStats() {
    const state = tutorialManager.getState();

    // Player stats
    const pHP = Math.max(0, state.player.hp);
    document.getElementById('tut-player-hp').textContent = Math.round(pHP);
    document.getElementById('tut-player-hp-bar').style.width = (pHP / state.player.maxHP * 100) + '%';
    document.getElementById('tut-player-pos').textContent = Math.round(state.player.posture);
    document.getElementById('tut-player-pos-bar').style.width = Math.max(0, state.player.posture / state.player.maxPosture * 100) + '%';
    document.getElementById('tut-player-foc').textContent = Math.round(state.player.focus);
    document.getElementById('tut-player-foc-bar').style.width = (state.player.focus / state.player.maxFocus * 100) + '%';

    // Enemy stats
    const eHP = Math.max(0, state.enemy.hp);
    document.getElementById('tut-enemy-hp').textContent = Math.round(eHP);
    document.getElementById('tut-enemy-hp-bar').style.width = (eHP / state.enemy.maxHP * 100) + '%';
    document.getElementById('tut-enemy-pos').textContent = Math.round(state.enemy.posture);
    document.getElementById('tut-enemy-pos-bar').style.width = Math.max(0, state.enemy.posture / state.enemy.maxPosture * 100) + '%';

    // Distance
    const distNames = ['CLOSE', 'MID', 'FAR'];
    const distEl = document.getElementById('tut-enemy-dist');
    if (distEl) {
        distEl.textContent = distNames[state.enemy.dist] || 'MID';
        distEl.className = 'dist-label ' + ['dist-close', 'dist-mid', 'dist-far'][state.enemy.dist];
    }

    // Status badges
    renderStatusBadges('tut-player-status', state.player.status);
    renderStatusBadges('tut-enemy-status', state.enemy.status);
}

function renderStatusBadges(elementId, status) {
    const el = document.getElementById(elementId);
    if (!el) return;
    el.innerHTML = '';
    const classes = {
        Airborne: 'badge-airborne', Frozen: 'badge-frozen', Slowed: 'badge-slowed',
        Brittle: 'badge-brittle', Stunned: 'badge-stunned', CursedPain: 'badge-cursedpain',
        Shield: 'badge-shield', MaxShield: 'badge-maxshield',
    };
    for (const [key, val] of Object.entries(status)) {
        if (val > 0) {
            const badge = document.createElement('span');
            badge.className = `status-badge ${classes[key] || ''}`;
            badge.textContent = key === 'CursedPain' ? `DoT${val}` : key;
            el.appendChild(badge);
        }
    }
}

function renderTutorialSpells() {
    const bar = document.getElementById('tutorial-spell-bar');
    if (!bar) return;
    bar.innerHTML = '';

    const lesson = tutorialManager.currentLesson;
    if (!lesson) return;

    const state = tutorialManager.getState();

    lesson.allowedSpells.forEach(idx => {
        const spell = SPELLS[idx];
        if (!spell) return;

        const btn = document.createElement('div');
        btn.className = 'tut-spell-btn';
        if (state.player.focus < spell.cost) btn.classList.add('disabled');
        btn.style.borderColor = spell.color;
        btn.innerHTML = `
            <div class="tut-spell-icon">${spell.icon}</div>
            <div class="tut-spell-name">${spell.name}</div>
            <div class="tut-spell-cost">${spell.cost > 0 ? spell.cost + ' FOC' : 'FREE'}</div>
        `;
        btn.addEventListener('click', () => {
            if (btn.classList.contains('disabled')) return;
            const result = tutorialManager.castSpell(idx);
            if (!result) return;

            // Play SFX
            if (typeof audioManager !== 'undefined') {
                audioManager.playSpellSFX(spell.name);
                if (result.events.some(e => (e.hp_dmg || 0) > 0)) {
                    setTimeout(() => audioManager.playHitSFX(), 200);
                }
            }

            updateTutorialStats();
            renderTutorialSpells();

            const msg = document.getElementById('tutorial-message');
            if (msg && result.message) {
                msg.textContent = result.message;
                if (result.victory) msg.className = 'tutorial-message victory';
                else if (result.done && !result.victory) msg.className = 'tutorial-message defeat';
            }

            if (result.done) {
                // Disable spells
                bar.querySelectorAll('.tut-spell-btn').forEach(b => b.classList.add('disabled'));
                if (result.victory && tutorialManager.lessonIndex < TUTORIAL_LESSONS.length - 1) {
                    document.getElementById('tut-next-btn')?.classList.remove('hidden');
                }
            }
        });
        bar.appendChild(btn);
    });
}

// ─── DRAFT UI ───

function initDraftUI() {
    document.getElementById('draft-back-btn')?.addEventListener('click', () => {
        switchScreen('lobby');
        document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
        document.querySelector('.mode-tab[data-mode="classic"]')?.classList.add('active');
        currentMode = 'classic';
    });
    document.getElementById('draft-reroll-btn')?.addEventListener('click', () => {
        const result = draftManager.reroll();
        if (result) renderDraftState(result);
    });
    document.getElementById('draft-fight-btn')?.addEventListener('click', () => {
        draftedDeck = draftManager.getFinalDeck();
        switchScreen('lobby');
        createGame(); // Start battle with drafted deck
    });
}

function startDraftUI() {
    const result = draftManager.startDraft();
    document.getElementById('draft-complete')?.classList.add('hidden');
    document.querySelector('.draft-area')?.style.setProperty('display', '');
    renderDraftState(result);
}

function renderDraftState(state) {
    if (state.done) {
        // Draft complete
        document.querySelector('.draft-area')?.style.setProperty('display', 'none');
        document.getElementById('draft-complete')?.classList.remove('hidden');

        const finalDeck = document.getElementById('draft-final-deck');
        if (finalDeck) {
            finalDeck.innerHTML = '';
            const allSpells = draftManager.getFinalDeck();
            allSpells.forEach(spell => {
                const slot = document.createElement('div');
                slot.className = 'draft-slot filled';
                slot.innerHTML = `<div class="slot-icon">${spell.icon}</div><div class="slot-name">${spell.name}</div>`;
                finalDeck.appendChild(slot);
            });
        }

        const synList = document.getElementById('draft-final-synergies');
        if (synList && state.synergies.length > 0) {
            synList.innerHTML = '<strong>Synergies:</strong> ' + state.synergies.map(s => s.label).join(', ');
        }
        return;
    }

    // Update round label
    const roundLabel = document.getElementById('draft-round-label');
    if (roundLabel) roundLabel.textContent = `Round ${state.round} / ${state.maxRounds}`;

    // Update reroll button
    const rerollBtn = document.getElementById('draft-reroll-btn');
    if (rerollBtn) {
        rerollBtn.textContent = `🎲 Reroll (${state.rerollsLeft})`;
        if (state.rerollsLeft <= 0) rerollBtn.classList.add('disabled');
        else rerollBtn.classList.remove('disabled');
    }

    // Render deck slots
    const slots = document.getElementById('draft-deck-slots');
    if (slots) {
        slots.innerHTML = '';
        for (let i = 0; i < draftManager.maxRounds; i++) {
            const slot = document.createElement('div');
            slot.className = 'draft-slot';
            if (i < state.deck.length) {
                slot.classList.add('filled');
                slot.innerHTML = `<div class="slot-icon">${state.deck[i].icon}</div><div class="slot-name">${state.deck[i].name}</div>`;
            } else {
                slot.innerHTML = `<div class="slot-empty">Slot ${i + 1}</div>`;
            }
            slots.appendChild(slot);
        }
    }

    // Render choices
    const choices = document.getElementById('draft-choices');
    if (choices) {
        choices.innerHTML = '';
        state.choices.forEach(spell => {
            const card = document.createElement('div');
            card.className = 'draft-choice-card';
            card.dataset.type = spell.type;

            // Check for potential synergies with current deck
            let synergyHint = '';
            for (const deckSpell of state.deck) {
                const syn = SYNERGIES[deckSpell.name];
                if (syn && syn[spell.name]) {
                    synergyHint = `Synergy with ${deckSpell.name}!`;
                    break;
                }
                const synRev = SYNERGIES[spell.name];
                if (synRev && synRev[deckSpell.name]) {
                    synergyHint = `Synergy with ${deckSpell.name}!`;
                    break;
                }
            }

            card.innerHTML = `
                <div class="choice-icon">${spell.icon}</div>
                <div class="choice-name">${spell.name}</div>
                <div class="choice-type" style="color:${spell.color}; background:${spell.color}22">${spell.type}</div>
                <div class="choice-desc">${spell.desc}</div>
                <div class="choice-cost">${spell.cost > 0 ? spell.cost + ' Focus' : 'FREE'}</div>
                ${synergyHint ? `<div class="choice-synergy">🔗 ${synergyHint}</div>` : ''}
            `;
            card.addEventListener('click', () => {
                if (typeof audioManager !== 'undefined') audioManager.playSFX('sfx_cast_attack');
                const result = draftManager.pickSpell(spell.name);
                if (result) renderDraftState(result);
            });
            choices.appendChild(card);
        });
    }

    // Synergies display
    const synContainer = document.getElementById('draft-synergies');
    const synListEl = document.getElementById('draft-synergy-list');
    if (synContainer && synListEl) {
        if (state.synergies.length > 0) {
            synContainer.classList.remove('hidden');
            synListEl.innerHTML = '';
            state.synergies.forEach(s => {
                const item = document.createElement('div');
                item.className = 'synergy-item';
                item.innerHTML = `<span class="synergy-label">${s.label}</span>`;
                synListEl.appendChild(item);
            });
        } else {
            synContainer.classList.add('hidden');
        }
    }
}

document.addEventListener('DOMContentLoaded', init);
