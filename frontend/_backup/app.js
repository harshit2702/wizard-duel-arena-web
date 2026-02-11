const API_URL = "/api";
let session_id = null;
let selected_spell_idx = null;
let current_state = null;
let is_processing = false;

// Spells definition (Client-side mirror for UI)
const SPELLS = [
    { name: "Basic Cast", type: "Damage", cost: 0, color: "#cbd5e1" },
    { name: "Protego", type: "Defense", cost: 0, color: "#22c55e" },
    { name: "Protego Maximus", type: "Defense", cost: 30, color: "#166534" },
    { name: "Revelio", type: "Info", cost: 10, color: "#3b82f6" },
    { name: "Levioso", type: "Control", cost: 15, color: "#a855f7" },
    { name: "Glacius", type: "Control", cost: 20, color: "#0ea5e9" },
    { name: "Accio", type: "Force", cost: 15, color: "#eab308" },
    { name: "Depulso", type: "Force", cost: 15, color: "#ca8a04" },
    { name: "Descendo", type: "Force", cost: 20, color: "#854d0e" },
    { name: "Incendio", type: "Damage", cost: 25, color: "#dc2626" },
    { name: "Confringo", type: "Damage", cost: 30, color: "#b91c1c" },
    { name: "Avada Kedavra", type: "Curse", cost: 100, color: "#020617" }
];

// --- DOM Elements ---
const screens = {
    lobby: document.getElementById('lobby-screen'),
    battle: document.getElementById('battle-screen')
};
const team_zones = {
    a: document.getElementById('team-a'),
    b: document.getElementById('team-b')
};
const log_box = document.getElementById('log-box');
const spell_bar = document.getElementById('spell-bar');
const svg_layer = document.getElementById('effects-layer');

// --- Debug / Error Handling ---
// --- Debug / Error Handling ---
function logToScreen(msg, isError = false) {
    if (isError) {
        console.error(msg);
        // Optional: Keep error toasts? User asked to hide logs. 
        // Let's keep errors as alerts or small toasts, but hide general info.
        // For now, I'll just log to console to be clean as requested.
    } else {
        console.log(msg);
    }
}

// --- Initialization ---
function init() {
    try {
        const startBtn = document.getElementById('start-btn');
        if (startBtn) startBtn.addEventListener('click', createGame);
        renderSpellBar();
        if (window.lucide) {
            try {
                lucide.createIcons();
            } catch (e) {
                logToScreen("Lucide init error: " + e, true);
            }
        } else {
            logToScreen("Warning: Lucide icons not loaded (Offline?)", true);
        }
    } catch (e) {
        logToScreen("Init failed: " + e, true);
    }
}

function renderSpellBar() {
    spell_bar.innerHTML = '';
    SPELLS.forEach((spell, idx) => {
        const btn = document.createElement('div');
        btn.className = 'spell-btn interactable';
        btn.dataset.idx = idx;
        btn.style.borderColor = spell.color;

        btn.innerHTML = `
            <div class="spell-name">${spell.name}</div>
            <div class="spell-cost">${spell.cost}</div>
        `;

        btn.addEventListener('click', () => selectSpell(idx, btn));
        spell_bar.appendChild(btn);
    });
}

function selectSpell(idx, btn) {
    if (is_processing) return;

    // Deselect all
    document.querySelectorAll('.spell-btn').forEach(b => b.classList.remove('selected'));

    // Select this
    selected_spell_idx = idx;
    btn.classList.add('selected');

    // Enable targeting mode
    document.body.classList.add('targeting-mode');
}

// --- Game Logic ---

async function createGame() {
    const avatar = document.getElementById('p-avatar').value;
    const archetype = document.getElementById('p-archetype').value;
    const team_a_size = parseInt(document.getElementById('team-a-size').value);
    const position = parseInt(document.getElementById('p-position').value);
    const enemy_count = parseInt(document.getElementById('enemy-count').value);

    // Ensure position <= size
    if (position > team_a_size) {
        logToScreen(`Player Position ${position} cannot be greater than Team Size ${team_a_size}!`, true);
        return;
    }

    const payload = {
        team_a: {
            size: team_a_size,
            control: "player",
            avatars: [{ name: "Hero", archetype: archetype, avatar_id: avatar }],
            variant: { player_position: position }
        },
        team_b: {
            size: enemy_count,
            control: document.getElementById('enemy-ai').value,
            avatars: Array(enemy_count).fill({ name: "Enemy", archetype: "default", avatar_id: "default" })
        }
    };

    try {
        logToScreen("Creating game...");
        const res = await fetch(`${API_URL}/create`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const txt = await res.text();
            throw new Error(`Server Error (${res.status}): ${txt}`);
        }

        const data = await res.json();
        session_id = data.session_id;
        logToScreen("Game created: " + session_id);

        // Switch screens
        if (screens.lobby) screens.lobby.classList.add('hidden');
        if (screens.battle) screens.battle.classList.remove('hidden');

        pollState();
        // Start polling loop
        setInterval(pollState, 1000);
    } catch (e) {
        logToScreen("Failed to start game: " + e, true);
    }
}

async function pollState() {
    if (!session_id) return;
    try {
        const res = await fetch(`${API_URL}/state/${session_id}`);
        if (!res.ok) {
            throw new Error(`Poll Status: ${res.status}`);
        }
        const state = await res.json();
        renderState(state);
    } catch (e) {
        logToScreen("Poll error: " + e, true);
        is_processing = false; // Reset lock if stuck
    }
}

async function submitAction(target_id) {
    if (selected_spell_idx === null || is_processing) return;

    is_processing = true;
    const spell = SPELLS[selected_spell_idx];

    try {
        const res = await fetch(`${API_URL}/action`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: session_id,
                spell_name: spell.name,
                target_id: target_id
            })
        });
        const state = await res.json();

        // Clear selection
        selected_spell_idx = null;
        document.querySelectorAll('.spell-btn').forEach(b => b.classList.remove('selected'));
        document.body.classList.remove('targeting-mode');

        renderState(state);
    } catch (e) {
        logToScreen("Action failed: " + e, true);
    } finally {
        is_processing = false;
    }
}

// --- Rendering ---

function renderState(state) {
    current_state = state;
    document.getElementById('turn-indicator').textContent = `Turn ${state.turn} / ${state.max_turns}`;

    // Render Logs
    if (state.logs) {
        log_box.innerHTML = state.logs.map(l => `<div class="log-entry">${l}</div>`).join('');
        log_box.scrollTop = log_box.scrollHeight;
    }

    // Render Teams
    logToScreen(`Rendering State: Turn ${state.turn}, A: ${state.team_a.length}, B: ${state.team_b.length}`);
    updateTeamZone(team_zones.a, state.team_a);
    updateTeamZone(team_zones.b, state.team_b);

    // Trigger Animations from state events
    if (state.animation_events && state.animation_events.length > 0) {
        playAnimations(state.animation_events);
    }
}

function updateTeamZone(zone, wizards) {
    // Basic diffing: verify IDs match, else rebuild
    // For simplicity in Vanilla JS, we'll try to find existing card or create new

    if (!zone) {
        logToScreen("Error: Zone not found available", true);
        return;
    }

    logToScreen(`Zone ${zone.id}: processing ${wizards.length} wizards`);

    wizards.forEach(wiz => {
        let card = zone.querySelector(`.wizard-card[data-id="${wiz.id}"]`);
        if (!card) {
            const template = document.getElementById('card-template');
            if (!template) {
                logToScreen("Error: card-template missing", true);
                return;
            }
            const clone = template.content.cloneNode(true);
            card = clone.querySelector('.wizard-card');
            card.dataset.id = wiz.id;

            // Interaction: Target clicking
            card.addEventListener('click', () => {
                if (selected_spell_idx !== null) {
                    submitAction(wiz.id);
                }
            });

            zone.appendChild(card);
            logToScreen(`Created Card: ${wiz.name} (ID: ${wiz.id})`);
        }

        updateCard(card, wiz);
    });

    // logToScreen(`Zone ${zone.id} children: ${zone.children.length}`);
}

function updateCard(card, wiz) {
    card.querySelector('.name-tag').textContent = wiz.name;

    // HP Bar
    const hp_pct = wiz.hp === "???" ? 100 : (wiz.hp / wiz.max_hp) * 100;
    card.querySelector('.fb-hp').style.width = `${Math.max(0, hp_pct)}%`;

    // Posture Bar
    const pos_pct = wiz.posture === "???" ? 100 : (wiz.posture / wiz.max_posture) * 100;
    card.querySelector('.fb-pos').style.width = `${Math.max(0, pos_pct)}%`;

    // Status
    const status_box = card.querySelector('.status-icons');
    status_box.innerHTML = ''; // clear
    if (wiz.status) {
        Object.entries(wiz.status).forEach(([key, val]) => {
            if (val > 0) {
                const icon = document.createElement('span');
                icon.className = 'status-badge';
                icon.textContent = key.substr(0, 1) + val; // e.g. F1 (Frozen 1)
                icon.title = `${key}: ${val}`;
                status_box.appendChild(icon);
            }
        });
    }

    // Revealed State
    if (wiz.is_revealed === false && wiz.hp === "???") {
        card.classList.add('fog-of-war');
    } else {
        card.classList.remove('fog-of-war');
    }

    if (wiz.hp <= 0 && wiz.hp !== "???") {
        card.classList.add('defeated');
    }
}

function playAnimations(events) {
    events.forEach(ev => {
        if (ev.type === 'cast') {
            shootBeam(ev.caster_id, ev.target_id, ev.color);
        }
    });
}

function shootBeam(fromId, toId, colorType) {
    const fromEl = document.querySelector(`.wizard-card[data-id="${fromId}"]`);
    const toEl = document.querySelector(`.wizard-card[data-id="${toId}"]`);

    if (!fromEl || !toEl) return;

    const r1 = fromEl.getBoundingClientRect();
    const r2 = toEl.getBoundingClientRect();
    const container = document.getElementById('battlefield').getBoundingClientRect();

    // Start center of card
    const x1 = r1.left + r1.width / 2 - container.left;
    const y1 = r1.top + r1.height / 2 - container.top;
    const x2 = r2.left + r2.width / 2 - container.left;
    const y2 = r2.top + r2.height / 2 - container.top;

    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", x1);
    line.setAttribute("y1", y1);
    line.setAttribute("x2", x2);
    line.setAttribute("y2", y2);

    let color = "white";
    if (colorType === "Damage") color = "red";
    if (colorType === "Control") color = "cyan";
    if (colorType === "Force") color = "orange";

    line.setAttribute("stroke", color);
    line.setAttribute("stroke-width", "4");
    line.setAttribute("class", "beam-anim");

    svg_layer.appendChild(line);

    setTimeout(() => line.remove(), 500);
}

// Start
init();
