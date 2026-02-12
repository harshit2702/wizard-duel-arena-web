/* ======================================================
   SPELL TUTORIAL — tutorial.js
   Interactive 1v1 spell practice that teaches each spell
   ====================================================== */

const TUTORIAL_LESSONS = [
    {
        id: 'basic_cast',
        title: 'Basic Cast',
        spellIdx: 10,
        icon: '✨',
        description: 'The foundation of every wizard duel. Basic Cast is free to use and deals 5 HP damage + 5 Posture damage. It also regenerates 5 posture for the caster.',
        tip: 'Use Basic Cast when you\'re low on Focus, or to chip away at an enemy\'s posture.',
        enemy: { hp: 40, posture: 20, focus: 0, archetype: 'default' },
        objective: 'Defeat the target dummy using only Basic Cast.',
        allowedSpells: [10],
        victoryCondition: 'kill',
    },
    {
        id: 'protego',
        title: 'Protego (Shield)',
        spellIdx: 0,
        icon: '🛡️',
        description: 'Protego blocks the next incoming attack and reflects Control spells. It costs 0 Focus and is your primary survival tool.',
        tip: 'Cast Protego when you expect a big hit. Timing is everything — predict your enemy\'s moves!',
        enemy: { hp: 999, posture: 50, focus: 100, archetype: 'default', ai_pattern: ['Incendio', 'Incendio', 'Basic Cast'] },
        objective: 'Block 3 attacks using Protego, then strike back with Basic Cast.',
        allowedSpells: [0, 10],
        victoryCondition: 'survive_5',
    },
    {
        id: 'levioso_descendo',
        title: 'Levioso + Descendo Combo',
        spellIdx: 4,
        icon: '🪶',
        description: 'Levioso lifts the target into the air (Airborne status). Descendo slams airborne targets for MASSIVE bonus damage! This is a core combo.',
        tip: 'Cast Levioso first, then Descendo while the enemy is still Airborne for a devastating slam!',
        enemy: { hp: 80, posture: 30, focus: 0, archetype: 'default' },
        objective: 'Lift the enemy with Levioso, then slam them with Descendo.',
        allowedSpells: [4, 9, 10],
        victoryCondition: 'kill',
    },
    {
        id: 'glacius_diffindo',
        title: 'Glacius + Diffindo (Ice Shatter)',
        spellIdx: 5,
        icon: '❄️',
        description: 'Glacius freezes the target (they skip a turn) AND applies Brittle — meaning the NEXT hit deals 2x damage. Follow up with Diffindo (45 base damage) for a devastating 90-damage shatter!',
        tip: 'Glacius → Diffindo is one of the deadliest 2-spell combos in the game.',
        enemy: { hp: 100, posture: 40, focus: 0, archetype: 'default' },
        objective: 'Freeze the enemy with Glacius, then shatter them with Diffindo.',
        allowedSpells: [5, 13, 10],
        victoryCondition: 'kill',
    },
    {
        id: 'accio_incendio',
        title: 'Accio + Incendio (Pull & Burn)',
        spellIdx: 7,
        icon: '🧲',
        description: 'Accio pulls the target to CLOSE range. Incendio deals 30 damage at CLOSE range (only 15 at FAR). Pull them in, then burn them!',
        tip: 'Range matters! Incendio is devastating at CLOSE but weak at FAR. Use Accio to close the gap.',
        enemy: { hp: 80, posture: 40, focus: 0, archetype: 'default', startDist: 2 },
        objective: 'Pull the enemy close with Accio, then burn them with Incendio.',
        allowedSpells: [7, 11, 10],
        victoryCondition: 'kill',
    },
    {
        id: 'depulso',
        title: 'Depulso (Force Push)',
        spellIdx: 8,
        icon: '⚡',
        description: 'Depulso pushes the target away, dealing 25 posture damage and 5 HP. Use it to create distance and break posture!',
        tip: 'Depulso is great for keeping aggressive enemies at bay and breaking their posture for Avada Kedavra.',
        enemy: { hp: 60, posture: 30, focus: 0, archetype: 'default' },
        objective: 'Break the enemy\'s posture with Depulso and finish them off.',
        allowedSpells: [8, 10],
        victoryCondition: 'kill',
    },
    {
        id: 'revelio_legilimens',
        title: 'Revelio & Legilimens (Info Spells)',
        spellIdx: 2,
        icon: '👁️',
        description: 'In battle, enemy stats are hidden (Fog of War). Revelio reveals ALL enemies for 3 turns. Legilimens reveals one target in detail for 5 turns. Information is power!',
        tip: 'Cast Revelio early to plan your strategy. Use Legilimens on a priority target.',
        enemy: { hp: 60, posture: 30, focus: 0, archetype: 'default', fogOfWar: true },
        objective: 'Use Revelio to see enemy stats, then defeat the target.',
        allowedSpells: [2, 3, 10, 13],
        victoryCondition: 'kill',
    },
    {
        id: 'confringo',
        title: 'Confringo (Long Range)',
        spellIdx: 12,
        icon: '💥',
        description: 'Confringo is an explosive spell that deals 25 HP + 10 posture. It gains a +10 BONUS at FAR range! Unlike most spells, distance helps Confringo.',
        tip: 'Use Depulso to push enemies FAR, then bombard them with Confringo for maximum damage.',
        enemy: { hp: 90, posture: 40, focus: 0, archetype: 'default' },
        objective: 'Push enemies back and hit with long-range Confringo.',
        allowedSpells: [8, 12, 10],
        victoryCondition: 'kill',
    },
    {
        id: 'crucio',
        title: 'Crucio (Unforgivable Curse)',
        spellIdx: 14,
        icon: '💀',
        description: 'Crucio costs 50 Focus but STUNS the target for 2 turns and applies CursedPain (Damage over Time: 15 → 10 → 7). Devastating against tough enemies.',
        tip: 'Save Crucio for high-value targets. The DoT damage stacks up over multiple turns!',
        enemy: { hp: 80, posture: 50, focus: 0, archetype: 'brawler' },
        objective: 'Curse the enemy and watch them suffer through DoT damage.',
        allowedSpells: [14, 10],
        victoryCondition: 'kill',
    },
    {
        id: 'avada_kedavra',
        title: 'Avada Kedavra (Killing Curse)',
        spellIdx: 15,
        icon: '☠️',
        description: 'The ultimate spell. Costs 100 Focus. If the target\'s posture is BROKEN (≤ 0), it\'s an INSTANT KILL. Otherwise, deals 40 HP. Also chains to any target with CursedPain!',
        tip: 'Break their posture first with Depulso/Levioso, THEN cast Avada Kedavra for the instakill!',
        enemy: { hp: 120, posture: 25, focus: 0, archetype: 'brawler' },
        objective: 'Break posture, then execute with Avada Kedavra.',
        allowedSpells: [15, 8, 4, 10],
        victoryCondition: 'kill',
    },
    {
        id: 'protego_maximus',
        title: 'Protego Maximus (Focus Shield)',
        spellIdx: 1,
        icon: '🛡️',
        description: 'Protego Maximus costs 30 Focus but converts incoming damage to Focus drain instead of HP loss. Powerful against heavy hitters!',
        tip: 'Use it when you expect massive damage. Your Focus absorbs the hit instead of HP.',
        enemy: { hp: 999, posture: 50, focus: 150, archetype: 'curse_specialist', ai_pattern: ['Diffindo', 'Confringo', 'Incendio', 'Crucio'] },
        objective: 'Use Protego Maximus to survive 3 turns against the assault.',
        allowedSpells: [1, 0, 10],
        victoryCondition: 'survive_5',
    },
    {
        id: 'arresto',
        title: 'Arresto Momentum (Slow)',
        spellIdx: 6,
        icon: '⏸️',
        description: 'Arresto Momentum slows the target — they cannot regenerate posture. Combined with posture-damaging spells, this leads to a posture break!',
        tip: 'Slow + Depulso = rapid posture destruction. Then finish with a kill spell.',
        enemy: { hp: 80, posture: 40, focus: 0, archetype: 'default' },
        objective: 'Slow the enemy to prevent posture regen, break posture, and win.',
        allowedSpells: [6, 8, 10, 15],
        victoryCondition: 'kill',
    },
];

class TutorialManager {
    constructor() {
        this.currentLesson = null;
        this.lessonIndex = -1;
        this.playerHP = 100;
        this.playerMaxHP = 100;
        this.playerPosture = 50;
        this.playerMaxPosture = 50;
        this.playerFocus = 100;
        this.playerMaxFocus = 150;
        this.playerDist = 1; // MID
        this.enemyHP = 0;
        this.enemyMaxHP = 0;
        this.enemyPosture = 0;
        this.enemyMaxPosture = 0;
        this.enemyFocus = 0;
        this.enemyDist = 1;
        this.enemyStatus = {};
        this.playerStatus = {};
        this.turnCount = 0;
        this.blockedCount = 0;
        this.comboState = {};
        this.isActive = false;
        this.completedLessons = new Set(JSON.parse(localStorage.getItem('completedTutorials') || '[]'));
    }

    startLesson(index) {
        if (index < 0 || index >= TUTORIAL_LESSONS.length) return;
        this.lessonIndex = index;
        this.currentLesson = TUTORIAL_LESSONS[index];
        this.turnCount = 0;
        this.blockedCount = 0;
        this.comboState = {};
        this.isActive = true;

        // Reset player
        this.playerHP = this.playerMaxHP = 100;
        this.playerPosture = this.playerMaxPosture = 50;
        this.playerFocus = this.playerMaxFocus = 150;
        this.playerDist = 1;
        this.playerStatus = {};

        // Setup enemy
        const e = this.currentLesson.enemy;
        this.enemyHP = this.enemyMaxHP = e.hp || 60;
        this.enemyPosture = this.enemyMaxPosture = e.posture || 30;
        this.enemyFocus = e.focus || 0;
        this.enemyDist = e.startDist ?? 1;
        this.enemyStatus = {};

        return this.currentLesson;
    }

    /** Process a player spell cast in tutorial mode. Returns animation events + result. */
    castSpell(spellIdx) {
        if (!this.isActive || !this.currentLesson) return null;

        const spell = SPELLS[spellIdx];
        if (!spell) return null;

        // Check focus cost
        if (this.playerFocus < spell.cost) {
            return { events: [], message: 'Not enough Focus!', done: false };
        }

        this.playerFocus -= spell.cost;
        this.turnCount++;

        const events = [];
        let message = '';

        // ─── RESOLVE PLAYER SPELL ───
        const result = this._resolveSpell(spell, true);
        events.push({
            type: 'cast',
            caster_id: 'player',
            target_id: 'enemy',
            spell: spell.name,
            hp_dmg: result.hpDmg,
            pos_dmg: result.posDmg,
            effect: result.effect,
            caster_name: 'You',
            target_name: 'Target Dummy',
        });
        message = result.message || '';

        // ─── ENEMY AI TURN ───
        if (this.enemyHP > 0 && this.currentLesson.enemy.ai_pattern) {
            const patternIdx = (this.turnCount - 1) % this.currentLesson.enemy.ai_pattern.length;
            const enemySpellName = this.currentLesson.enemy.ai_pattern[patternIdx];

            // Check if player has shield
            if (this.playerStatus.Shield > 0) {
                events.push({
                    type: 'cast',
                    caster_id: 'enemy',
                    target_id: 'player',
                    spell: enemySpellName,
                    hp_dmg: 0,
                    pos_dmg: 0,
                    effect: 'Blocked by Protego!',
                    caster_name: 'Enemy',
                    target_name: 'You',
                });
                this.playerStatus.Shield = 0;
                this.blockedCount++;
            } else {
                const enemySpell = SPELLS.find(s => s.name === enemySpellName);
                if (enemySpell) {
                    const dmg = this._getSpellDamage(enemySpell, false);
                    this.playerHP -= dmg.hpDmg;
                    this.playerPosture -= dmg.posDmg;
                    events.push({
                        type: 'cast',
                        caster_id: 'enemy',
                        target_id: 'player',
                        spell: enemySpellName,
                        hp_dmg: dmg.hpDmg,
                        pos_dmg: dmg.posDmg,
                        effect: `Hit for ${dmg.hpDmg} HP`,
                        caster_name: 'Enemy',
                        target_name: 'You',
                    });
                }
            }
        }

        // ─── TICK STATUS EFFECTS ───
        this._tickStatuses();

        // ─── CHECK VICTORY ───
        let done = false;
        let victory = false;

        if (this.currentLesson.victoryCondition === 'kill' && this.enemyHP <= 0) {
            done = true;
            victory = true;
            message = '🎉 Lesson Complete! Enemy defeated!';
        } else if (this.currentLesson.victoryCondition === 'survive_5' && this.turnCount >= 5 && this.playerHP > 0) {
            done = true;
            victory = true;
            message = '🎉 Lesson Complete! You survived!';
        } else if (this.playerHP <= 0) {
            done = true;
            victory = false;
            message = '💀 You were defeated. Try again!';
        }

        if (victory) {
            this.completedLessons.add(this.currentLesson.id);
            localStorage.setItem('completedTutorials', JSON.stringify([...this.completedLessons]));
        }

        // Focus regen
        this.playerFocus = Math.min(this.playerMaxFocus, this.playerFocus + 15);

        return {
            events,
            message,
            done,
            victory,
            state: this.getState(),
        };
    }

    _resolveSpell(spell, isPlayer) {
        let hpDmg = 0, posDmg = 0, effect = '', message = '';

        switch (spell.name) {
            case 'Basic Cast':
                hpDmg = 5; posDmg = 5;
                this.enemyHP -= hpDmg;
                this.enemyPosture -= posDmg;
                if (isPlayer) this.playerPosture = Math.min(this.playerMaxPosture, this.playerPosture + 5);
                effect = `${hpDmg} HP, ${posDmg} Posture`;
                break;

            case 'Protego':
                this.playerStatus.Shield = 1;
                effect = 'Shield raised!';
                message = '🛡️ Shield is up — next attack will be blocked.';
                break;

            case 'Protego Maximus':
                this.playerStatus.MaxShield = 2;
                effect = 'Focus Shield active!';
                message = '🛡️ Focus Shield — damage converts to focus drain.';
                break;

            case 'Levioso':
                posDmg = 20;
                this.enemyPosture -= posDmg;
                this.enemyStatus.Airborne = 2;
                effect = 'Target lifted! (Airborne)';
                message = '🪶 Enemy is Airborne! Follow up with Descendo!';
                break;

            case 'Descendo':
                posDmg = 20;
                hpDmg = 10;
                if (this.enemyStatus.Airborne > 0) {
                    hpDmg = 40; posDmg = 30;
                    this.enemyStatus.Airborne = 0;
                    effect = 'SLAM! Bonus from Airborne!';
                    message = '💥 COMBO! Airborne slam for massive damage!';
                } else {
                    effect = 'Ground slam.';
                }
                this.enemyHP -= hpDmg;
                this.enemyPosture -= posDmg;
                break;

            case 'Glacius':
                this.enemyStatus.Frozen = 1;
                this.enemyStatus.Brittle = 1;
                effect = 'Frozen + Brittle!';
                message = '❄️ Enemy is Frozen and Brittle! Next hit deals 2x damage!';
                break;

            case 'Diffindo':
                hpDmg = 45;
                if (this.enemyStatus.Brittle > 0) {
                    hpDmg *= 2;
                    this.enemyStatus.Brittle = 0;
                    effect = `ICE SHATTER! ${hpDmg} HP!`;
                    message = '💎 SHATTER! Brittle doubled the damage!';
                } else {
                    effect = `Slash for ${hpDmg} HP`;
                }
                posDmg = 5;
                this.enemyHP -= hpDmg;
                this.enemyPosture -= posDmg;
                break;

            case 'Accio':
                hpDmg = 5; posDmg = 15;
                this.enemyHP -= hpDmg;
                this.enemyPosture -= posDmg;
                this.enemyDist = 0; // Pull to CLOSE
                effect = 'Pulled to CLOSE range!';
                message = '🧲 Enemy pulled to CLOSE range!';
                break;

            case 'Depulso':
                hpDmg = 5; posDmg = 25;
                this.enemyHP -= hpDmg;
                this.enemyPosture -= posDmg;
                this.enemyDist = Math.min(2, this.enemyDist + 1);
                effect = 'Pushed back!';
                message = '⚡ Enemy pushed to ' + ['CLOSE', 'MID', 'FAR'][this.enemyDist] + ' range!';
                break;

            case 'Incendio':
                const incendioDmg = [30, 20, 15]; // CLOSE, MID, FAR
                hpDmg = incendioDmg[this.enemyDist] || 20;
                posDmg = 5;
                if (this.enemyStatus.Brittle > 0) { hpDmg *= 2; this.enemyStatus.Brittle = 0; }
                this.enemyHP -= hpDmg;
                this.enemyPosture -= posDmg;
                effect = `🔥 ${hpDmg} fire damage at ${['CLOSE', 'MID', 'FAR'][this.enemyDist]}!`;
                break;

            case 'Confringo':
                hpDmg = this.enemyDist === 2 ? 35 : 25;
                posDmg = 10;
                if (this.enemyStatus.Brittle > 0) { hpDmg *= 2; this.enemyStatus.Brittle = 0; }
                this.enemyHP -= hpDmg;
                this.enemyPosture -= posDmg;
                effect = `💥 ${hpDmg} explosive damage!`;
                break;

            case 'Revelio':
                effect = 'All enemies revealed!';
                message = '👁️ Enemy stats are now visible for 3 turns.';
                break;

            case 'Legilimens':
                effect = 'Target mind-read!';
                message = '🧠 Target stats revealed in detail for 5 turns.';
                break;

            case 'Arresto Momentum':
                posDmg = 30;
                this.enemyPosture -= posDmg;
                this.enemyStatus.Slowed = 3;
                effect = 'Slowed! (No posture regen)';
                message = '⏸️ Enemy slowed — posture won\'t regenerate!';
                break;

            case 'Crucio':
                this.enemyStatus.Stunned = 2;
                this.enemyStatus.CursedPain = 3;
                effect = 'CRUCIO! Stun + DoT!';
                message = '💀 Enemy stunned and cursed with damage over time!';
                break;

            case 'Avada Kedavra':
                if (this.enemyPosture <= 0) {
                    this.enemyHP = 0;
                    effect = '☠️ INSTANT KILL! Posture was broken!';
                    message = '☠️ AVADA KEDAVRA — INSTANT KILL!';
                } else {
                    hpDmg = 40;
                    this.enemyHP -= hpDmg;
                    effect = `${hpDmg} HP (posture not broken)`;
                    message = 'Avada Kedavra hit but posture wasn\'t broken — only 40 damage.';
                }
                break;
        }

        return { hpDmg, posDmg, effect, message };
    }

    _getSpellDamage(spell, isPlayer) {
        const dmgMap = {
            'Basic Cast': { hpDmg: 5, posDmg: 5 },
            'Incendio': { hpDmg: 25, posDmg: 5 },
            'Confringo': { hpDmg: 25, posDmg: 10 },
            'Diffindo': { hpDmg: 45, posDmg: 5 },
            'Crucio': { hpDmg: 0, posDmg: 0 },
        };
        return dmgMap[spell.name] || { hpDmg: 10, posDmg: 5 };
    }

    _tickStatuses() {
        // Enemy status decay
        for (const key of Object.keys(this.enemyStatus)) {
            if (typeof this.enemyStatus[key] === 'number' && this.enemyStatus[key] > 0) {
                // CursedPain deals DoT
                if (key === 'CursedPain') {
                    const dotDmg = [15, 10, 7][3 - this.enemyStatus[key]] || 7;
                    this.enemyHP -= dotDmg;
                }
                this.enemyStatus[key]--;
                if (this.enemyStatus[key] <= 0) delete this.enemyStatus[key];
            }
        }
        // Player status decay
        for (const key of Object.keys(this.playerStatus)) {
            if (key === 'Shield') continue; // Shield persists until used
            if (typeof this.playerStatus[key] === 'number' && this.playerStatus[key] > 0) {
                this.playerStatus[key]--;
                if (this.playerStatus[key] <= 0) delete this.playerStatus[key];
            }
        }
        // Posture regen (if not slowed)
        if (!this.enemyStatus.Slowed) {
            this.enemyPosture = Math.min(this.enemyMaxPosture, this.enemyPosture + 5);
        }
    }

    getState() {
        return {
            player: {
                hp: Math.max(0, this.playerHP),
                maxHP: this.playerMaxHP,
                posture: this.playerPosture,
                maxPosture: this.playerMaxPosture,
                focus: this.playerFocus,
                maxFocus: this.playerMaxFocus,
                dist: this.playerDist,
                status: { ...this.playerStatus },
            },
            enemy: {
                hp: Math.max(0, this.enemyHP),
                maxHP: this.enemyMaxHP,
                posture: this.enemyPosture,
                maxPosture: this.enemyMaxPosture,
                focus: this.enemyFocus,
                dist: this.enemyDist,
                status: { ...this.enemyStatus },
            },
            turn: this.turnCount,
        };
    }

    getProgress() {
        return {
            total: TUTORIAL_LESSONS.length,
            completed: this.completedLessons.size,
            lessons: TUTORIAL_LESSONS.map(l => ({
                ...l,
                completed: this.completedLessons.has(l.id),
            })),
        };
    }

    resetLesson() {
        if (this.lessonIndex >= 0) {
            this.startLesson(this.lessonIndex);
        }
    }
}

// Global instance
const tutorialManager = new TutorialManager();
