/* ======================================================
   THE GRIMOIRE — Deck Building Draft System
   draft.js — Roguelike spell drafting before battle
   ====================================================== */

// ─── SPELL TAGS & SYNERGIES ───

const SPELL_TAGS = {
    'Protego':          ['Defense', 'Shield', 'Free'],
    'Protego Maximus':  ['Defense', 'Shield', 'Expensive'],
    'Revelio':          ['Info', 'Utility'],
    'Legilimens':       ['Info', 'Utility', 'Single-Target'],
    'Levioso':          ['Control', 'Combo-Starter', 'Airborne'],
    'Glacius':          ['Control', 'Combo-Starter', 'Freeze', 'Brittle'],
    'Arresto Momentum': ['Control', 'Slow', 'Posture-Breaker'],
    'Accio':            ['Force', 'Pull', 'Close-Range-Setup'],
    'Depulso':          ['Force', 'Push', 'Posture-Breaker'],
    'Descendo':         ['Force', 'Combo-Finisher', 'Airborne-Payoff'],
    'Basic Cast':       ['Damage', 'Free', 'Basic'],
    'Incendio':         ['Damage', 'Fire', 'AoE', 'Close-Range'],
    'Confringo':        ['Damage', 'Fire', 'Long-Range'],
    'Diffindo':         ['Damage', 'High-Damage', 'Combo-Finisher', 'Brittle-Payoff'],
    'Crucio':           ['Curse', 'DoT', 'Stun', 'Combo-Starter'],
    'Avada Kedavra':    ['Curse', 'Instakill', 'Combo-Finisher', 'Expensive'],
};

const SYNERGIES = {
    'Levioso':          { 'Descendo': 5.0, 'Accio': 2.0, 'Depulso': 1.5 },
    'Glacius':          { 'Diffindo': 5.0, 'Incendio': 3.0, 'Confringo': 2.0 },
    'Accio':            { 'Incendio': 4.0, 'Basic Cast': 1.5 },
    'Depulso':          { 'Confringo': 3.0, 'Avada Kedavra': 2.0 },
    'Arresto Momentum': { 'Depulso': 3.0, 'Avada Kedavra': 3.0 },
    'Crucio':           { 'Avada Kedavra': 10.0, 'Diffindo': 2.0 },
    'Incendio':         { 'Accio': 2.0 },
    'Confringo':        { 'Depulso': 2.0 },
    'Diffindo':         { 'Glacius': 2.0 },
    'Protego':          { 'Protego Maximus': 1.5 },
};

// ─── ARCHETYPE PREFERENCES (for AI drafting) ───

const ARCHETYPE_PREFERENCES = {
    defender: {
        mustPick: ['Protego', 'Protego Maximus'],
        preferred: ['Defense', 'Shield'],
        weights: { 'Protego': 10, 'Protego Maximus': 8 },
    },
    aggressor: {
        mustPick: ['Diffindo', 'Incendio'],
        preferred: ['Damage', 'High-Damage', 'Fire'],
        weights: { 'Diffindo': 8, 'Incendio': 7, 'Confringo': 6 },
    },
    controller: {
        mustPick: ['Levioso', 'Glacius'],
        preferred: ['Control', 'Combo-Starter'],
        weights: { 'Levioso': 8, 'Glacius': 8, 'Arresto Momentum': 6 },
    },
    curse_master: {
        mustPick: ['Crucio'],
        preferred: ['Curse', 'DoT', 'Combo-Finisher'],
        weights: { 'Crucio': 10, 'Avada Kedavra': 9, 'Arresto Momentum': 5 },
    },
    balanced: {
        mustPick: [],
        preferred: [],
        weights: {},
    },
};


class DraftManager {
    constructor() {
        this.pool = [];
        this.deck = [];
        this.history = []; // Rejected choices
        this.round = 0;
        this.maxRounds = 6;
        this.choicesPerRound = 3;
        this.rerollsLeft = 1;
        this.currentChoices = [];
        this.isActive = false;
        this.synergiesFound = [];
    }

    /** Start a new draft session */
    startDraft() {
        // Pool = all non-Basic-Cast spells (Basic Cast is always innate)
        this.pool = SPELLS
            .filter(s => s.name !== 'Basic Cast')
            .map(s => ({ ...s }));

        this.deck = [];
        this.history = [];
        this.round = 0;
        this.maxRounds = 6;
        this.rerollsLeft = 1;
        this.currentChoices = [];
        this.synergiesFound = [];
        this.isActive = true;

        return this.nextRound();
    }

    /** Advance to next draft round, return 3 weighted choices */
    nextRound() {
        if (this.round >= this.maxRounds) {
            this.isActive = false;
            return { done: true, deck: this.deck, synergies: this.synergiesFound };
        }

        this.round++;
        this.currentChoices = this._getWeightedChoices();

        return {
            done: false,
            round: this.round,
            maxRounds: this.maxRounds,
            choices: this.currentChoices,
            deck: this.deck,
            rerollsLeft: this.rerollsLeft,
            synergies: this.synergiesFound,
        };
    }

    /** Player picks a spell from current choices */
    pickSpell(spellName) {
        const chosen = this.currentChoices.find(c => c.name === spellName);
        if (!chosen) return null;

        // Add to deck
        this.deck.push(chosen);

        // Remove from pool
        this.pool = this.pool.filter(s => s.name !== spellName);

        // Track rejected
        this.currentChoices
            .filter(c => c.name !== spellName)
            .forEach(c => this.history.push(c.name));

        // Check if new synergies unlocked
        this._checkSynergies(chosen);

        return this.nextRound();
    }

    /** Use reroll to get 3 new choices */
    reroll() {
        if (this.rerollsLeft <= 0) return null;
        this.rerollsLeft--;

        // Add current choices to history
        this.currentChoices.forEach(c => this.history.push(c.name));
        this.currentChoices = this._getWeightedChoices();

        return {
            done: false,
            round: this.round,
            maxRounds: this.maxRounds,
            choices: this.currentChoices,
            deck: this.deck,
            rerollsLeft: this.rerollsLeft,
            synergies: this.synergiesFound,
        };
    }

    /** AI auto-draft based on archetype */
    autoDraft(archetypeKey = 'balanced') {
        this.startDraft();
        const arch = ARCHETYPE_PREFERENCES[archetypeKey] || ARCHETYPE_PREFERENCES.balanced;

        while (this.round < this.maxRounds) {
            const state = this.nextRound();
            if (state.done) break;

            // Prioritize must-pick spells
            let pick = null;
            for (const must of arch.mustPick) {
                const found = this.currentChoices.find(c => c.name === must);
                if (found) { pick = found; break; }
            }

            // Then pick by preference weights
            if (!pick) {
                let bestScore = -1;
                for (const choice of this.currentChoices) {
                    let score = arch.weights[choice.name] || 0;
                    const tags = SPELL_TAGS[choice.name] || [];
                    for (const tag of tags) {
                        if (arch.preferred.includes(tag)) score += 3;
                    }
                    // Synergy bonus
                    for (const deckSpell of this.deck) {
                        const syn = SYNERGIES[deckSpell.name];
                        if (syn && syn[choice.name]) score += syn[choice.name];
                    }
                    if (score > bestScore) {
                        bestScore = score;
                        pick = choice;
                    }
                }
            }

            if (!pick) pick = this.currentChoices[0];
            this.pickSpell(pick.name);
        }

        return { deck: this.deck, synergies: this.synergiesFound };
    }

    /** Get the final deck including Basic Cast */
    getFinalDeck() {
        const basicCast = SPELLS.find(s => s.name === 'Basic Cast');
        return [basicCast, ...this.deck];
    }

    // ─── INTERNAL ───

    _getWeightedChoices() {
        if (this.pool.length === 0) return [];

        // Calculate weights for each spell in pool
        const weighted = this.pool.map(spell => {
            let weight = 1.0;
            const tags = SPELL_TAGS[spell.name] || [];

            // ── SYNERGY BOOST ──
            for (const deckSpell of this.deck) {
                const syn = SYNERGIES[deckSpell.name];
                if (syn && syn[spell.name]) {
                    weight += syn[spell.name];
                }
            }

            // ── DIRECTOR AI: balance check ──
            const deckTags = this.deck.flatMap(s => SPELL_TAGS[s.name] || []);
            const hasDamage = deckTags.filter(t => t === 'Damage' || t === 'High-Damage').length;
            const hasDefense = deckTags.filter(t => t === 'Defense' || t === 'Shield').length;
            const hasControl = deckTags.filter(t => t === 'Control').length;

            // If no defense by round 4, boost defensive spells
            if (this.round >= 4 && hasDefense === 0 && tags.includes('Defense')) {
                weight += 4.0;
            }
            // If no damage, boost damage spells
            if (this.round >= 3 && hasDamage === 0 && tags.includes('Damage')) {
                weight += 3.0;
            }
            // If no control, slight boost
            if (this.round >= 3 && hasControl === 0 && tags.includes('Control')) {
                weight += 2.0;
            }

            // ── ANTI-REPEAT: don't offer recently rejected spells ──
            if (this.history.includes(spell.name)) {
                weight *= 0.5;
            }

            // ── EXPENSIVE PENALTY early rounds ──
            if (this.round <= 2 && spell.cost >= 50) {
                weight *= 0.4;
            }

            return { spell, weight };
        });

        // Weighted random selection of 3
        const count = Math.min(this.choicesPerRound, weighted.length);
        const selected = [];
        const remaining = [...weighted];

        for (let i = 0; i < count; i++) {
            const totalWeight = remaining.reduce((sum, w) => sum + w.weight, 0);
            let rand = Math.random() * totalWeight;

            for (let j = 0; j < remaining.length; j++) {
                rand -= remaining[j].weight;
                if (rand <= 0) {
                    selected.push(remaining[j].spell);
                    remaining.splice(j, 1);
                    break;
                }
            }
        }

        return selected;
    }

    _checkSynergies(newSpell) {
        for (const deckSpell of this.deck) {
            if (deckSpell.name === newSpell.name) continue;

            // Check if existing → new is a synergy
            const syn = SYNERGIES[deckSpell.name];
            if (syn && syn[newSpell.name]) {
                this.synergiesFound.push({
                    from: deckSpell.name,
                    to: newSpell.name,
                    label: this._getSynergyLabel(deckSpell.name, newSpell.name),
                });
            }
            // Check reverse
            const synRev = SYNERGIES[newSpell.name];
            if (synRev && synRev[deckSpell.name]) {
                this.synergiesFound.push({
                    from: newSpell.name,
                    to: deckSpell.name,
                    label: this._getSynergyLabel(newSpell.name, deckSpell.name),
                });
            }
        }
    }

    _getSynergyLabel(from, to) {
        const labels = {
            'Levioso→Descendo': '🪶⬇️ Air Slam Combo',
            'Glacius→Diffindo': '❄️🗡️ Ice Shatter Combo',
            'Glacius→Incendio': '❄️🔥 Melt Combo',
            'Accio→Incendio': '🧲🔥 Pull & Burn',
            'Depulso→Confringo': '⚡💥 Push & Blast',
            'Depulso→Avada Kedavra': '⚡☠️ Posture Break → Kill',
            'Arresto Momentum→Depulso': '⏸️⚡ Slow → Shatter Posture',
            'Arresto Momentum→Avada Kedavra': '⏸️☠️ Slow → Execute',
            'Crucio→Avada Kedavra': '💀☠️ THE FORBIDDEN COMBO',
            'Crucio→Diffindo': '💀🗡️ Curse & Cut',
        };
        return labels[`${from}→${to}`] || `${from} + ${to}`;
    }
}

// Global instance
const draftManager = new DraftManager();
