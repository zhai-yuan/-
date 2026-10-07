// Prove two shipped .so files decide IDENTICALLY, by dlopening both and feeding them the same
// sequence of inputs.
//
// Why this and not a source-level check: the requirement is that a latency-only change does
// not alter behaviour, and the thing that gets submitted is the binary. Two builds of the
// same source with different flags (or a different vector width) can in principle differ, so
// the artifact pair is what has to be compared.
//
// Both strategies carry cross-round state (persistent direction target, bomb phase), so the
// inputs are replayed as an ORDERED sequence and the outputs compared step by step: a state
// divergence shows up on the round after it happens, not only on the round that caused it.
//
// Input sources:
//   1. the 1024 real rounds of tools/rank1_rounds.h, replayed in order
//   2. a deterministic pseudo-random board generator, so cells/positions outside the fixture
//      distribution are covered too (bombs, fog, dense walls, units at the border)
//
// Build (on the machine that produced the .so files):
//   clang++ -std=c++17 -O2 -I<dir with game_api.h> -I tools -o so_equiv tools/so_equiv.cpp -ldl
//   ./so_equiv A.so B.so [rounds]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <dlfcn.h>

#include "game_api.h"
#include "rank1_rounds.h"

using Fn = GameOutput (*)(const GameInput*);

static Fn load(const char* path, void** handle) {
    *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!*handle) {
        std::fprintf(stderr, "dlopen %s: %s\n", path, dlerror());
        return nullptr;
    }
    dlerror();
    Fn f = reinterpret_cast<Fn>(dlsym(*handle, "moveDecision"));
    const char* e = dlerror();
    if (e) {
        std::fprintf(stderr, "dlsym moveDecision in %s: %s\n", path, e);
        return nullptr;
    }
    return f;
}

static bool same(const GameOutput& a, const GameOutput& b) {
    if (a.k != b.k || a.order != b.order || a.vp != b.vp) return false;
    for (int t = 0; t < S; ++t) if (a.actions[t] != b.actions[t]) return false;
    return true;
}

static void printOutput(const char* name, const GameOutput& out) {
    std::printf("%s k=%d order=%d vp=%d actions=", name, out.k, out.order, out.vp);
    for (int t = 0; t < S; ++t) std::printf("%s%d", t ? "," : "", out.actions[t]);
    std::printf("\n");
}

static uint32_t rngState = 0x2c1b3f5du;
static uint32_t xs() {
    rngState ^= rngState << 13; rngState ^= rngState >> 17; rngState ^= rngState << 5;
    return rngState;
}

// Fill a random but RULE-SHAPED board: -5 fog, -3 bomb, -1 wall, 0 empty, >=1 gold.
static void randomInput(GameInput* in, int round) {
    std::memset(in, 0, sizeof(*in));
    in->round = round;
    const int wallPct = 5 + (int)(xs() % 50);
    for (int r = 0; r < GRID_SIZE; ++r)
        for (int c = 0; c < GRID_SIZE; ++c) {
            const uint32_t roll = xs() % 100u;
            int v;
            if (roll < (uint32_t)wallPct) v = -1;
            else if (roll < (uint32_t)wallPct + 4u) v = -3;
            else if (roll < (uint32_t)wallPct + 12u) v = -5;
            else if (roll < (uint32_t)wallPct + 30u) v = 1 + (int)(xs() % 205u);
            else v = 0;
            in->grid[r][c] = v;
        }
    for (int i = 0; i < 2; ++i) {
        in->my_units[i].row = (int)(xs() % GRID_SIZE);
        in->my_units[i].col = (int)(xs() % GRID_SIZE);
        in->my_units_gold[i] = (int)(xs() % 500u);
        in->grid[in->my_units[i].row][in->my_units[i].col] = 0;
    }
    for (int i = 0; i < 2; ++i) {
        if (xs() % 4u) {
            in->visible_enemies[i].row = (int)(xs() % GRID_SIZE);
            in->visible_enemies[i].col = (int)(xs() % GRID_SIZE);
        } else {
            in->visible_enemies[i].row = -1;
            in->visible_enemies[i].col = -1;
        }
    }
    for (int i = 0; i < MAX_NPCS; ++i) {
        if (xs() % 3u) {
            in->visible_npcs[i].id = -(1 + i);
            in->visible_npcs[i].pos.row = (int)(xs() % GRID_SIZE);
            in->visible_npcs[i].pos.col = (int)(xs() % GRID_SIZE);
        } else {
            in->visible_npcs[i].id = 0;
            in->visible_npcs[i].pos.row = -1;
            in->visible_npcs[i].pos.col = -1;
        }
    }
    in->gold_opp = (int)(xs() % 3000u);
    // Snapshot: exercise both "fresh" and "absent" so a build that reads it is caught even
    // though the current configuration should ignore it entirely.
    in->snapshot_valid = (int)(xs() % 2u);
    in->snapshot.window_begin = in->snapshot_valid ? round - 5 : -1;
    in->snapshot.window_end = round;
    for (int i = 0; i < REGION_COUNT; ++i) {
        in->snapshot.regions[i].id = i + 1;
        in->snapshot.regions[i].enter = (int)(xs() % 20u);
        in->snapshot.regions[i].leave = (int)(xs() % 20u);
        in->snapshot.regions[i].gold_generated = (int)(xs() % 400u);
        in->snapshot.regions[i].gold_collected = (int)(xs() % 400u);
        in->snapshot.regions[i].gold_remaining = (int)(xs() % 900u);
        in->snapshot.regions[i].occupants = (int)(xs() % 5u);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s A.so B.so [random_rounds]\n", argv[0]);
        return 2;
    }
    const int randomRounds = (argc > 3) ? std::atoi(argv[3]) : 200000;

    void* ha = nullptr;
    void* hb = nullptr;
    Fn a = load(argv[1], &ha);
    Fn b = load(argv[2], &hb);
    if (!a || !b) return 2;

    long checked = 0, diff = 0;
    int firstDiff = -1;
    GameOutput firstA = {}, firstB = {};
    int firstRound = -1;
    Position firstU0 = {-1, -1}, firstU1 = {-1, -1};

    // 1. the real fixture, in order, so cross-round state is exercised the way a game does
    for (int i = 0; i < NROUND; ++i) {
        GameInput in;
        std::memset(&in, 0, sizeof(in));
        in.round = i;
        std::memcpy(in.grid, ROUNDS[i].grid, sizeof(in.grid));
        in.my_units[0] = Position{ROUNDS[i].u0[0], ROUNDS[i].u0[1]};
        in.my_units[1] = Position{ROUNDS[i].u1[0], ROUNDS[i].u1[1]};
        in.visible_enemies[0] = Position{ROUNDS[i].enemy[0][0], ROUNDS[i].enemy[0][1]};
        in.visible_enemies[1] = Position{ROUNDS[i].enemy[1][0], ROUNDS[i].enemy[1][1]};
        in.snapshot.window_begin = -1;
        const GameOutput oa = a(&in);
        const GameOutput ob = b(&in);
        ++checked;
        if (!same(oa, ob)) {
            ++diff;
            if (firstDiff < 0) {
                firstDiff = i; firstA = oa; firstB = ob; firstRound = in.round;
                firstU0 = in.my_units[0]; firstU1 = in.my_units[1];
            }
        }
    }
    const long fixtureDiff = diff;

    // 2. random rule-shaped boards, same ordered sequence for both
    for (int i = 0; i < randomRounds; ++i) {
        GameInput in;
        randomInput(&in, i);
        const GameOutput oa = a(&in);
        const GameOutput ob = b(&in);
        ++checked;
        if (!same(oa, ob)) {
            ++diff;
            if (firstDiff < 0) {
                firstDiff = NROUND + i; firstA = oa; firstB = ob; firstRound = in.round;
                firstU0 = in.my_units[0]; firstU1 = in.my_units[1];
            }
        }
    }

    std::printf("A            %s\n", argv[1]);
    std::printf("B            %s\n", argv[2]);
    std::printf("fixture      %d rounds, %ld differ\n", NROUND, fixtureDiff);
    std::printf("random       %d rounds, %ld differ\n", randomRounds, diff - fixtureDiff);
    std::printf("total        %ld compared, %ld differ\n", checked, diff);
    if (diff) {
        std::printf("first差异 index %d round=%d u0=(%d,%d) u1=(%d,%d)\n",
                    firstDiff, firstRound, firstU0.row, firstU0.col,
                    firstU1.row, firstU1.col);
        printOutput("first A", firstA);
        printOutput("first B", firstB);
    }
    std::printf("%s\n", diff == 0 ? "IDENTICAL" : "DIFFERENT");
    dlclose(ha);
    dlclose(hb);
    return diff == 0 ? 0 : 1;
}
