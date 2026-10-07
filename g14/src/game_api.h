// game_api.h — GoldRush 2.0 contestant interface.
#pragma once

constexpr int GRID_SIZE = 17;
constexpr int MAX_NPCS = 7;
constexpr int S = 6;
constexpr int REGION_COUNT = 5;

struct Position {
    int row;
    int col;
};

struct NpcInfo {
    int id;
    Position pos;
};

struct RegionStat {
    int id;
    int enter;
    int leave;
    int gold_generated;
    int gold_collected;
    int gold_remaining;
    int occupants;
};

struct Snapshot {
    int window_begin;
    int window_end;
    RegionStat regions[REGION_COUNT];
};

struct GameInput {
    int round;
    int grid[GRID_SIZE][GRID_SIZE];
    Position my_units[2];
    int my_units_gold[2];
    int gold_opp;
    Position visible_enemies[2];
    int num_visible_npcs;
    NpcInfo visible_npcs[MAX_NPCS];
    int snapshot_valid;
    Snapshot snapshot;
};

struct GameOutput {
    int actions[S];
    int k;
    int order;
    int vp;
};

extern "C" GameOutput moveDecision(const GameInput* input);
