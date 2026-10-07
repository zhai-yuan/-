// Compare several already-built player .so files: prove they decide IDENTICALLY, then time
// them against each other.
//
// Why a separate tool. Picking compiler flags is only meaningful if every variant computes the
// same thing, otherwise the fast one may simply be doing less. So equality is checked FIRST on
// all fixture rounds and any divergence is fatal. Timing is interleaved across variants within
// each repetition, so a drifting machine hits every variant equally instead of penalising
// whichever ran last.
//
// It loads real .so files through dlopen, which is what the engine does, rather than linking
// player.cpp into the harness and measuring a different binary.
//
// Build (on the Linux box):
//   g++ -O2 -std=c++17 -o bench_so bench_so.cpp -ldl -I.
// Run:
//   taskset -c 5 ./bench_so a.so b.so c.so
#include <cstdio>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <vector>
#include <string>
#include <dlfcn.h>

#include "game_api.h"
#include "rank1_rounds.h"

using MoveFn = GameOutput (*)(const GameInput*);

static const int WARM_N = 16;
static const int REPS = 21;
static const int PASSES = 200;

static void fillInput(GameInput* in, int i) {
    std::memset(in, 0, sizeof(GameInput));
    in->round = i + 1;
    std::memcpy(in->grid, ROUNDS[i].grid, sizeof(in->grid));
    in->my_units[0] = Position{ROUNDS[i].u0[0], ROUNDS[i].u0[1]};
    in->my_units[1] = Position{ROUNDS[i].u1[0], ROUNDS[i].u1[1]};
    in->visible_enemies[0] = Position{ROUNDS[i].enemy[0][0], ROUNDS[i].enemy[0][1]};
    in->visible_enemies[1] = Position{ROUNDS[i].enemy[1][0], ROUNDS[i].enemy[1][1]};
}

struct Variant {
    std::string path;
    void* handle = nullptr;
    MoveFn fn = nullptr;
    double warm = 1e18, cold = 1e18;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: %s [--noeq] <player1.so> [player2.so ...]\n", argv[0]);
        printf("  --noeq  skip the equality check, for timing DIFFERENT strategies against\n"
               "          each other. Only use it when the variants are meant to differ;\n"
               "          for compiler/flag comparisons the check must stay on.\n");
        return 2;
    }
    int first = 1;
    bool check_eq = true;
    if (std::strcmp(argv[1], "--noeq") == 0) { check_eq = false; first = 2; }
    std::vector<Variant> vs;
    for (int i = first; i < argc; ++i) {
        Variant v;
        v.path = argv[i];
        v.handle = dlopen(argv[i], RTLD_NOW | RTLD_LOCAL);
        if (!v.handle) { printf("dlopen %s: %s\n", argv[i], dlerror()); return 2; }
        dlerror();
        v.fn = (MoveFn)dlsym(v.handle, "moveDecision");
        if (!v.fn) { printf("dlsym %s: %s\n", argv[i], dlerror()); return 2; }
        vs.push_back(v);
    }

    static std::vector<GameInput> pool(NROUND);
    for (int i = 0; i < NROUND; ++i) fillInput(&pool[i], i);

    // ---- equality first: a faster variant that decides differently is not comparable ----
    // Each .so keeps cross-round state, so every variant must walk the fixture in the same
    // order from a fresh start. Round 0 resets that state inside the kernel.
    if (!check_eq)
        printf("equality check SKIPPED (--noeq): these variants are expected to differ\n\n");
    std::vector<std::vector<GameOutput>> outs(vs.size());
    if (check_eq) {
    printf("equality check over %d rounds x %zu variants\n", NROUND, vs.size());
    for (size_t k = 0; k < vs.size(); ++k) {
        outs[k].resize(NROUND);
        GameInput warmup;
        fillInput(&warmup, 0);
        warmup.round = 0;                       // clear the kernel's cross-round state
        vs[k].fn(&warmup);
        for (int i = 0; i < NROUND; ++i) outs[k][i] = vs[k].fn(&pool[i]);
    }
    long diffs = 0;
    for (size_t k = 1; k < vs.size(); ++k)
        for (int i = 0; i < NROUND; ++i) {
            const GameOutput& a = outs[0][i];
            const GameOutput& b = outs[k][i];
            bool same = a.k == b.k && a.order == b.order && a.vp == b.vp;
            for (int t = 0; t < S; ++t) if (a.actions[t] != b.actions[t]) same = false;
            if (!same) {
                if (diffs < 5)
                    printf("  DIVERGE round %d: %s vs %s\n", i, vs[0].path.c_str(),
                           vs[k].path.c_str());
                ++diffs;
            }
        }
    if (diffs) {
        printf("  %ld divergences -- variants are NOT comparable, aborting\n", diffs);
        return 1;
    }
    printf("  0 divergences, all variants decide identically\n\n");
    }   // end of the equality check

    // ---- interleaved timing ----
    static std::vector<int> order(NROUND);
    uint32_t rs = 99u;
    auto xs = [&]() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; };
    for (int i = 0; i < NROUND; ++i) order[i] = i;
    for (int i = NROUND - 1; i > 0; --i) std::swap(order[i], order[xs() % (i + 1)]);

    volatile long sink = 0;
    for (int rep = 0; rep < REPS; ++rep) {
        for (auto& v : vs) {
            for (int set = 0; set < 2; ++set) {
                const int n = set ? NROUND : WARM_N;
                auto t0 = std::chrono::steady_clock::now();
                long acc = 0;
                for (int p = 0; p < PASSES; ++p)
                    for (int i = 0; i < n; ++i) {
                        const GameOutput o = v.fn(&pool[order[i]]);
                        acc += o.k + o.actions[0];
                    }
                auto t1 = std::chrono::steady_clock::now();
                sink += acc;
                const double ns =
                    std::chrono::duration<double, std::nano>(t1 - t0).count() / (PASSES * n);
                if (set) v.cold = std::min(v.cold, ns);
                else     v.warm = std::min(v.warm, ns);
            }
        }
    }

    printf("%-42s %10s %10s\n", "variant", "WARM ns", "COLD ns");
    printf("%s\n", std::string(64, '-').c_str());
    double bw = 1e18, bc = 1e18;
    for (auto& v : vs) { bw = std::min(bw, v.warm); bc = std::min(bc, v.cold); }
    for (auto& v : vs) {
        const char* slash = strrchr(v.path.c_str(), '/');
        printf("%-42s %10.1f %10.1f   %+6.1f%% / %+6.1f%%\n",
               slash ? slash + 1 : v.path.c_str(), v.warm, v.cold,
               100.0 * (v.warm - bw) / bw, 100.0 * (v.cold - bc) / bc);
    }
    printf("\nbest WARM %.1f ns, best COLD %.1f ns, %d reps of %d passes, sink=%ld\n",
           bw, bc, REPS, PASSES, (long)sink);
    return 0;
}
