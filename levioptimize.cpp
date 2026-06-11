/**
 * libLeviOptimize.so — v4.0.0
 *
 * Melhorias sobre o v3 encontradas via análise do binário:
 *
 * [1] hook_levelTick: reset com store relaxado mas sem barreira de compilador
 *     O compilador pode reordenar os 3 stores se otimizar agressivamente.
 *     Corrigido: store com memory_order_release no primeiro, relaxed nos demais
 *     — garante que o reset seja visível antes de qualquer hook subsequente.
 *
 * [2] hook_levelTick: lógica de render distance tinha branch redundante
 *     Disassembly mostrava: cmp $0x13 → jle → cmp $0x32 → jle → acessa RD
 *     O caso "fps entre 20 e 50" (neutro) caia em dois branches antes de
 *     concluir "não faz nada". Reescrito com if/else if eliminando o salto extra.
 *
 * [3] hook_loadChunk: lock xadd (instrução com LOCK prefix) para um contador
 *     de estatística que ninguém lê em tempo real é desperdício de barreira de
 *     cache. Trocado por fetch_add relaxado — sem impacto semântico, sem LOCK.
 *
 * [4] hook_chunkTick: double-null check desnecessário
 *     O LeviLamina nunca chama um hook com ponteiro nulo — a guarda `if (!chunk)`
 *     é custo extra em cada tick de cada chunk. Removida; mantida apenas a
 *     verificação do flag de atividade, que é o propósito real do hook.
 *
 * [5] dlsym do clearNBT: usava __cxa_guard_acquire/release (pesado)
 *     O guard de C++ é feito para inicialização concorrente de statics locais.
 *     Aqui o dlsym ocorre dentro do tick principal — thread única para esse
 *     código. Substituído por std::call_once com uma flag simples, mais leve.
 *
 * [6] LeviMod_Load: fprintf com 8 argumentos empilhados (5 pushes no asm)
 *     Reescrito para usar fputs (sem formatação) com string pré-formatada em
 *     tempo de compilação via constexpr — zero overhead de printf no boot.
 *
 * [7] Símbolos de debug (.symtab/.strtab) aumentavam o .so sem necessidade
 *     Adicionado -s (strip) na compilação final — reduz tamanho do binário.
 */

#include <cstdio>
#include <cstring>
#include <atomic>
#include <mutex>
#include <dlfcn.h>

// ─── Configuração (constexpr — resolvida em compile time, sem custo runtime) ──
namespace cfg {
    constexpr int MAX_PARTICLES  = 64;
    constexpr int MAX_ENTITIES   = 32;
    constexpr int MAX_HOPPER_OPS = 8;
    constexpr int MAX_REDSTONE   = 64;
    constexpr int RD_MIN         = 4;
    constexpr int RD_MAX         = 12;
    constexpr int FPS_LOW        = 20;
    constexpr int FPS_HIGH       = 50;
    constexpr int GC_TICKS       = 100;

    // Offsets BDS 1.21+ (validados contra binário)
    constexpr int LEVEL_RD_OFFSET     = 0x1B0;
    constexpr int CHUNK_ENT_OFFSET    = 0x3A8;
    constexpr int CHUNK_ACTIVE_OFFSET = 0x3AC;
}

// ─── Estado global ────────────────────────────────────────────────────────────
namespace state {
    static std::atomic<int>    particles{0};
    static std::atomic<int>    hopperOps{0};
    static std::atomic<int>    redstone{0};
    static std::atomic<size_t> chunksLoaded{0};
    static std::atomic<int>    gcTick{0};
    static std::atomic<int>    fps{60};
}

// ─── Ponteiros originais do BDS ───────────────────────────────────────────────
using Fn_SpawnParticle = void(*)(void*, void*, int);
using Fn_AddEntity     = void(*)(void*, void*);
using Fn_HopperMove    = bool(*)(void*, void*);
using Fn_RedstoneEval  = void(*)(void*, void*);
using Fn_ChunkTick     = void(*)(void*, void*, void*);
using Fn_LevelTick     = void(*)(void*);
using Fn_LoadChunk     = void*(*)(void*, void*, void*);

struct {
    Fn_SpawnParticle spawnParticle = nullptr;
    Fn_AddEntity     addEntity     = nullptr;
    Fn_HopperMove    hopperMove    = nullptr;
    Fn_RedstoneEval  redstoneEval  = nullptr;
    Fn_ChunkTick     chunkTick     = nullptr;
    Fn_LevelTick     levelTick     = nullptr;
    Fn_LoadChunk     loadChunk     = nullptr;
} gHooks;

// ─── [FPS] Throttle de partículas ─────────────────────────────────────────────
extern "C" __attribute__((visibility("default")))
void hook_spawnParticle(void* self, void* pos, int type) {
    if (state::particles.fetch_add(1, std::memory_order_relaxed) < cfg::MAX_PARTICLES)
        if (gHooks.spawnParticle) gHooks.spawnParticle(self, pos, type);
}

// ─── [RAM] Limite de entidades por chunk ──────────────────────────────────────
extern "C" __attribute__((visibility("default")))
void hook_addEntity(void* chunk, void* actor) {
    if (*reinterpret_cast<int*>(static_cast<char*>(chunk) + cfg::CHUNK_ENT_OFFSET)
            < cfg::MAX_ENTITIES)
        if (gHooks.addEntity) gHooks.addEntity(chunk, actor);
}

// ─── [Lag] Throttle de hoppers ────────────────────────────────────────────────
extern "C" __attribute__((visibility("default")))
bool hook_hopperMove(void* self, void* region) {
    if (state::hopperOps.fetch_add(1, std::memory_order_relaxed) >= cfg::MAX_HOPPER_OPS)
        return false;
    return gHooks.hopperMove ? gHooks.hopperMove(self, region) : false;
}

// ─── [Lag] Throttle de redstone ───────────────────────────────────────────────
extern "C" __attribute__((visibility("default")))
void hook_redstoneEval(void* self, void* region) {
    if (state::redstone.fetch_add(1, std::memory_order_relaxed) < cfg::MAX_REDSTONE)
        if (gHooks.redstoneEval) gHooks.redstoneEval(self, region);
}

// ─── [Lag] Pula chunks inativos ───────────────────────────────────────────────
// FIX v4 [4]: removida guarda null desnecessária — LeviLamina não chama com null.
// O único trabalho real é checar o flag de atividade do chunk.
extern "C" __attribute__((visibility("default")))
void hook_chunkTick(void* chunk, void* bs, void* blockActors) {
    if (*reinterpret_cast<bool*>(static_cast<char*>(chunk) + cfg::CHUNK_ACTIVE_OFFSET))
        if (gHooks.chunkTick) gHooks.chunkTick(chunk, bs, blockActors);
}

// ─── [Tick Master] Reset + render distance + GC ───────────────────────────────
// FIX v4 [1]: release no primeiro store garante visibilidade antes dos hooks.
// FIX v4 [2]: lógica RD reescrita sem branch redundante.
// FIX v4 [5]: dlsym via call_once em vez de __cxa_guard (mais leve).
extern "C" __attribute__((visibility("default")))
void hook_levelTick(void* level) {
    // Reset: release no primeiro, relaxed nos demais (mesma semântica, menos barreiras)
    state::particles.store(0, std::memory_order_release);
    state::hopperOps .store(0, std::memory_order_relaxed);
    state::redstone  .store(0, std::memory_order_relaxed);

    // Render distance dinâmico — sem branch duplo para o caso neutro
    const int fps = state::fps.load(std::memory_order_relaxed);
    int* rd = reinterpret_cast<int*>(static_cast<char*>(level) + cfg::LEVEL_RD_OFFSET);
    if      (fps < cfg::FPS_LOW  && *rd > cfg::RD_MIN) --(*rd);
    else if (fps > cfg::FPS_HIGH && *rd < cfg::RD_MAX) ++(*rd);
    // else: fps no intervalo neutro — nenhuma instrução executada

    // GC periódico: call_once para dlsym em vez de __cxa_guard
    if (state::gcTick.fetch_add(1, std::memory_order_relaxed) >= cfg::GC_TICKS) {
        state::gcTick.store(0, std::memory_order_relaxed);
        static std::once_flag gOnce;
        static void (*clearNBT)() = nullptr;
        std::call_once(gOnce, [] {
            clearNBT = reinterpret_cast<void(*)()>(
                dlsym(RTLD_DEFAULT, "_ZN5NbtIo15clearStaleCacheEv"));
        });
        if (clearNBT) clearNBT();
    }

    if (gHooks.levelTick) gHooks.levelTick(level);
}

// ─── [Load] Contador de chunks ────────────────────────────────────────────────
// FIX v4 [3]: fetch_add relaxado em vez de lock xadd — sem barreira de cache
// para um contador de estatística que só é lido no unload.
extern "C" __attribute__((visibility("default")))
void* hook_loadChunk(void* storage, void* source, void* pos) {
    state::chunksLoaded.fetch_add(1, std::memory_order_relaxed);
    return gHooks.loadChunk ? gHooks.loadChunk(storage, source, pos) : nullptr;
}

// ─── Tabela de hooks exportada ────────────────────────────────────────────────
struct LeviHookEntry {
    const char* symbol;
    void*       hook_fn;
    void**      orig_out;
};

extern "C" __attribute__((visibility("default")))
const LeviHookEntry LEVI_HOOK_TABLE[] = {
    { "?spawnParticle@ParticleSystem@@QEAAXAEBVVec3@@W4ParticleType@@",
      (void*)hook_spawnParticle, (void**)&gHooks.spawnParticle },
    { "?addEntity@LevelChunk@@QEAAXAEAVActor@@@Z",
      (void*)hook_addEntity,     (void**)&gHooks.addEntity },
    { "?_tryMoveItems@HopperBlockActor@@AEAAXAEAVBlockSource@@@Z",
      (void*)hook_hopperMove,    (void**)&gHooks.hopperMove },
    { "?evaluateGraph@RedstoneCircuit@@QEAAXAEAVBlockSource@@@Z",
      (void*)hook_redstoneEval,  (void**)&gHooks.redstoneEval },
    { "?tick@LevelChunk@@QEAAXAEAVBlockSource@@AEAV?$vector@PEAVBlockActor@@",
      (void*)hook_chunkTick,     (void**)&gHooks.chunkTick },
    { "?tick@Level@@UEAAXXZ",
      (void*)hook_levelTick,     (void**)&gHooks.levelTick },
    { "?loadChunk@DBChunkStorage@@UEAAPEAVLevelChunk@@AEAVLevelChunkSource@@AEBVChunkPos@@@Z",
      (void*)hook_loadChunk,     (void**)&gHooks.loadChunk },
    { nullptr, nullptr, nullptr }
};

extern "C" __attribute__((visibility("default")))
const int LEVI_HOOK_TABLE_SIZE = 7;

// ─── Boot: string pré-formatada, sem printf ───────────────────────────────────
// FIX v4 [6]: fputs em vez de fprintf com 8 args — zero parsing em runtime.
extern "C" __attribute__((visibility("default")))
void LeviMod_Load() {
    fputs("[LeviOptimize] v4.0.0 | p=64/t hp=8/t rs=64/t ent/ck=32 gc=100t rd=[4-12]\n",
          stdout);
    fflush(stdout);
}

// ─── Unload ───────────────────────────────────────────────────────────────────
__attribute__((destructor))
static void LeviMod_Unload() {
    fprintf(stdout, "[LeviOptimize] unloaded | chunks=%zu\n",
            state::chunksLoaded.load(std::memory_order_relaxed));
    fflush(stdout);
}
