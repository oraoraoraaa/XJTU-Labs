// ============================================================================
//  Cache Simulator (Simulator A) - Computer Architecture Lab 4
//
//  A configurable set-associative cache simulator.
//    * Replacement policy : LRU
//    * Write-miss policy   : write-allocate
//    * Write-hit policy    : write-back (does not affect miss-rate statistics)
//
//  Trace format (Dinero "din"):  <access_type> <address_hex> [size/data]
//    access_type 0 = load  data  (read)
//    access_type 1 = store data  (write)
//    access_type 2 = instruction fetch  (ignored, per lab requirement H)
//
//  Build : g++ -O2 -std=c++17 -o cache_sim src/cache_sim.cpp
//  Usage : ./cache_sim --trace <file> --cache_size <bytes>
//                      --assoc <n> --block_size <bytes> [--csv] [--verbose]
// ============================================================================
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

// ---------------------------------------------------------------------------
// One cache line: just a tag plus a validity flag.  LRU order inside a set is
// kept implicitly by the position in the `ways` vector (front = MRU).
// ---------------------------------------------------------------------------
struct Line {
    uint64_t tag = 0;
    bool     valid = false;
};

class Cache {
public:
    Cache(uint64_t cacheSize, uint32_t blockSize, uint32_t assoc)
        : blockSize_(blockSize), assoc_(assoc) {
        numBlocks_ = cacheSize / blockSize;
        numSets_   = numBlocks_ / assoc;
        offsetBits_ = log2u(blockSize);
        indexBits_  = log2u(numSets_);
        // sets_[s] is an LRU-ordered list of ways; front = most-recently-used.
        sets_.assign(numSets_, std::vector<Line>(assoc));
    }

    // Process a single memory reference.  isWrite = true for stores.
    void access(uint64_t addr, bool isWrite) {
        uint64_t blockAddr = addr >> offsetBits_;
        uint64_t index = numSets_ > 1 ? (blockAddr & (numSets_ - 1)) : 0;
        uint64_t tag   = blockAddr >> indexBits_;

        if (isWrite) ++writes_; else ++reads_;

        auto& set = sets_[index];
        // Look for the tag among the valid ways of this set.
        for (size_t i = 0; i < set.size(); ++i) {
            if (set[i].valid && set[i].tag == tag) {   // ---- HIT ----
                moveToFront(set, i);                    // update LRU order
                return;
            }
        }
        // ---- MISS ---- (write-allocate: a miss always brings the block in)
        if (isWrite) ++writeMisses_; else ++readMisses_;

        // Is the set already full of valid lines? Then we must evict the LRU.
        if (set.back().valid) ++replacements_;

        // Insert the new block at the front (MRU); the back is dropped/reused.
        set.pop_back();
        set.insert(set.begin(), Line{tag, true});
    }

    // ----- accessors used for reporting -----
    uint64_t reads()        const { return reads_; }
    uint64_t writes()       const { return writes_; }
    uint64_t readMisses()   const { return readMisses_; }
    uint64_t writeMisses()  const { return writeMisses_; }
    uint64_t replacements() const { return replacements_; }
    uint64_t numSets()      const { return numSets_; }

private:
    static uint32_t log2u(uint64_t x) {
        uint32_t b = 0;
        while (x > 1) { x >>= 1; ++b; }
        return b;
    }
    // Promote way i to the front of the LRU order.
    static void moveToFront(std::vector<Line>& set, size_t i) {
        Line tmp = set[i];
        set.erase(set.begin() + i);
        set.insert(set.begin(), tmp);
    }

    uint32_t blockSize_, assoc_;
    uint64_t numBlocks_, numSets_;
    uint32_t offsetBits_, indexBits_;
    std::vector<std::vector<Line>> sets_;

    uint64_t reads_ = 0, writes_ = 0;
    uint64_t readMisses_ = 0, writeMisses_ = 0;
    uint64_t replacements_ = 0;
};

// ---------------------------------------------------------------------------
static bool isPow2(uint64_t x) { return x && !(x & (x - 1)); }

static void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s --trace <file> --cache_size <bytes> "
        "--assoc <n> --block_size <bytes> [--csv] [--verbose]\n"
        "  Sizes accept a K/M suffix, e.g. --cache_size 32K --block_size 64\n",
        prog);
}

// Parse a size that may carry a K or M suffix (e.g. "32K", "16384").
static uint64_t parseSize(const std::string& s) {
    char suf = s.empty() ? 0 : s.back();
    uint64_t mul = 1;
    std::string num = s;
    if (suf == 'K' || suf == 'k') { mul = 1024ULL;            num.pop_back(); }
    else if (suf == 'M' || suf == 'm') { mul = 1024ULL * 1024; num.pop_back(); }
    return std::strtoull(num.c_str(), nullptr, 10) * mul;
}

int main(int argc, char** argv) {
    std::string tracePath;
    uint64_t cacheSize = 0, blockSize = 0;
    uint32_t assoc = 0;
    bool csv = false, verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(argv[0]); std::exit(1); }
            return argv[++i];
        };
        if      (a == "--trace")       tracePath = next();
        else if (a == "--cache_size")  cacheSize = parseSize(next());
        else if (a == "--assoc")       assoc     = (uint32_t)std::strtoul(next().c_str(), nullptr, 10);
        else if (a == "--block_size")  blockSize = parseSize(next());
        else if (a == "--csv")         csv = true;
        else if (a == "--verbose")     verbose = true;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "Unknown argument: %s\n", a.c_str()); usage(argv[0]); return 1; }
    }

    // ----- validate parameters -----
    if (tracePath.empty() || !cacheSize || !blockSize || !assoc) {
        usage(argv[0]); return 1;
    }
    if (!isPow2(cacheSize) || !isPow2(blockSize) || !isPow2(assoc)) {
        std::fprintf(stderr, "Error: cache_size, block_size and assoc must be powers of two.\n");
        return 1;
    }
    if (blockSize > cacheSize || (cacheSize / blockSize) % assoc != 0) {
        std::fprintf(stderr, "Error: incompatible geometry "
            "(need block_size <= cache_size and (cache_size/block_size) %% assoc == 0).\n");
        return 1;
    }

    std::ifstream in(tracePath);
    if (!in) { std::fprintf(stderr, "Error: cannot open trace '%s'\n", tracePath.c_str()); return 1; }

    Cache cache(cacheSize, (uint32_t)blockSize, assoc);

    // ----- stream the trace; only loads (0) and stores (1) are simulated -----
    uint64_t lines = 0;
    int type; std::string addrHex;
    std::string rest;
    while (in >> type >> addrHex) {
        std::getline(in, rest);            // discard size/data field
        ++lines;
        if (type == 0)      cache.access(std::strtoull(addrHex.c_str(), nullptr, 16), false);
        else if (type == 1) cache.access(std::strtoull(addrHex.c_str(), nullptr, 16), true);
        // type == 2 (instruction fetch) and anything else: ignored
    }

    uint64_t reads = cache.reads(), writes = cache.writes();
    uint64_t rMiss = cache.readMisses(), wMiss = cache.writeMisses();
    uint64_t accesses = reads + writes, misses = rMiss + wMiss;
    auto rate = [](uint64_t m, uint64_t n) { return n ? 100.0 * (double)m / (double)n : 0.0; };

    if (csv) {
        // trace,cache_size,assoc,block_size,reads,writes,read_miss,write_miss,
        // replacements,read_miss_rate,write_miss_rate,total_miss_rate
        std::printf("%s,%llu,%u,%llu,%llu,%llu,%llu,%llu,%llu,%.4f,%.4f,%.4f\n",
            tracePath.c_str(),
            (unsigned long long)cacheSize, assoc, (unsigned long long)blockSize,
            (unsigned long long)reads, (unsigned long long)writes,
            (unsigned long long)rMiss, (unsigned long long)wMiss,
            (unsigned long long)cache.replacements(),
            rate(rMiss, reads), rate(wMiss, writes), rate(misses, accesses));
    } else {
        std::printf("==================== Cache Simulation Result ====================\n");
        std::printf(" Trace file       : %s\n", tracePath.c_str());
        std::printf(" Cache size       : %llu B (%.0f KB)\n",
                    (unsigned long long)cacheSize, cacheSize / 1024.0);
        std::printf(" Block size       : %llu B\n", (unsigned long long)blockSize);
        std::printf(" Associativity    : %u-way\n", assoc);
        std::printf(" Number of sets   : %llu\n", (unsigned long long)cache.numSets());
        std::printf(" Replacement      : LRU      Write-miss: write-allocate\n");
        std::printf("-----------------------------------------------------------------\n");
        std::printf(" Trace lines read : %llu\n", (unsigned long long)lines);
        std::printf(" Data accesses    : %llu  (read %llu, write %llu)\n",
                    (unsigned long long)accesses,
                    (unsigned long long)reads, (unsigned long long)writes);
        std::printf("-----------------------------------------------------------------\n");
        std::printf(" Read  misses     : %llu\n", (unsigned long long)rMiss);
        std::printf(" Write misses     : %llu\n", (unsigned long long)wMiss);
        std::printf(" Replaced blocks  : %llu\n", (unsigned long long)cache.replacements());
        std::printf("-----------------------------------------------------------------\n");
        std::printf(" Read  miss rate  : %.4f %%\n", rate(rMiss, reads));
        std::printf(" Write miss rate  : %.4f %%\n", rate(wMiss, writes));
        std::printf(" Total miss rate  : %.4f %%\n", rate(misses, accesses));
        std::printf("=================================================================\n");
        (void)verbose;
    }
    return 0;
}
