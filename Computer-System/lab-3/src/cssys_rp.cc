/**
 * @file
 * Implementation of the Cssys replacement policy with evict_mru switching
 * and insertion/reuse instrumentation statistics.
 */

#include "mem/cache/replacement_policies/cssys_rp.hh"

#include <cassert>
#include <memory>

#include "params/CssysRP.hh"
#include "sim/cur_tick.hh"

namespace gem5
{

namespace replacement_policy
{

Cssys::Cssys(const Params &p)
  : Base(p), evictMru(p.evict_mru), cssysStats(this)
{
}

Cssys::CssysStats::CssysStats(statistics::Group *parent)
  : statistics::Group(parent),
    ADD_STAT(insertions, statistics::units::Count::get(),
             "Number of block insertions (reset calls)"),
    ADD_STAT(touches, statistics::units::Count::get(),
             "Number of cache hit updates (touch calls)"),
    ADD_STAT(victimizations, statistics::units::Count::get(),
             "Number of valid victims evicted"),
    ADD_STAT(deadOnArrival, statistics::units::Count::get(),
             "Number of victims evicted with zero hits after insertion"),
    ADD_STAT(victimRefSum, statistics::units::Count::get(),
             "Total touches accumulated by evicted victims"),
    ADD_STAT(firstTouchDelaySum, statistics::units::Tick::get(),
             "Sum of ticks from insertion to first hit"),
    ADD_STAT(firstTouchCount, statistics::units::Count::get(),
             "Number of blocks experiencing at least one hit"),
    ADD_STAT(refsPerInsertion, statistics::units::Rate<
             statistics::units::Count, statistics::units::Count>::get(),
             "Average touches per block insertion"),
    ADD_STAT(deadFraction, statistics::units::Ratio::get(),
             "Fraction of victims evicted without any hit"),
    ADD_STAT(avgFirstTouchDelay, statistics::units::Tick::get(),
             "Average ticks between insertion and first hit")
{
    refsPerInsertion = touches / insertions;
    deadFraction = deadOnArrival / victimizations;
    avgFirstTouchDelay = firstTouchDelaySum / firstTouchCount;
}

void
Cssys::invalidate(const std::shared_ptr<ReplacementData>& replacement_data)
{
    auto d = std::static_pointer_cast<CssysReplData>(replacement_data);
    d->lastTouchTick = Tick(0);
    d->insertTick = Tick(0);
    d->refCount = 0;
}

void
Cssys::touch(const std::shared_ptr<ReplacementData>& replacement_data) const
{
    auto d = std::static_pointer_cast<CssysReplData>(replacement_data);

    if (d->refCount == 0) {
        cssysStats.firstTouchDelaySum += curTick() - d->insertTick;
        cssysStats.firstTouchCount++;
    }
    d->refCount++;
    d->lastTouchTick = curTick();
    cssysStats.touches++;
}

void
Cssys::reset(const std::shared_ptr<ReplacementData>& replacement_data) const
{
    auto d = std::static_pointer_cast<CssysReplData>(replacement_data);
    d->lastTouchTick = curTick();
    d->insertTick = curTick();
    d->refCount = 0;
    cssysStats.insertions++;
}

ReplaceableEntry*
Cssys::getVictim(const ReplacementCandidates& candidates) const
{
    assert(candidates.size() > 0);

    ReplaceableEntry* victim = candidates[0];
    for (const auto& candidate : candidates) {
        auto cand_data = std::static_pointer_cast<CssysReplData>(
            candidate->replacementData);

        if (cand_data->lastTouchTick == 0) {
            victim = candidate;
            break;
        }

        Tick victim_tick = std::static_pointer_cast<CssysReplData>(
            victim->replacementData)->lastTouchTick;
        bool better = evictMru ? (cand_data->lastTouchTick > victim_tick)
                               : (cand_data->lastTouchTick < victim_tick);
        if (better) {
            victim = candidate;
        }
    }

    auto victim_data = std::static_pointer_cast<CssysReplData>(
        victim->replacementData);
    if (victim_data->lastTouchTick != 0) {
        cssysStats.victimizations++;
        cssysStats.victimRefSum += victim_data->refCount;
        if (victim_data->refCount == 0) {
            cssysStats.deadOnArrival++;
        }
    }

    return victim;
}

std::shared_ptr<ReplacementData>
Cssys::instantiateEntry()
{
    return std::shared_ptr<ReplacementData>(new CssysReplData());
}

} // namespace replacement_policy
} // namespace gem5
