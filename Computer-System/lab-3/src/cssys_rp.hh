/**
 * @file
 * Declaration of the Cssys replacement policy supporting configurable
 * LRU/MRU victim selection and insertion/reuse instrumentation statistics.
 */

#ifndef __MEM_CACHE_REPLACEMENT_POLICIES_CSSYS_RP_HH__
#define __MEM_CACHE_REPLACEMENT_POLICIES_CSSYS_RP_HH__

#include <cstdint>

#include "base/statistics.hh"
#include "base/stats/group.hh"
#include "base/types.hh"
#include "mem/cache/replacement_policies/base.hh"

namespace gem5
{

struct CssysRPParams;

namespace replacement_policy
{

class Cssys : public Base
{
  protected:
    /** Per-line replacement metadata for policy order and reuse tracking. */
    struct CssysReplData : ReplacementData
    {
        /** Tick on which the entry was last accessed (0 means invalid/empty). */
        Tick lastTouchTick;
        /** Tick on which the entry was inserted via reset(). */
        Tick insertTick;
        /** Number of demand/hit touches since insertion. */
        uint64_t refCount;

        CssysReplData() : lastTouchTick(0), insertTick(0), refCount(0) {}
    };

    /** Runtime policy switch: false = LRU (<), true = MRU (>). */
    const bool evictMru;

    /**
     * Instrumentation statistics group. Declared mutable because the Base
     * replacement interface declares touch(), reset(), and getVictim() const.
     */
    struct CssysStats : public statistics::Group
    {
        CssysStats(statistics::Group *parent);

        statistics::Scalar insertions;
        statistics::Scalar touches;
        statistics::Scalar victimizations;
        statistics::Scalar deadOnArrival;
        statistics::Scalar victimRefSum;
        statistics::Scalar firstTouchDelaySum;
        statistics::Scalar firstTouchCount;

        statistics::Formula refsPerInsertion;
        statistics::Formula deadFraction;
        statistics::Formula avgFirstTouchDelay;
    };
    mutable CssysStats cssysStats;

  public:
    typedef CssysRPParams Params;
    Cssys(const Params &p);
    ~Cssys() = default;

    void invalidate(const std::shared_ptr<ReplacementData>& replacement_data)
                                                                    override;
    void touch(const std::shared_ptr<ReplacementData>& replacement_data) const
                                                                     override;
    void reset(const std::shared_ptr<ReplacementData>& replacement_data) const
                                                                     override;
    ReplaceableEntry* getVictim(const ReplacementCandidates& candidates) const
                                                                     override;
    std::shared_ptr<ReplacementData> instantiateEntry() override;
};

} // namespace replacement_policy
} // namespace gem5

#endif // __MEM_CACHE_REPLACEMENT_POLICIES_CSSYS_RP_HH__
