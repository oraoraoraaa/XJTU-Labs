"""Throwaway probe: list the SimObject names this gem5 build actually exposes."""
import m5
from m5 import objects
from m5.objects import *

names = dir(objects)


def show(title, pred):
    hits = sorted(n for n in names if pred(n))
    print(f"--- {title} ({len(hits)}) ---")
    print(", ".join(hits))


show("CPU (Riscv)", lambda n: n.startswith("Riscv") and "CPU" in n)
show("BranchPredictor", lambda n: n.endswith("BP") or "TAGE" in n or "Perceptron" in n)
show("Prefetcher", lambda n: n.endswith("Prefetcher"))
show("ReplacementPolicy", lambda n: n.endswith("RP"))
show("Indexing", lambda n: "Indexing" in n)
show("MemCtrl/DRAM", lambda n: n.startswith("DDR") or n.startswith("LPDDR") or n.startswith("HBM"))

print("--- RiscvMMU attrs ---")
try:
    mmu = RiscvMMU()
    print([a for a in dir(mmu) if "walker" in a.lower() or "tlb" in a.lower()])
except Exception as exc:  # noqa: BLE001
    print("mmu probe failed:", exc)

print("--- RiscvO3CPU params ---")
try:
    cpu = RiscvO3CPU()
    interesting = [
        a
        for a in dir(cpu)
        if any(k in a for k in ("Width", "Entries", "numROB", "numPhys", "LQ", "SQ", "branchPred", "fuPool"))
    ]
    print(sorted(interesting))
except Exception as exc:  # noqa: BLE001
    print("cpu probe failed:", exc)
