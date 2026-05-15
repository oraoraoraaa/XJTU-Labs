import argparse
import json
import shlex
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Tuple
from urllib.parse import urlparse


@dataclass
class Instruction:
    op: str
    args: Tuple
    text: str

    @staticmethod
    def nop() -> "Instruction":
        return Instruction("nop", tuple(), "nop")

    def is_nop(self) -> bool:
        return self.op == "nop"


@dataclass
class IFID:
    instr: Instruction
    pc: int


@dataclass
class IDEX:
    instr: Instruction
    pc: int
    rs: int
    rt: int
    rd: int
    imm: int
    rs_val: int
    rt_val: int


@dataclass
class EXMEM:
    instr: Instruction
    alu_result: int
    rt_val: int
    dest: int
    mem_read: bool
    mem_write: bool
    reg_write: bool
    mem_to_reg: bool
    branch_taken: bool
    branch_target: int


@dataclass
class MEMWB:
    instr: Instruction
    mem_data: int
    alu_result: int
    dest: int
    reg_write: bool
    mem_to_reg: bool


class MIPSPipelineSim:
    def __init__(self, forwarding: bool = True) -> None:
        self.forwarding = forwarding
        self.reset()

    def reset(self) -> None:
        self.regs = [0] * 32
        self.mem: Dict[int, int] = {}
        self.program: List[Instruction] = []
        self.labels: Dict[str, int] = {}
        self.pc = 0
        self.ifid = IFID(Instruction.nop(), 0)
        self.idex = IDEX(Instruction.nop(), 0, 0, 0, 0, 0, 0, 0)
        self.exmem = EXMEM(Instruction.nop(), 0, 0, 0, False, False, False, False, False, 0)
        self.memwb = MEMWB(Instruction.nop(), 0, 0, 0, False, False)
        self.cycles = 0
        self.completed = 0
        self.stalls = 0
        self.flushes = 0

    def load_program(self, text: str) -> None:
        self.reset()
        lines = text.splitlines()
        self.labels = self._collect_labels(lines)
        self.program = self._parse_instructions(lines)

    def _collect_labels(self, lines: List[str]) -> Dict[str, int]:
        labels: Dict[str, int] = {}
        pc = 0
        for raw in lines:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" in line:
                parts = [p.strip() for p in line.split(":")]
                label = parts[0]
                if label:
                    labels[label] = pc
                if parts[1]:
                    pc += 1
            else:
                pc += 1
        return labels

    def _parse_instructions(self, lines: List[str]) -> List[Instruction]:
        program: List[Instruction] = []
        for raw in lines:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" in line:
                label, rest = [p.strip() for p in line.split(":", 1)]
                line = rest
                if not line:
                    continue
            instr = self._parse_instruction(line)
            program.append(instr)
        return program

    def _parse_instruction(self, line: str) -> Instruction:
        tokens = [t.strip() for t in line.replace(",", " ").split()]
        if not tokens:
            return Instruction.nop()
        op = tokens[0].lower()
        if op == "add":
            rd = self._parse_reg(tokens[1])
            rs = self._parse_reg(tokens[2])
            rt = self._parse_reg(tokens[3])
            return Instruction(op, (rd, rs, rt), line)
        if op in ("lw", "sw"):
            rt = self._parse_reg(tokens[1])
            offset, rs = self._parse_mem(tokens[2])
            return Instruction(op, (rt, rs, offset), line)
        if op == "beqz":
            rs = self._parse_reg(tokens[1])
            label = tokens[2]
            if label not in self.labels:
                raise ValueError(f"Unknown label: {label}")
            target = self.labels[label]
            return Instruction(op, (rs, target), line)
        if op == "nop":
            return Instruction.nop()
        raise ValueError(f"Unsupported op: {op}")

    def _parse_reg(self, token: str) -> int:
        token = token.strip().lower()
        if token.startswith("$"):
            token = token[1:]
        if token.isdigit():
            reg = int(token)
            if 0 <= reg < 32:
                return reg
        raise ValueError(f"Invalid register: {token}")

    def _parse_mem(self, token: str) -> Tuple[int, int]:
        if "(" not in token or ")" not in token:
            raise ValueError(f"Invalid memory operand: {token}")
        offset_str, rest = token.split("(", 1)
        rs_str = rest.replace(")", "")
        offset = int(offset_str, 0)
        rs = self._parse_reg(rs_str)
        return offset, rs

    def _read_mem_word(self, addr: int) -> int:
        if addr % 4 != 0:
            raise ValueError(f"Unaligned memory access: {addr}")
        return self.mem.get(addr, 0)

    def _write_mem_word(self, addr: int, value: int) -> None:
        if addr % 4 != 0:
            raise ValueError(f"Unaligned memory access: {addr}")
        self.mem[addr] = value & 0xFFFFFFFF

    def _reg_read(self, reg: int) -> int:
        if reg == 0:
            return 0
        return self.regs[reg]

    def _reg_write(self, reg: int, value: int) -> None:
        if reg == 0:
            return
        self.regs[reg] = value & 0xFFFFFFFF

    def _instr_in_range(self, pc: int) -> bool:
        return 0 <= pc < len(self.program)

    def is_done(self) -> bool:
        if self.pc < len(self.program):
            return False
        return self.ifid.instr.is_nop() and self.idex.instr.is_nop() and self.exmem.instr.is_nop() and self.memwb.instr.is_nop()

    def step(self) -> None:
        if self.is_done():
            return

        self.cycles += 1

        old_ifid = self.ifid
        old_idex = self.idex
        old_exmem = self.exmem
        old_memwb = self.memwb

        # WB
        if old_memwb.reg_write and not old_memwb.instr.is_nop():
            value = old_memwb.mem_data if old_memwb.mem_to_reg else old_memwb.alu_result
            self._reg_write(old_memwb.dest, value)
            self.completed += 1

        # MEM
        mem_data = 0
        if not old_exmem.instr.is_nop():
            if old_exmem.mem_read:
                mem_data = self._read_mem_word(old_exmem.alu_result)
            if old_exmem.mem_write:
                self._write_mem_word(old_exmem.alu_result, old_exmem.rt_val)
        new_memwb = MEMWB(
            old_exmem.instr,
            mem_data,
            old_exmem.alu_result,
            old_exmem.dest,
            old_exmem.reg_write,
            old_exmem.mem_to_reg,
        )

        # EX
        branch_taken = False
        branch_target = 0
        alu_result = 0
        dest = 0
        mem_read = False
        mem_write = False
        reg_write = False
        mem_to_reg = False
        rt_forward_val = old_idex.rt_val

        if not old_idex.instr.is_nop():
            rs_val, rt_val = self._apply_forwarding(old_idex)
            rt_forward_val = rt_val
            if old_idex.instr.op == "add":
                rd, _, _ = old_idex.instr.args
                alu_result = (rs_val + rt_val) & 0xFFFFFFFF
                dest = rd
                reg_write = True
            elif old_idex.instr.op == "lw":
                rt, _, offset = old_idex.instr.args
                alu_result = (rs_val + offset) & 0xFFFFFFFF
                dest = rt
                mem_read = True
                reg_write = True
                mem_to_reg = True
            elif old_idex.instr.op == "sw":
                _, _, offset = old_idex.instr.args
                alu_result = (rs_val + offset) & 0xFFFFFFFF
                mem_write = True
            elif old_idex.instr.op == "beqz":
                rs, target = old_idex.instr.args
                if rs_val == 0:
                    branch_taken = True
                    branch_target = target
            elif old_idex.instr.op == "nop":
                pass
            else:
                raise ValueError(f"Unsupported op in EX: {old_idex.instr.op}")

        new_exmem = EXMEM(
            old_idex.instr,
            alu_result,
            rt_forward_val,
            dest,
            mem_read,
            mem_write,
            reg_write,
            mem_to_reg,
            branch_taken,
            branch_target,
        )

        # ID
        stall = self._should_stall(old_ifid, old_idex, old_exmem, old_memwb)
        if stall:
            self.stalls += 1
            new_idex = IDEX(Instruction.nop(), 0, 0, 0, 0, 0, 0, 0)
        else:
            new_idex = self._decode(old_ifid)

        # IF
        flush = False
        fetch_pc = self.pc
        next_pc = self.pc
        if branch_taken:
            next_pc = branch_target
            flush = True
            self.flushes += 1
        elif not stall:
            next_pc = self.pc + 1

        new_ifid = old_ifid
        if not stall:
            instr = Instruction.nop()
            if self._instr_in_range(fetch_pc):
                instr = self.program[fetch_pc]
            new_ifid = IFID(instr, fetch_pc)

        if flush:
            new_ifid = IFID(Instruction.nop(), next_pc)

        self.pc = next_pc
        self.ifid = new_ifid
        self.idex = new_idex
        self.exmem = new_exmem
        self.memwb = new_memwb

    def _decode(self, ifid: IFID) -> IDEX:
        instr = ifid.instr
        if instr.is_nop():
            return IDEX(Instruction.nop(), 0, 0, 0, 0, 0, 0, 0)
        rs = rt = rd = imm = 0
        if instr.op == "add":
            rd, rs, rt = instr.args
        elif instr.op in ("lw", "sw"):
            rt, rs, imm = instr.args
        elif instr.op == "beqz":
            rs, _ = instr.args
        else:
            raise ValueError(f"Unsupported op in ID: {instr.op}")
        return IDEX(instr, ifid.pc, rs, rt, rd, imm, self._reg_read(rs), self._reg_read(rt))

    def _apply_forwarding(self, idex: IDEX) -> Tuple[int, int]:
        rs_val = idex.rs_val
        rt_val = idex.rt_val
        if not self.forwarding:
            return rs_val, rt_val

        # EX/MEM forwarding (except load data)
        if self.exmem.reg_write and self.exmem.dest != 0 and not self.exmem.mem_to_reg:
            if self.exmem.dest == idex.rs:
                rs_val = self.exmem.alu_result
            if self.exmem.dest == idex.rt:
                rt_val = self.exmem.alu_result

        # MEM/WB forwarding
        if self.memwb.reg_write and self.memwb.dest != 0:
            wb_val = self.memwb.mem_data if self.memwb.mem_to_reg else self.memwb.alu_result
            if self.memwb.dest == idex.rs:
                rs_val = wb_val
            if self.memwb.dest == idex.rt:
                rt_val = wb_val

        return rs_val, rt_val

    def _should_stall(self, ifid: IFID, idex: IDEX, exmem: EXMEM, memwb: MEMWB) -> bool:
        instr = ifid.instr
        if instr.is_nop():
            return False

        rs, rt = self._get_source_regs(instr)

        # Load-use hazard (even with forwarding)
        if idex.instr.op == "lw":
            load_rt = idex.instr.args[0]
            if load_rt != 0 and (load_rt == rs or load_rt == rt):
                return True

        if self.forwarding:
            return False

        # No forwarding: stall on any RAW with results not yet written
        idex_dest = self._dest_reg(idex.instr)
        if self._writes_reg(idex.instr) and idex_dest != 0 and idex_dest in (rs, rt):
            return True
        if self._writes_reg(exmem.instr) and exmem.dest in (rs, rt):
            return True
        return False

    def _writes_reg(self, instr: Instruction) -> bool:
        if instr.is_nop():
            return False
        if instr.op in ("add", "lw"):
            return True
        return False

    def _dest_reg(self, instr: Instruction) -> int:
        if instr.is_nop():
            return 0
        if instr.op == "add":
            rd, _, _ = instr.args
            return rd
        if instr.op == "lw":
            rt, _, _ = instr.args
            return rt
        return 0

    def _get_source_regs(self, instr: Instruction) -> Tuple[int, int]:
        if instr.op == "add":
            _, rs, rt = instr.args
            return rs, rt
        if instr.op == "lw":
            _, rs, _ = instr.args
            return rs, 0
        if instr.op == "sw":
            rt, rs, _ = instr.args
            return rs, rt
        if instr.op == "beqz":
            rs, _ = instr.args
            return rs, 0
        return 0, 0

    def dump_pipeline(self) -> str:
        def fmt(instr: Instruction) -> str:
            return instr.text

        return (
            f"IF/ID: {fmt(self.ifid.instr)}\n"
            f"ID/EX: {fmt(self.idex.instr)}\n"
            f"EX/MEM: {fmt(self.exmem.instr)}\n"
            f"MEM/WB: {fmt(self.memwb.instr)}"
        )

    def dump_regs(self) -> str:
        lines = []
        for i in range(0, 32, 4):
            chunk = " ".join([f"${j:02d}={self.regs[j]:08x}" for j in range(i, i + 4)])
            lines.append(chunk)
        return "\n".join(lines)

    def dump_stats(self) -> str:
        instrs = self.completed
        cpi = (self.cycles / instrs) if instrs else 0.0
        return (
            f"Cycles: {self.cycles}\n"
            f"Instructions: {instrs}\n"
            f"Stalls: {self.stalls}\n"
            f"Flushes: {self.flushes}\n"
            f"CPI: {cpi:.2f}"
        )


class CLI:
    def __init__(self, sim: MIPSPipelineSim) -> None:
        self.sim = sim
        self.breakpoints: set[int] = set()

    def repl(self) -> None:
        print("MIPS 5-stage pipeline simulator. Type 'help' for commands.")
        while True:
            try:
                raw = input("mips> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not raw:
                continue
            parts = shlex.split(raw)
            cmd = parts[0].lower()
            args = parts[1:]
            try:
                if cmd in ("quit", "exit"):
                    break
                if cmd == "help":
                    self._help()
                elif cmd == "load":
                    self._cmd_load(args)
                elif cmd == "input":
                    self._cmd_input()
                elif cmd == "step":
                    self._cmd_step(args)
                elif cmd == "run":
                    self._cmd_run()
                elif cmd == "runto":
                    self._cmd_runto(args)
                elif cmd == "break":
                    self._cmd_break(args)
                elif cmd == "breaks":
                    self._cmd_breaks()
                elif cmd == "clear":
                    self._cmd_clear(args)
                elif cmd == "regs":
                    print(self.sim.dump_regs())
                elif cmd == "pipe":
                    print(self.sim.dump_pipeline())
                elif cmd == "mem":
                    self._cmd_mem(args)
                elif cmd == "memset":
                    self._cmd_memset(args)
                elif cmd == "stats":
                    print(self.sim.dump_stats())
                elif cmd == "forwarding":
                    self._cmd_forwarding(args)
                elif cmd == "reset":
                    self.sim.reset()
                    print("Simulator reset.")
                else:
                    print(f"Unknown command: {cmd}")
            except Exception as exc:
                print(f"Error: {exc}")

    def _help(self) -> None:
        print(
            "Commands:\n"
            "  load <file>           Load program from file\n"
            "  input                 Enter program from stdin, finish with .end\n"
            "  step [n]              Execute 1 (or n) cycles\n"
            "  run                   Run until breakpoint or program end\n"
            "  runto <pc|label>      Run until pc reaches target\n"
            "  break <pc|label>      Set breakpoint\n"
            "  breaks                List breakpoints\n"
            "  clear <pc|all>        Clear breakpoint\n"
            "  regs                  Show registers\n"
            "  mem <addr> <count>    Dump memory words\n"
            "  memset <addr> <value> Set one memory word\n"
            "  pipe                  Show pipeline registers\n"
            "  stats                 Show performance stats\n"
            "  forwarding on|off     Toggle data forwarding\n"
            "  reset                 Reset pipeline and state\n"
            "  quit                  Exit\n"
        )

    def _cmd_load(self, args: List[str]) -> None:
        if len(args) != 1:
            raise ValueError("Usage: load <file>")
        with open(args[0], "r", encoding="utf-8") as f:
            text = f.read()
        self.sim.load_program(text)
        print(f"Loaded {len(self.sim.program)} instructions.")

    def _cmd_input(self) -> None:
        print("Enter program, end with .end")
        lines = []
        while True:
            line = input()
            if line.strip() == ".end":
                break
            lines.append(line)
        self.sim.load_program("\n".join(lines))
        print(f"Loaded {len(self.sim.program)} instructions.")

    def _cmd_step(self, args: List[str]) -> None:
        count = int(args[0]) if args else 1
        for _ in range(count):
            if self.sim.is_done():
                print("Program complete.")
                break
            self.sim.step()
        print(f"PC={self.sim.pc}, cycles={self.sim.cycles}")

    def _cmd_run(self) -> None:
        while not self.sim.is_done():
            if self.sim.pc in self.breakpoints:
                print(f"Hit breakpoint at PC={self.sim.pc}")
                break
            self.sim.step()
        if self.sim.is_done():
            print("Program complete.")

    def _cmd_runto(self, args: List[str]) -> None:
        if len(args) != 1:
            raise ValueError("Usage: runto <pc|label>")
        target = self._resolve_target(args[0])
        while not self.sim.is_done():
            if self.sim.pc == target:
                print(f"Reached PC={self.sim.pc}")
                break
            self.sim.step()

    def _cmd_break(self, args: List[str]) -> None:
        if len(args) != 1:
            raise ValueError("Usage: break <pc|label>")
        target = self._resolve_target(args[0])
        self.breakpoints.add(target)
        print(f"Breakpoint set at PC={target}")

    def _cmd_breaks(self) -> None:
        if not self.breakpoints:
            print("No breakpoints.")
            return
        print("Breakpoints:")
        for bp in sorted(self.breakpoints):
            print(f"  PC={bp}")

    def _cmd_clear(self, args: List[str]) -> None:
        if len(args) != 1:
            raise ValueError("Usage: clear <pc|all>")
        if args[0] == "all":
            self.breakpoints.clear()
            print("All breakpoints cleared.")
            return
        target = self._resolve_target(args[0])
        self.breakpoints.discard(target)
        print(f"Breakpoint cleared at PC={target}")

    def _cmd_mem(self, args: List[str]) -> None:
        if len(args) != 2:
            raise ValueError("Usage: mem <addr> <count>")
        addr = int(args[0], 0)
        count = int(args[1], 0)
        for i in range(count):
            a = addr + i * 4
            val = self.sim.mem.get(a, 0)
            print(f"0x{a:08x}: 0x{val:08x}")

    def _cmd_memset(self, args: List[str]) -> None:
        if len(args) != 2:
            raise ValueError("Usage: memset <addr> <value>")
        addr = int(args[0], 0)
        value = int(args[1], 0)
        self.sim._write_mem_word(addr, value)
        print(f"0x{addr:08x}: 0x{value:08x}")

    def _cmd_forwarding(self, args: List[str]) -> None:
        if len(args) != 1:
            raise ValueError("Usage: forwarding on|off")
        if args[0] == "on":
            self.sim.forwarding = True
        elif args[0] == "off":
            self.sim.forwarding = False
        else:
            raise ValueError("Usage: forwarding on|off")
        print(f"Forwarding {'on' if self.sim.forwarding else 'off'}")

    def _resolve_target(self, token: str) -> int:
        if token.isdigit():
            return int(token)
        if token in self.sim.labels:
            return self.sim.labels[token]
        raise ValueError(f"Unknown target: {token}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MIPS 5-stage pipeline simulator")
    parser.add_argument("-f", "--file", help="Program file to load")
    parser.add_argument("--forwarding", choices=["on", "off"], default="on")
    args = parser.parse_args()

    sim = MIPSPipelineSim(forwarding=(args.forwarding == "on"))
    cli = CLI(sim)

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            sim.load_program(f.read())
        print(f"Loaded {len(sim.program)} instructions from {args.file}")

    cli.repl()


if __name__ == "__main__":
    main()
