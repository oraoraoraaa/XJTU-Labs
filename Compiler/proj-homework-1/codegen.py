import sys
import re

def main():
    if sys.stdin.isatty():
        print("Please pipe the qtac file into this script. e.g. python codegen.py < fact.qtac")
        return
    qtac_code = sys.stdin.read()
    
    # Preprocess
    qtac_code = qtac_code.replace('\n', ' ')
    instructions = [i.strip() for i in qtac_code.split(';') if i.strip()]
    
    reg_map = {
        'n': 'X0',
        'a': 'X1',
        't1': 'X2',
        't2': 'X3',
        't3': 'X4',
        't4': 'X5'
    }
    
    def get_op(op):
        if op == '+': return 'ADD'
        if op == '-': return 'SUB'
        if op == '*': return 'MUL'
        if op == '/': return 'SDIV'
        return 'UNKNOWN'

    def get_jrop(op):
        mapping = {
            '<': 'B.LT', '<=': 'B.LE', '>': 'B.GT', '>=': 'B.GE', '==': 'B.EQ', '!=': 'B.NE'
        }
        return mapping.get(op, 'B.EQ')
    
    asm = []
    asm.append(".global fact")
    asm.append(".global _fact")
    asm.append("fact:")
    asm.append("_fact:")
    
    i = 0
    while i < len(instructions):
        inst = instructions[i]
        
        # LABEL l
        m = re.match(r'^LABEL\s+(\w+)$', inst)
        if m:
            asm.append(f"{m.group(1)}:")
            i += 1
            continue
            
        # GOTO l
        m = re.match(r'^GOTO\s+(\w+)$', inst)
        if m:
            asm.append(f"\tB {m.group(1)}")
            i += 1
            continue
            
        # RETURN q
        m = re.match(r'^RETURN\s+(\w+)$', inst)
        if m:
            var = m.group(1)
            reg = reg_map.get(var, var)
            asm.append(f"\tMOV X0, {reg}\n\tRET")
            i += 1
            continue
            
        # IF qs rop qt THEN l1 ELSE l2
        m = re.match(r'^IF\s+(\w+)\s*([<>=!]+)\s*(\w+)\s+THEN\s+(\w+)\s+ELSE\s+(\w+)$', inst)
        if m:
            qs, rop, qt, l1, l2 = m.groups()
            rqs = reg_map.get(qs, qs)
            
            # Check if qt is immediate or register
            is_imm = qt.isdigit()
            rqt = f"#{qt}" if is_imm else reg_map.get(qt, qt)
            
            # check next instruction to optimize conditional jump
            next_inst = instructions[i+1] if i+1 < len(instructions) else ""
            m_next = re.match(r'^LABEL\s+(\w+)$', next_inst)
            if m_next and m_next.group(1) == l2:
                # IF qs < qt THEN l1 ELSE l2; LABEL l2
                asm.append(f"\tCMP {rqs}, {rqt}")
                asm.append(f"\t{get_jrop(rop)} {l1}")
                asm.append(f"{l2}:")
                i += 2
                continue
            elif m_next and m_next.group(1) == l1:
                # Need inverted logic (not fully implemented, skipping to general case)
                pass
                
            asm.append(f"\tCMP {rqs}, {rqt}")
            asm.append(f"\t{get_jrop(rop)} {l1}")
            asm.append(f"\tB {l2}")
            i += 1
            continue
            
        # qd = qs op qt or qd = qs op k
        m = re.match(r'^(\w+)\s*=\s*(\w+)\s*([\+\-\*\/])\s*(\w+)$', inst)
        if m:
            qd, qs, op, qt = m.groups()
            rqd = reg_map.get(qd, qd)
            rqs = reg_map.get(qs, qs)
            is_imm = qt.isdigit()
            rqt = f"#{qt}" if is_imm else reg_map.get(qt, qt)
            aop = get_op(op)
            asm.append(f"\t{aop} {rqd}, {rqs}, {rqt}")
            i += 1
            continue
            
        # qd = qs (move) or qd = k
        m = re.match(r'^(\w+)\s*=\s*(\w+)$', inst)
        if m:
            qd, qs = m.groups()
            rqd = reg_map.get(qd, qd)
            is_imm = qs.isdigit()
            rqs = f"#{qs}" if is_imm else reg_map.get(qs, qs)
            asm.append(f"\tMOV {rqd}, {rqs}")
            i += 1
            continue

        asm.append(f"\t// UNKNOWN: {inst}")
        i += 1
        
    for line in asm:
        print(line)

if __name__ == '__main__':
    main()