# RAW hazard sample
# F4 depends on the loaded value in F2
R1 = 100
F6 = 3.0
F8 = 5.0
MEM[100] = 8.0
MEM[108] = 4.0

L.D F2, 0(R1)
ADD.D F4, F2, F6
MUL.D F10, F4, F8
SUB.D F12, F10, F6
S.D F12, 8(R1)
