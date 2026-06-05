# No hazard sample
# Independent operations to demonstrate a clean Tomasulo schedule
R1 = 100
F0 = 1.0
F2 = 2.0
F4 = 3.0
F6 = 4.0
F8 = 5.0
F10 = 6.0
F12 = 7.0
MEM[100] = 11.0
MEM[108] = 22.0
MEM[116] = 33.0

L.D F14, 0(R1)
L.D F16, 8(R1)
ADD.D F18, F2, F4
SUB.D F20, F6, F0
MUL.D F22, F8, F10
DIV.D F24, F12, F2
S.D F14, 16(R1)
