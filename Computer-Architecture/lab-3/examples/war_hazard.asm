# WAR hazard sample
# ADD.D reads F2 before a later instruction writes F2
R1 = 100
F2 = 2.0
F4 = 4.0
F6 = 6.0
F8 = 8.0
MEM[100] = 1.0
MEM[108] = 2.0

L.D F10, 0(R1)
ADD.D F12, F2, F4
MUL.D F2, F10, F6
DIV.D F14, F2, F8
S.D F14, 8(R1)
