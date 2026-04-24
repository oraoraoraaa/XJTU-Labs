.text
main:
ADDIU $r8,$r0,A
ADDIU $r9,$r0,B
ADDIU $r10,$r0,C
ADDIU $r11,$r0,RES1
ADDIU $r12,$r0,RES2
LW    $r1,0($r8)
LW    $r2,0($r9)
LW    $r3,0($r10)
ADD   $r7,$r0,$r0
ADD   $r4,$r1,$r2
ADD   $r5,$r2,$r3
SW    $r4,0($r11)
SW    $r5,0($r12)
TEQ   $r0,$r0

.data
A:    .word 7
B:    .word 11
C:    .word 5
RES1: .word 0
RES2: .word 0