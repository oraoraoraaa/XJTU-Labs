.text
main:
ADDIU $r8,$r0,FLAG
ADDIU $r9,$r0,X
ADDIU $r10,$r0,Y
ADDIU $r11,$r0,OUT
LW    $r1,0($r8)
LW    $r2,0($r9)
LW    $r3,0($r10)
BEQ   $r1,$r0,DO_ADD
ADD   $r4,$r0,$r0
SW    $r4,0($r11)
TEQ   $r0,$r0

DO_ADD:
ADD   $r4,$r2,$r3
SW    $r4,0($r11)
TEQ   $r0,$r0

.data
FLAG: .word 0
X:    .word 12
Y:    .word 30
OUT:  .word 0