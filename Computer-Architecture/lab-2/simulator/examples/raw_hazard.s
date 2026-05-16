.text
main:
ADDIU $r8,$r0,INC
ADDIU $r9,$r0,VAL
ADDIU $r10,$r0,OUT
LW    $r3,0($r8)
LW    $r1,0($r9)
ADD   $r2,$r1,$r3
SW    $r2,0($r10)
TEQ   $r0,$r0

.data
VAL:  .word 9
INC:  .word 1
OUT:  .word 0