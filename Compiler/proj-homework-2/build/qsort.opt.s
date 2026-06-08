	.data
	.balign 8
	.global ql_a
	.global _ql_a
ql_a:
_ql_a:
	.skip 80
	.text

	.global ql_swap
	.global _ql_swap
ql_swap:
_ql_swap:
	stp x29, x30, [sp, #-16]!
	mov x29, sp
	sub sp, sp, #112
	STR X0, [sp, #0]
	STR X1, [sp, #8]
	STR X2, [sp, #16]
	LDR X9, [sp, #8]
	MOV X10, #8
	MUL X9, X9, X10
	STR X9, [sp, #24]
	LDR X9, [sp, #0]
	LDR X10, [sp, #24]
	ADD X9, X9, X10
	STR X9, [sp, #32]
	LDR X9, [X9]
	STR X9, [sp, #40]
	STR X9, [sp, #48]
	LDR X9, [sp, #8]
	MOV X10, #8
	MUL X9, X9, X10
	STR X9, [sp, #56]
	LDR X9, [sp, #0]
	LDR X10, [sp, #56]
	ADD X9, X9, X10
	STR X9, [sp, #64]
	LDR X9, [sp, #16]
	MOV X10, #8
	MUL X9, X9, X10
	STR X9, [sp, #72]
	LDR X9, [sp, #0]
	LDR X10, [sp, #72]
	ADD X9, X9, X10
	STR X9, [sp, #80]
	LDR X9, [X9]
	STR X9, [sp, #88]
	LDR X9, [sp, #64]
	LDR X10, [sp, #88]
	STR X10, [X9]
	LDR X9, [sp, #16]
	MOV X10, #8
	MUL X9, X9, X10
	STR X9, [sp, #96]
	LDR X9, [sp, #0]
	LDR X10, [sp, #96]
	ADD X9, X9, X10
	STR X9, [sp, #104]
	LDR X10, [sp, #48]
	STR X10, [X9]
	MOV X0, #0
ql_swap_exit:
	add sp, sp, #112
	ldp x29, x30, [sp], #16
	ret

	.global ql_partition
	.global _ql_partition
ql_partition:
_ql_partition:
	stp x29, x30, [sp, #-16]!
	mov x29, sp
	sub sp, sp, #160
	STR X0, [sp, #0]
	STR X1, [sp, #8]
	STR X2, [sp, #16]
	LDR X9, [sp, #16]
	MOV X10, #8
	MUL X9, X9, X10
	STR X9, [sp, #24]
	LDR X9, [sp, #0]
	LDR X10, [sp, #24]
	ADD X9, X9, X10
	STR X9, [sp, #32]
	LDR X9, [X9]
	STR X9, [sp, #40]
	STR X9, [sp, #48]
	LDR X9, [sp, #8]
	MOV X10, #1
	SUB X9, X9, X10
	STR X9, [sp, #56]
	STR X9, [sp, #64]
	LDR X9, [sp, #8]
	MOV X10, #1
	SUB X9, X9, X10
	STR X9, [sp, #72]
	STR X9, [sp, #80]
l0:
	LDR X9, [sp, #80]
	MOV X10, #1
	ADD X9, X9, X10
	STR X9, [sp, #88]
	STR X9, [sp, #80]
	LDR X10, [sp, #16]
	CMP X9, X10
	B.LT l1
	B l2
l1:
	LDR X9, [sp, #80]
	MOV X10, #8
	MUL X9, X9, X10
	STR X9, [sp, #96]
	LDR X9, [sp, #0]
	LDR X10, [sp, #96]
	ADD X9, X9, X10
	STR X9, [sp, #104]
	LDR X9, [X9]
	STR X9, [sp, #112]
	LDR X10, [sp, #48]
	CMP X9, X10
	B.LT l3
	B l4
l3:
	LDR X9, [sp, #64]
	MOV X10, #1
	ADD X9, X9, X10
	STR X9, [sp, #120]
	STR X9, [sp, #64]
	LDR X0, [sp, #0]
	LDR X1, [sp, #64]
	LDR X2, [sp, #80]
	BL ql_swap
	STR X0, [sp, #128]
l4:
	B l0
l2:
	LDR X9, [sp, #64]
	MOV X10, #1
	ADD X9, X9, X10
	STR X9, [sp, #136]
	LDR X0, [sp, #0]
	LDR X1, [sp, #136]
	LDR X2, [sp, #16]
	BL ql_swap
	STR X0, [sp, #144]
	LDR X9, [sp, #64]
	MOV X10, #1
	ADD X9, X9, X10
	STR X9, [sp, #152]
	LDR X0, [sp, #152]
	B ql_partition_exit
	MOV X0, #0
ql_partition_exit:
	add sp, sp, #160
	ldp x29, x30, [sp], #16
	ret

	.global ql_qsort
	.global _ql_qsort
ql_qsort:
_ql_qsort:
	stp x29, x30, [sp, #-16]!
	mov x29, sp
	sub sp, sp, #80
	STR X0, [sp, #0]
	STR X1, [sp, #8]
	STR X2, [sp, #16]
	LDR X9, [sp, #8]
	LDR X10, [sp, #16]
	CMP X9, X10
	B.LT l5
	B l6
l5:
	LDR X0, [sp, #0]
	LDR X1, [sp, #8]
	LDR X2, [sp, #16]
	BL ql_partition
	STR X0, [sp, #24]
	LDR X9, [sp, #24]
	STR X9, [sp, #32]
	MOV X10, #1
	SUB X9, X9, X10
	STR X9, [sp, #40]
	LDR X0, [sp, #0]
	LDR X1, [sp, #8]
	LDR X2, [sp, #40]
	BL ql_qsort
	STR X0, [sp, #48]
	LDR X9, [sp, #32]
	MOV X10, #1
	ADD X9, X9, X10
	STR X9, [sp, #56]
	LDR X0, [sp, #0]
	LDR X1, [sp, #56]
	LDR X2, [sp, #16]
	BL ql_qsort
	STR X0, [sp, #64]
l6:
	MOV X0, #0
ql_qsort_exit:
	add sp, sp, #80
	ldp x29, x30, [sp], #16
	ret

	.global ql_entry
	.global _ql_entry
ql_entry:
_ql_entry:
	stp x29, x30, [sp, #-16]!
	mov x29, sp
	sub sp, sp, #16
	adrp X0, ql_a@PAGE
	add X0, X0, ql_a@PAGEOFF
	MOV X1, #0
	MOV X2, #9
	BL ql_qsort
	STR X0, [sp, #0]
	MOV X0, #0
ql_entry_exit:
	add sp, sp, #16
	ldp x29, x30, [sp], #16
	ret
