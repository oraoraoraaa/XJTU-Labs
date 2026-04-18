; empty asm file

title I love asm

data segment
    ; Assuming student ID: 2234412799
    ; First 5 words: 22H, 34H, 41H, 27H, 99H
    ; n = 16, so 16 words total
    M dw 22H, 34H, 41H, 27H, 99H, 0FFFFH, 7FFFH, 8000H, 1234H, 5678H, 9ABCH, 0DEF0H, 1111H, 2222H, 3333H, 4444H
    ; M+2n = M[16], M+2(n+1)=M[18]
data ends

code segment
    assume cs:code, ds:data
    main    proc
        ; assign the data segment base address to DS
        mov   ax, data
        mov   ds, ax
        
        ; TODO ...
		; | add your code between arrows |
		; v ---------------------------- v
        
        ; Initialize
        mov cx, 16          ; n = 16
        lea si, M           ; SI points to M
        mov bx, 0           ; BX = index of max abs
        mov ax, [si]        ; AX = first element
        cmp ax, 0
        jge positive1
        neg ax              ; abs value
positive1:
        mov dx, ax          ; DX = current max abs
        
        ; Loop to find max abs
        mov di, 1           ; start from index 1
loop_start:
        cmp di, cx
        jge loop_end
        mov ax, [si + di*2] ; AX = M[di]
        cmp ax, 0
        jge positive
        neg ax              ; abs value
positive:
        cmp ax, dx
        jle not_greater
        mov dx, ax          ; update max abs
        mov bx, di          ; update index
not_greater:
        inc di
        jmp loop_start
loop_end:
        
        ; Store the max abs value at M+2n (M[16])
        mov [si + 32], dx   ; 2n=32
        
        ; Store the offset address at M+2(n+1) (M[18])
        ; Offset is bx*2 (since word array)
        mov ax, bx
        shl ax, 1           ; AX = bx * 2
        mov [si + 36], ax   ; 2(n+1)=36
        
        ; ^ ---------------------------- ^
		; |          The END             |
        
        ; method 2: return to dos
        mov   ax, 4c00h
        int   21h
    main    endp
code ends
end main 