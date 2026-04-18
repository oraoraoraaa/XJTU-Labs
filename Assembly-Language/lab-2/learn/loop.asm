; loop.asm program

title I love asm

data segment
    ; Assuming student ID: 2234412799
    ; First 5 words: 22H, 34H, 41H, 27H, 99H
    ; n = 16, so 16 words in M
    M dw 22H, 34H, 41H, 27H, 99H, 0FFFFH, 7FFFH, 8000H, 0, 1, 2, 3, 4, 5, 6
    ; M+32 (2n) will hold the max absolute value
    ; M+36 (2(n+1)) will hold the offset address
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
        mov si, offset M  ; SI points to start of M
        mov cx, 16        ; n = 16
        mov bx, 0         ; BX will hold max absolute value
        mov di, 0         ; DI will hold offset of max
        
        ; Loop to find max absolute value
find_max:
        mov ax, [si]      ; Load current word
        ; Compute absolute value
        test ax, ax       ; Check if negative
        jns positive
        neg ax            ; If negative, negate
positive:
        cmp ax, bx        ; Compare with current max
        jle not_greater
        mov bx, ax        ; Update max
        mov di, si        ; Update offset
not_greater:
        add si, 2         ; Next word
        loop find_max
        
        ; Store max value at M+32
        mov si, offset M
        add si, 32
        mov [si], bx
        
        ; Store offset at M+36
        add si, 4
        mov [si], di
        
        ; ^ ---------------------------- ^
		; |          The END             |
        
        ; method 2: return to dos
        mov   ax, 4c00h
        int   21h
    main    endp
code ends
end main 