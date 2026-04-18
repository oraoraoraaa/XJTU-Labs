; empty asm file

title I love asm

data segment
    ; Student ID string: first 10 chars are 2234412799, then additional chars for testing
    STR db '22344127990012345678901234567890'  ; length > 20
    COUNT db 10 dup(0)  ; count for 0-9
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
        
        ; Count occurrences of 0-9 in STR
        lea si, STR
        lea di, COUNT
        mov cx, 30  ; length of STR
        
count_loop:
        lodsb       ; AL = [si], si++
        cmp al, '0'
        jb not_digit
        cmp al, '9'
        ja not_digit
        sub al, '0' ; AL = digit 0-9
        mov bl, al
        mov bh, 0
        inc byte ptr [di + bx]
not_digit:
        loop count_loop
        
        ; Find the digit with max count, if tie, largest digit
        mov al, 9   ; start from '9'
        mov bl, 0   ; max count
        mov cl, al  ; current max digit
        
find_max:
        cmp al, 0
        jl find_end
        mov bh, 0
        mov bl, al
        mov dl, [di + bx]  ; DL = count
        cmp dl, bl  ; BL is max count so far
        jle not_max
        mov bl, dl  ; update max count
        mov cl, al  ; update max digit
not_max:
        dec al
        jmp find_max
find_end:
        
        ; Output: digit, count
        mov dl, cl
        add dl, '0' ; to ASCII
        mov ah, 2
        int 21h     ; output digit
        
        mov dl, ','
        int 21h
        
        mov dl, bl
        add dl, '0' ; assume count < 10
        int 21h
        
        ; ^ ---------------------------- ^
		; |          The END             |
        
        ; method 2: return to dos
        mov   ax, 4c00h
        int   21h
    main    endp
code ends
end main 