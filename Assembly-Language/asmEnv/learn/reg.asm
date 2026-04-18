; empty asm file

title I love asm

data segment
    X  db  11H
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
        mov   bl, 28
        mov   cl, 5

        xor   ah, ah
        mov   al, bl
        div   cl

        xor   ah, ah
        shl   ax, 1
        mov   dx, ax
        
        ; ^ ---------------------------- ^
		; |          The END             |
        
        ; method 2: return to dos
        mov   ax, 4c00h
        int   21h
    main    endp
code ends
end main 