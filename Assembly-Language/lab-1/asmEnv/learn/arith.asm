; empty asm file

title I love asm

data segment
    stu db  "2234412799"
    X   db  5
    Y   db  6
    Z   db  0
    W   db  20
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
        mov   al, W
        sub   al, X
        cbw

        mov   bl, 5
        idiv  bl

        imul  Y

        mov   bl, 2
        idiv  bl

        mov   Z, al
        
        ; ^ ---------------------------- ^
		; |          The END             |
        
        ; method 2: return to dos
        mov   ax, 4c00h
        int   21h
    main    endp
code ends
end main 