; task2.asm
title Task 2 - BIOS and DOS Interrupt

data segment
    ID db '2234412799'
    BUFFER db 10 dup (?)
    TABLE db 7, 5, 9, 1, 3, 6, 8, 0, 2, 4
data ends

code segment
    assume cs:code, ds:data
    main proc
        mov ax, data
        mov ds, ax
        
        mov cx, 10
        mov di, 0
        mov bx, offset TABLE
        
    READ_LOOP:
        ; Read char from keyboard
        mov ah, 01h
        int 21h
        
        ; Check for Enter (0Dh)
        cmp al, 0dh
        je END_INPUT
        
        ; Convert ASCII to number
        sub al, 30h
        
        ; Encrypt using XLAT
        xlat
        
        ; Store to BUFFER
        mov BUFFER[di], al
        inc di
        loop READ_LOOP
        
    END_INPUT:
        ; Return to DOS
        mov ax, 4c00h
        int 21h
    main endp
code ends
end main