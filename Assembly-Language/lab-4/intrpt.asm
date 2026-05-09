; task1.asm
title Task 1 - Interrupt
data segment
    ID db '2234412799'
    NUM dw 0
    OLD_INT1C_OFF dw ?
    OLD_INT1C_SEG dw ?
data ends

code segment
    assume cs:code, ds:data
    main proc
        mov ax, seg data
        mov ds, ax
        
        ; Save old 1CH interrupt vector
        mov al, 1ch
        mov ah, 35h
        int 21h
        mov OLD_INT1C_OFF, bx
        mov OLD_INT1C_SEG, es
        
        ; Set new 1CH interrupt vector
        push ds
        mov ax, seg COUNT
        mov ds, ax
        mov dx, offset COUNT
        mov al, 1ch
        mov ah, 25h
        int 21h
        pop ds
        
    WAIT_INPUT:
        ; Wait for user input
        mov ah, 01h
        int 21h
        cmp al, 'Q'
        je EXIT
        cmp al, 'q'
        je EXIT
        jmp WAIT_INPUT
        
    EXIT:
        ; Restore old 1CH interrupt vector
        push ds
        mov dx, OLD_INT1C_OFF
        mov ax, OLD_INT1C_SEG
        mov ds, ax
        mov al, 1ch
        mov ah, 25h
        int 21h
        pop ds
        
        ; Display NUM in hex
        call PRINT_NUM
        
        ; Return to DOS
        mov ax, 4c00h
        int 21h
    main endp

    COUNT proc far
        push ax
        push ds
        mov ax, seg data
        mov ds, ax
        inc NUM
        pop ds
        pop ax
        iret
    COUNT endp

    PRINT_NUM proc
        ; Print word in NUM as hex
        mov ax, NUM
        mov cx, 4
    PRINT_LOOP:
        push cx
        mov cl, 4
        rol ax, cl
        pop cx
        push ax
        and al, 0fh
        cmp al, 9
        jle PRINT_DIGIT
        add al, 7
    PRINT_DIGIT:
        add al, 30h
        mov dl, al
        mov ah, 02h
        int 21h
        pop ax
        loop PRINT_LOOP
        
        ; Print 'h'
        mov dl, 'h'
        mov ah, 02h
        int 21h
        ret
    PRINT_NUM endp
code ends
end main