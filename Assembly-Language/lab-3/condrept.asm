title Conditional And Repeat Assembly Demo

data segment
    ID      db  '2234412799'

    ;X db 0                 ; scenario 1: empty string
    X db 'abcde',0       ; scenario 2: length = 5
    ; X db '2234412799',0  ; scenario 3: length = 10

    X_LEN   equ ($ - X - 1)
    RESULT  dw  0
data ends

code segment
    assume cs:code, ds:data

    main    proc
        mov   ax, data
        mov   ds, ax

        mov   ax, 1

        if X_LEN le 5
            rept X_LEN
                add ax, ax
            endm
        else
            rept 6
                add ax, ax
            endm
        endif

        mov   RESULT, ax

        mov   ax, 4c00h
        int   21h
    main    endp

code ends
end main
