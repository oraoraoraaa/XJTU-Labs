; Task 2: conditional assembly + repeat assembly

title Conditional And Repeat Assembly Demo

data segment
    ID      db  '2186123456'      ; replace with your own student ID if needed

    ; Test scenario: only change the definition content of X.
    ; X db ''            ; scenario 1: empty string
    ; X db 'abcde'       ; scenario 2: length = 5
    X       db  '2186123456'      ; scenario 3: length = 10

    X_LEN   equ ($ - X)
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
