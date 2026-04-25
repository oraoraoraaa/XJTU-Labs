; Task 1: subroutine to count score ranges for 10 students

title Score Range Statistics

data segment
    ID      db  '2186123456'      ; replace with your own student ID if needed
    array   db  76,69,84,90,73,88,99,63,100,80

    S6      db  0                 ; 60~69
    S7      db  0                 ; 70~79
    S8      db  0                 ; 80~89
    S9      db  0                 ; 90~99
    S10     db  0                 ; 100
data ends

code segment
    assume cs:code, ds:data

    main    proc
        mov   ax, data
        mov   ds, ax

        call  STAT_SCORE

        mov   ax, 4c00h
        int   21h
    main    endp

    ; Count values in array into S6/S7/S8/S9/S10
    STAT_SCORE  proc
        mov   S6, 0
        mov   S7, 0
        mov   S8, 0
        mov   S9, 0
        mov   S10, 0

        lea   si, array
        mov   cx, 10

    NEXT_SCORE:
        mov   al, [si]

        cmp   al, 100
        je    IN_100

        cmp   al, 90
        jae   IN_90

        cmp   al, 80
        jae   IN_80

        cmp   al, 70
        jae   IN_70

        cmp   al, 60
        jae   IN_60

        jmp   SCORE_DONE

    IN_60:
        inc   S6
        jmp   SCORE_DONE

    IN_70:
        inc   S7
        jmp   SCORE_DONE

    IN_80:
        inc   S8
        jmp   SCORE_DONE

    IN_90:
        inc   S9
        jmp   SCORE_DONE

    IN_100:
        inc   S10

    SCORE_DONE:
        inc   si
        loop  NEXT_SCORE

        ret
    STAT_SCORE  endp

code ends
end main
