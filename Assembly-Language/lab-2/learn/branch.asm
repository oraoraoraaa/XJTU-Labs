; empty asm file

title I love asm

data segment
    ; Student ID based string (length > 20)
    STR    db  '223441279901399999990831'
    STRLEN equ $ - STR

    ; COUNT[0]..COUNT[9] store occurrences for character '0'..'9'
    COUNT  db  10 dup(0)
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
        ; Count each digit's occurrences into COUNT array
        lea   si, STR
        mov   cx, STRLEN
    count_loop:
        mov   al, [si]
        cmp   al, '0'
        jb    next_char
        cmp   al, '9'
        ja    next_char
        sub   al, '0'
        xor   ah, ah
        mov   di, offset COUNT
        add   di, ax
        inc   byte ptr [di]
    next_char:
        inc   si
        loop  count_loop

        ; Find the digit with maximum count.
        ; If counts are equal, keep the larger digit.
        mov   si, offset COUNT
        mov   cx, 10
        xor   bl, bl        ; max count
        xor   bh, bh        ; max digit (0..9)
        xor   dl, dl        ; current digit index
    find_max:
        mov   al, [si]
        cmp   al, bl
        ja    update_max
        jb    skip_update
        cmp   dl, bh
        jbe   skip_update
    update_max:
        mov   bl, al
        mov   bh, dl
    skip_update:
        inc   si
        inc   dl
        loop  find_max

        ; Output format: digit,count   (example: 9,9)
        mov   dl, bh
        add   dl, '0'
        mov   ah, 02h
        int   21h

        mov   dl, ','
        mov   ah, 02h
        int   21h

        ; Print count in decimal (0..99)
        mov   al, bl
        xor   ah, ah
        mov   cl, 10
        div   cl            ; AL = tens, AH = ones
        cmp   al, 0
        je    print_ones
        add   al, '0'
        mov   dl, al
        mov   ah, 02h
        int   21h
    print_ones:
        mov   dl, ah
        add   dl, '0'
        mov   ah, 02h
        int   21h
        
        ; ^ ---------------------------- ^
		; |          The END             |
        
        ; method 2: return to dos
        mov   ax, 4c00h
        int   21h
    main    endp
code ends
end main 