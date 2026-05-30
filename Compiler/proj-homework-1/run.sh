echo "Generating ARM ASM file------------------------"
echo "python3 codegen.py < fact.qtac > fact_asm.s"
python3 codegen.py < fact.qtac > fact_asm.s
echo ""

echo "Assembling generated assembly------------------"
echo "gcc -c fact_asm.s -o fact_asm.o"
gcc -c fact_asm.s -o fact_asm.o
echo ""

echo "Compiling the C wrapper program----------------"
echo "gcc -c main.c -o main.o"
gcc -c main.c -o main.o
echo ""


echo "Linking into an executable---------------------"
echo "gcc fact_asm.o main.o -o fact_run"
gcc fact_asm.o main.o -o fact_run
echo ""

echo "Running the program----------------------------"
echo "./fact_run"
echo ""
echo "Result>>>>>>>>>>>"
./fact_run
echo "<<<<<<<<<<<<<<<<<"
echo ""

echo "Cleaning up------------------------------------"
echo "rm fact_asm.o && rm fact_asm.s && rm fact_run && rm main.o"
rm fact_asm.o && rm fact_asm.s && rm fact_run && rm main.o
echo ""
