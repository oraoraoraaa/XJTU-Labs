`timescale 1ns / 1ps
module tb_sequence_01;
    reg clk=0;
    reg rst, a;
    wire y_case, y_equations;
    reg previous_zero, expected;
    reg [6:0] sample;
    integer i, pattern, bit_index;
    sequence_01 u_case(.clk(clk),.rst(rst),.a(a),.y(y_case));
    sequence_01_equations u_equations(.clk(clk),.rst(rst),.a(a),.y(y_equations));
    always #5 clk=~clk;
    task send_bit;
        input value;
        begin
            @(negedge clk); a=value;
            expected=previous_zero && value;
            @(posedge clk); #1;
            if(y_case !== expected || y_equations !== expected ||
               u_case.state !== u_equations.state)
                $fatal(1,"Detector a=%b y=%b equation=%b expected=%b",a,y_case,y_equations,expected);
            previous_zero=~value;
        end
    endtask
    task reset_dut;
        begin
            @(negedge clk); rst=1; a=1; previous_zero=0;
            #1;
            if(y_case !== 0 || y_equations !== 0) $fatal(1,"Detector reset failed");
            @(negedge clk); rst=0;
        end
    endtask
    initial begin
        rst=1; a=1; previous_zero=0; expected=0; sample=7'b1001101;
        @(negedge clk); rst=0;
        // Handout example first, for a readable screenshot: Y=0001001.
        for(i=6;i>=0;i=i-1) send_bit(sample[i]);
        // All 64 six-bit streams, including repeated/overlapping 0101 cases.
        for(pattern=0;pattern<64;pattern=pattern+1) begin
            reset_dut;
            for(bit_index=5;bit_index>=0;bit_index=bit_index-1)
                send_bit((pattern>>bit_index)&1);
        end
        #4;
        $display("PASS tb_sequence_01: sample 1001101 + all 64 six-bit streams (384 bits)");
        $finish;
    end
endmodule
