`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11/11/2026 02:06:31 AM
// Design Name: 
// Module Name: tb_latch_vs_ff
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module tb_latch_vs_ff;
    reg clk;
    reg d;

    wire q_latch;
    wire q_ff;

    d_latch u_latch (
        .d(d), .en(clk), .q(q_latch)
    );

    d_ff u_ff (
        .d(d), .clk(clk), .q(q_ff)
    );

    initial clk = 0;
    always #5 clk = ~clk;  // 10 ns period

    initial begin
        #6;
        if ({q_latch,q_ff} !== 2'b00) $fatal(1,"Initial capture failed");
        #2;
        if ({q_latch,q_ff} !== 2'b10) $fatal(1,"Transparency check failed");
        #5;
        if ({q_latch,q_ff} !== 2'b10) $fatal(1,"Hold check failed");
        #3;
        if ({q_latch,q_ff} !== 2'b00) $fatal(1,"Edge capture failed");
        #2;
        if ({q_latch,q_ff} !== 2'b10) $fatal(1,"Second transparency failed");
        #5;
        if ({q_latch,q_ff} !== 2'b10) $fatal(1,"Second hold failed");
        #3;
        if ({q_latch,q_ff} !== 2'b00) $fatal(1,"Final capture failed");
        $display("PASS tb_latch_vs_ff: 7 latch/flip-flop timing checks");
    end

    initial begin
        d = 0;
        #7  d = 1;
        #5  d = 0;
        #5  d = 1;
        #5  d = 0;
        #8;

        $finish;
    end
endmodule
