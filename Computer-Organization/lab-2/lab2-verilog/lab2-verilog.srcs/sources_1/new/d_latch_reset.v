`timescale 1ns / 1ps
module d_latch_reset(input wire d, en, rst, output reg q, output wire qn);
    always @(*) begin
        if (rst) q <= 1'b0;
        else if (en) q <= d;
        // No else: level-sensitive storage is intentional.
    end
    assign qn = ~q;
endmodule
