`timescale 1ns / 1ps
// WIDTH >= 2. Active-high asynchronous clear, matching the slide's code.
module shift_register #(parameter WIDTH = 4)(
    input wire clk, rst,
    input wire [1:0] mode,
    input wire [WIDTH-1:0] d,
    input wire dsl, dsr,
    output reg [WIDTH-1:0] q
);
    always @(posedge clk or posedge rst) begin
        if (rst) q <= {WIDTH{1'b0}};
        else case (mode)
            2'b00: q <= q;
            2'b01: q <= {dsr, q[WIDTH-1:1]};
            2'b10: q <= {q[WIDTH-2:0], dsl};
            2'b11: q <= d;
        endcase
    end
endmodule
