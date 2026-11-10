`timescale 1ns / 1ps
// OE=1 disables the external output, without erasing the stored data.
module register_n #(parameter WIDTH = 8)(
    input wire clk, oe,
    input wire [WIDTH-1:0] d,
    output wire [WIDTH-1:0] q
);
    reg [WIDTH-1:0] stored;
    always @(posedge clk) stored <= d;
    assign q = oe ? {WIDTH{1'bz}} : stored;
endmodule
