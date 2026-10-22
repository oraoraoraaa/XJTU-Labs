`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10/21/2026 04:42:20 AM
// Design Name: 
// Module Name: tb_decoder2
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


`timescale 1ns/1ps

module tb_decoder2;

    reg  [1:0] a;
    wire [3:0] y;

    integer k;
    reg [3:0] expected;

    decoder2 dut (
        .a(a),
        .y(y)
    );

    initial begin
        a = 2'b00;

        for (k = 0; k < 4; k = k + 1) begin
            a = k;
            expected = 4'b0001 << k;
            #10;

            if (y !== expected) begin
                $display("FAIL: a=%b y=%b expected=%b",
                         a, y, expected);
                $fatal(1, "decoder2 failed");
            end
        end

        $display("PASS: decoder2, all 4 inputs");
        $finish;
    end

endmodule
