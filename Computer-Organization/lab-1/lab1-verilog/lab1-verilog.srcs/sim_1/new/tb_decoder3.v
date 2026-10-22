`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10/22/2026 04:19:36 PM
// Design Name: 
// Module Name: tb_decoder3
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


module tb_decoder3;

    reg  [2:0] test_input;
    wire [7:0] test_output;
    
    integer k;
    reg [7:0]  expected;
    
    decoder3 dut (
        .a(test_input),
        .y(test_output)
    );
    
    initial begin
    
        for (k = 0; k < 8; k = k + 1) begin
            test_input = k;
            expected = 8'b00000001 << k;
            #10;
            
            if (test_output !== expected) begin
                $display("FAIL: test_input = %b, test_output = %b, expected = %b", test_input, test_output, expected);
                $fatal(1, "decoder3 failed");
            end            
        end
    
        $display("ALL TESTS PASS");
        $finish;
    
    end 

endmodule
