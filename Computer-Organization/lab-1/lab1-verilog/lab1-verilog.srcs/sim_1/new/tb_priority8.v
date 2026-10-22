`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10/23/2026 01:19:20 AM
// Design Name: 
// Module Name: tb_priority8
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


module tb_priority8;

    reg  [7:0] test_input;
    wire [2:0] test_output;
    reg  [2:0] expected;
    
    priority8 dut (
        .i(test_input),
        .y(test_output)
    );
    
    integer k;
    integer bit_index;
    
    initial begin
    
        expected = 3'b000;
    
        for (k = 0; k < 256; k = k + 1) begin
        
            test_input = k;
            
            // scan upward: the last asserted bit is the highest
            for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1) begin
                if (test_input[bit_index]) expected = bit_index;
            end
            
            #10;
            
            if (test_output !== expected) begin
                $display("FAIL: test_input = %b, test_output = %b, expected = %b", test_input, test_output, expected);
                $fatal(1, "priority8 failed");
            end      
        end
        
        $display("ALL TESTS PASSED");
        $finish;
    
    end

endmodule
