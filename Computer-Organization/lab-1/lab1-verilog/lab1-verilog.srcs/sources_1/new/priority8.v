`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10/22/2026 05:02:04 PM
// Design Name: 
// Module Name: priority8
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


module priority8(
    input  wire  [7:0] i,
    output reg   [2:0] y
    );
    
    always @* begin
    
        if      (i[7]) y = 3'd7;
        else if (i[6]) y = 3'd6;
        else if (i[5]) y = 3'd5;
        else if (i[4]) y = 3'd4;
        else if (i[3]) y = 3'd3;
        else if (i[2]) y = 3'd2;
        else if (i[1]) y = 3'd1;
        else           y = 3'd0;
        
    end
    
endmodule
