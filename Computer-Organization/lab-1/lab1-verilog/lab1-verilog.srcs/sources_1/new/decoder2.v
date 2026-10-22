`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10/21/2026 04:27:00 AM
// Design Name: 
// Module Name: decoder2
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


module decoder2(
    input wire [1:0] a,
    output wire [3:0] y
    );
   
    assign y[0] = ~a[0] & ~a[1];    
    assign y[1] =  a[0] & ~a[1];    
    assign y[2] = ~a[0] &  a[1];    
    assign y[3] =  a[0] &  a[1];   
     
endmodule
