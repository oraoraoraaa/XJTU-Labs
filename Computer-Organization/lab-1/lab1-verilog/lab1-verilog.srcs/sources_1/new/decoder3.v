`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10/22/2026 04:15:25 PM
// Design Name: 
// Module Name: decoder3
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


module decoder3(
    input  wire [2:0] a,
    output wire [7:0] y
    );
    
    assign y[0] = ~a[2] & ~a[1] & ~a[0];
    assign y[1] = ~a[2] & ~a[1] &  a[0];
    assign y[2] = ~a[2] &  a[1] & ~a[0];
    assign y[3] = ~a[2] &  a[1] &  a[0];
    assign y[4] =  a[2] & ~a[1] & ~a[0];
    assign y[5] =  a[2] & ~a[1] &  a[0];
    assign y[6] =  a[2] &  a[1] & ~a[0];
    assign y[7] =  a[2] &  a[1] &  a[0];
    
endmodule
