`timescale 1ns / 1ps
// Synchronous write, asynchronous read and active-high asynchronous clear.
module ram_1kx16(input wire clk,rst,cs,re,we,input wire [9:0] addr,
 input wire [15:0] din,output wire [15:0] dout);
 reg [15:0] mem[0:1023];integer i;
 always @(posedge clk or posedge rst)
  if(rst) begin for(i=0;i<1024;i=i+1) mem[i]<=0;end
  else if(cs&&we&&!re) mem[addr]<=din;
 assign dout=(cs&&re&&!we&&!rst)?mem[addr]:16'bz;
endmodule
