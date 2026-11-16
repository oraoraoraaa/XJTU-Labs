`timescale 1ns / 1ps
module ram_4kx32(input wire clk,rst,cs,re,we,input wire [11:0] addr,
 input wire [31:0] din,output wire [31:0] dout);
 wire [3:0] bank_cs;
 genvar b;
 generate for(b=0;b<4;b=b+1)begin:bank
  assign bank_cs[b]=cs&&(addr[11:10]==b);
  ram_1kx16 lo(clk,rst,bank_cs[b],re,we,addr[9:0],din[15:0],dout[15:0]);
  ram_1kx16 hi(clk,rst,bank_cs[b],re,we,addr[9:0],din[31:16],dout[31:16]);
 end endgenerate
endmodule
