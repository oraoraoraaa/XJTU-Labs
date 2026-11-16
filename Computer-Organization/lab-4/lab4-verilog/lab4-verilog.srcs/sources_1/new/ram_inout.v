`timescale 1ns / 1ps
module ram_1kx16_inout(input wire clk,rst,cs,wr,input wire [9:0] addr,
 inout wire [15:0] data);
 // wr=1 reads; wr=0 writes. Writer must release bus before reading.
 ram_1kx16 core(clk,rst,cs,wr,!wr,addr,data,data);
endmodule
module ram_4kx32_inout(input wire clk,rst,cs,wr,input wire [11:0] addr,
 inout wire [31:0] data);
 wire [3:0] bank_cs;genvar b;
 generate for(b=0;b<4;b=b+1)begin:bank
  assign bank_cs[b]=cs&&(addr[11:10]==b);
  ram_1kx16_inout lo(clk,rst,bank_cs[b],wr,addr[9:0],data[15:0]);
  ram_1kx16_inout hi(clk,rst,bank_cs[b],wr,addr[9:0],data[31:16]);
 end endgenerate
endmodule
