`timescale 1ns / 1ps
module tb_ram_inout;
 reg clk=0,rst,cs,re,we;reg [11:0] addr;reg [31:0] din;
 wire [31:0] dout;integer i;integer checks=0;
 always #5 clk=~clk;
wire [31:0] data;reg drive;
 assign data=drive?din:32'bz;
 assign dout=data;
 ram_4kx32_inout dut(clk,rst,cs,re,addr,data);
function [31:0] pattern;
 input integer a;begin pattern=32'hA55A1234 ^ (a*32'h00010001);end endfunction
 task read_check;
 input integer a;input [31:0] expected;
 begin addr=a;cs=1;re=1;we=0;drive=0;#1;
 if(dout!==expected)$fatal(1,"RAM read addr=%h got=%h expected=%h",addr,dout,expected);
 checks=checks+1;end endtask
 task write_word;
 input integer a;input [31:0] value;
 begin @(negedge clk);addr=a;din=value;cs=1;re=0;we=1;drive=1;
 @(posedge clk);#1;end endtask
 initial begin
 rst=1;cs=0;re=0;we=0;addr=0;din=0;drive=0;#1;
 @(negedge clk);rst=0;
 write_word(0,pattern(0));write_word(4095,pattern(4095));
 @(negedge clk);read_check(0,pattern(0));#9;
 read_check(4095,pattern(4095));#9;
 cs=0;drive=0;#1;if(dout!==32 'bz)$fatal(1,"CS release");#9;
 // Chip-select inhibits writes; read mode also inhibits writing.
 @(negedge clk);addr=0;din=0;cs=0;re=0;we=1;drive=1;
 @(posedge clk);#1;
 @(negedge clk);read_check(0,pattern(0));#9;
for(i=0;i<4096;i=i+1)write_word(i,pattern(i));
 @(negedge clk);
 for(i=0;i<4096;i=i+1)read_check(i,pattern(i));
 // Reset all words, including unselected banks, between clock edges.
 cs=0;drive=0;rst=1;#1;rst=0;
for(i=0;i<4096;i=i+1)read_check(i,0);
 $display("PASS tb_ram_inout: 4096 address writes/readback, full reset sweep, select and bus controls");$finish;
 end
endmodule
