`timescale 1ns / 1ps
module tb_register_file;
 reg clk=0,rst,we3;reg [4:0] ra1,ra2,wa3;reg [31:0] wd3;
 wire [31:0] rd1,rd2;reg [31:0] model[0:31];integer i,seed=12799;
 register_file dut(clk,rst,we3,ra1,ra2,wa3,wd3,rd1,rd2);
 always #5 clk=~clk;
 task check;begin #1;
 if(rd1!==model[ra1]||rd2!==model[ra2])$fatal(1,"register file mismatch");end endtask
 task put;input [4:0] a;input [31:0] d;
 begin @(negedge clk);wa3=a;wd3=d;we3=1;ra1=a;ra2=0;
 check; // Old value before write edge.
 @(posedge clk);if(a!=0)model[a]=d;check;end endtask
 initial begin
 rst=1;we3=0;ra1=0;ra2=0;wa3=0;wd3=0;
 for(i=0;i<32;i=i+1)model[i]=0;
 #1;check;@(negedge clk);rst=0;
 put(1,32'h12345678);put(2,32'habcdef01);put(0,32'hffffffff);
 @(negedge clk);we3=0;ra1=1;ra2=2;check;
 @(negedge clk);wa3=1;wd3=0;@(posedge clk);check;
 for(i=1;i<32;i=i+1)put(i,32'h01010101*i);
 for(i=0;i<1000;i=i+1)begin
 @(negedge clk);we3=$random(seed);wa3=$random(seed);wd3=$random(seed);
 ra1=$random(seed);ra2=$random(seed);check;
 @(posedge clk);if(we3&&wa3!=0)model[wa3]=wd3;check;
 end
 @(negedge clk);#2;rst=1;for(i=0;i<32;i=i+1)model[i]=0;check;
 for(i=0;i<32;i=i+1)begin ra1=i;ra2=31-i;check;end
 $display("PASS tb_register_file: all 32 registers, zero protection, two read ports, 1000 random cycles, reset");$finish;
 end
endmodule
