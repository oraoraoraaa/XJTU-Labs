`timescale 1ns / 1ps
module tb_full_adder;
 reg a,b,cin; wire sum,cout; integer i;
 full_adder dut(a,b,cin,sum,cout);
 initial begin
  for(i=0;i<8;i=i+1) begin
   {cin,a,b}=i; #10;
   if({cout,sum} !== ( {1'b0,a}+{1'b0,b}+cin)) $fatal(1,"full adder failed");
  end
  $display("PASS tb_full_adder: all 8 truth-table rows"); $finish;
 end
endmodule
