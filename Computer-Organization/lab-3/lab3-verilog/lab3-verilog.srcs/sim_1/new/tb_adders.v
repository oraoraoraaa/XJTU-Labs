`timescale 1ns / 1ps
module tb_adders;
 reg [31:0] a,b; reg cin;
 wire [31:0] s32; wire c32;
 wire [15:0] s16; wire c16;
 wire [3:0] s4,sc; wire c4,cc;
 reg [32:0] expected; integer i,j,k,seed=12345;
 ripple_adder u32(a,b,cin,s32,c32);
 adder16 u16(a[15:0],b[15:0],cin,s16,c16);
 ripple_adder #(.WIDTH(4)) u4(a[3:0],b[3:0],cin,s4,c4);
 cla4 uc(a[3:0],b[3:0],cin,sc,cc);
 task check;
 begin
  #10; expected={1'b0,a}+{1'b0,b}+cin;
  if({c32,s32}!==expected) $fatal(1,"32-bit adder");
  if({c16,s16}!==({1'b0,a[15:0]}+{1'b0,b[15:0]}+cin)) $fatal(1,"16-bit adder");
  if({c4,s4}!==({1'b0,a[3:0]}+{1'b0,b[3:0]}+cin)) $fatal(1,"4-bit adder");
  if({cc,sc}!=={c4,s4}) $fatal(1,"CLA mismatch");
 end endtask
 initial begin
  a=32'hffffffff;b=1;cin=0;check;
  a=32'h0000ffff;b=1;cin=0;check;
  a=32'h7fffffff;b=0;cin=1;check;
  a=0;b=0;cin=0;check;
  for(i=0;i<16;i=i+1) for(j=0;j<16;j=j+1) for(k=0;k<2;k=k+1) begin
   a=i;b=j;cin=k;check;
  end
  for(i=0;i<1000;i=i+1) begin a=$random(seed);b=$random(seed);cin=$random(seed);check;end
  $display("PASS tb_adders: 4 directed + 512 exhaustive + 1000 random; widths 4/16/32 and CLA4");$finish;
 end
endmodule
