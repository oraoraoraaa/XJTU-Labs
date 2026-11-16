`timescale 1ns / 1ps
module tb_alu32;
 reg en;reg [3:0] op;reg [31:0] a,b;
 wire [31:0] f,fb;wire zf,cf,of,sf,pf,zb,cb,ob,sb,pb;
 reg [31:0] expected;reg ec,eo,ep;reg [32:0] wide;
 reg signed [63:0] signed_result;
 integer i,j,k,seed=271799;
 alu32 dut(en,op,a,b,f,zf,cf,of,sf,pf);
 alu_bus bus_dut(en,op,a,b,fb,zb,cb,ob,sb,pb);
 task check;
 begin
  #10;
  if(!en) begin
   if({f,zf,cf,of,sf,pf}!==37'bz || {fb,zb,cb,ob,sb,pb}!==37'bz) $fatal(1,"disable failed");
  end else begin
   expected=0;ec=0;eo=0;wide=0;signed_result=0;
   case(op)
    0: expected=a&b; 1: expected=a|b; 2: expected=a^b; 3:expected=~(a|b);
    4:begin wide={1'b0,a}+{1'b0,b};expected=wide[31:0];ec=wide[32];
      signed_result=$signed({{32{a[31]}},a})+$signed({{32{b[31]}},b});end
    5:begin expected=a-b;ec=(a<b);
      signed_result=$signed({{32{a[31]}},a})-$signed({{32{b[31]}},b});end
    6:expected=(a<b);7:expected=(a>=32)?0:(b<<a);
   endcase
   if(op==4||op==5) eo=(signed_result>64'sd2147483647)||(signed_result< -64'sd2147483648);
   ep=0;for(k=0;k<32;k=k+1) ep=ep^expected[k];
   if({f,zf,cf,of,sf,pf}!=={expected,(expected==0),ec,eo,expected[31],ep})
    $fatal(1,"ALU failed op=%h a=%h b=%h f=%h",op,a,b,f);
   if({fb,zb,cb,ob,sb,pb}!=={f,zf,cf,of,sf,pf}) $fatal(1,"bus mismatch");
  end
 end endtask
 initial begin
  en=1;a=4;b=13;
  for(j=0;j<8;j=j+1)begin op=j;check;end
  op=4;a=32'h7fffffff;b=1;check;
  a=32'hffffffff;b=1;check;
  op=5;a=0;b=1;check;
  a=32'h80000000;b=1;check;
  op=6;a=32'hffffffff;b=0;check;
  op=7;b=32'h80000001;
  a=0;check;a=1;check;a=31;check;a=32;check;a=33;check;
  en=0;check;en=1;op=8;check;
  for(i=0;i<2000;i=i+1) begin
   a=$random(seed);b=$random(seed);
   for(j=0;j<16;j=j+1)begin op=j;check;end
  end
  $display("PASS tb_alu32: 20 directed + 32000 random opcode checks; all results/flags and bus equivalence");$finish;
 end
endmodule
