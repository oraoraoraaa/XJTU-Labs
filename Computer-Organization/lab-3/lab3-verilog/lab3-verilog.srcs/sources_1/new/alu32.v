`timescale 1ns / 1ps
module alu_flags(input wire [3:0] op,input wire [31:0] a,b,f,
 input wire cf_in,output wire zf,cf,of,sf,pf);
 assign zf=(f==0); assign sf=f[31]; assign pf=^f;
 assign cf=((op==4)||(op==5)) ? cf_in:1'b0;
 assign of=(op==4) ? ((~(a[31]^b[31]))&(f[31]^a[31])):
           (op==5) ? ((a[31]^b[31])&(f[31]^a[31])):1'b0;
endmodule

module alu32(input wire en,input wire [3:0] op,input wire [31:0] a,b,
 output wire [31:0] f,output wire zf,cf,of,sf,pf);
 reg [31:0] result; reg [32:0] extended;
 wire z,c,o,s,p;
 always @(*) begin
  result=0; extended=0;
  case(op)
   0: result=a&b;
   1: result=a|b;
   2: result=a^b;
   3: result=~(a|b);
   4: begin extended={1'b0,a}+{1'b0,b}; result=extended[31:0]; end
   5: begin extended={1'b0,a}-{1'b0,b}; result=extended[31:0]; end
   6: result=(a<b)?32'd1:32'd0;
   7: result=b<<a;
   default: result=0;
  endcase
 end
 alu_flags flags(op,a,b,result,extended[32],z,c,o,s,p);
 assign f=en?result:32'bz;
 assign {zf,cf,of,sf,pf}=en?{z,c,o,s,p}:5'bzzzzz;
endmodule
