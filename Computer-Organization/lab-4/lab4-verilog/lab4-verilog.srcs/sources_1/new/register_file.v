`timescale 1ns / 1ps
module register_file(input wire clk,rst,we3,
 input wire [4:0] ra1,ra2,wa3,input wire [31:0] wd3,
 output wire [31:0] rd1,rd2);
 reg [31:0] rf[0:31];integer i;
 always @(posedge clk or posedge rst)
  if(rst) begin for(i=0;i<32;i=i+1)rf[i]<=0;end
  else if(we3&&wa3!=0) rf[wa3]<=wd3;
 assign rd1=(ra1==0)?32'b0:rf[ra1];
 assign rd2=(ra2==0)?32'b0:rf[ra2];
endmodule
