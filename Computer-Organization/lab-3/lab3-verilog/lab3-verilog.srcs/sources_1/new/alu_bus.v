`timescale 1ns / 1ps
// Each operation drives the shared bus only when selected.
module alu_bus(input wire en,input wire [3:0] op,input wire [31:0] a,b,
 output wire [31:0] f,output wire zf,cf,of,sf,pf);
 wire [7:0] select;
 wire [31:0] value[0:7];
 wire add_cout,sub_cout;
 wire [31:0] bus_data;
 assign value[0]=a&b; assign value[1]=a|b;
 assign value[2]=a^b; assign value[3]=~(a|b);
 ripple_adder add(a,b,1'b0,value[4],add_cout);
 ripple_adder sub(a,~b,1'b1,value[5],sub_cout);
 assign value[6]=(a<b)?32'd1:32'd0;
 assign value[7]=b<<a;
 genvar i;
 generate for(i=0;i<8;i=i+1) begin:g
  assign select[i]=en&&(op==i);
  assign bus_data=select[i]?value[i]:32'bz;
 end endgenerate
 // Reserved operations have a defined zero result while enabled.
 assign bus_data=(en&&op[3])?32'b0:32'bz;
 assign f=bus_data;
 wire carry=(op==4)?add_cout:(op==5)?~sub_cout:1'b0;
 wire z,c,o,s,p;
 alu_flags flags(op,a,b,bus_data,carry,z,c,o,s,p);
 assign {zf,cf,of,sf,pf}=en?{z,c,o,s,p}:5'bzzzzz;
endmodule
