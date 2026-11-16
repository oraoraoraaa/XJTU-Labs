`timescale 1ns / 1ps
module ripple_adder #(parameter WIDTH=32)(input wire [WIDTH-1:0] a,b,
 input wire cin, output wire [WIDTH-1:0] sum, output wire cout);
 wire [WIDTH:0] c;
 assign c[0]=cin; assign cout=c[WIDTH];
 genvar i;
 generate for(i=0;i<WIDTH;i=i+1) begin:g
  full_adder u(a[i],b[i],c[i],sum[i],c[i+1]);
 end endgenerate
endmodule

module cla4(input wire [3:0] a,b, input wire cin,
 output wire [3:0] sum, output wire cout);
 wire [3:0] p,g; wire [4:0] c;
 assign p=a^b; assign g=a&b; assign c[0]=cin;
 assign c[1]=g[0]|(p[0]&cin);
 assign c[2]=g[1]|(p[1]&g[0])|(p[1]&p[0]&cin);
 assign c[3]=g[2]|(p[2]&g[1])|(p[2]&p[1]&g[0])|(p[2]&p[1]&p[0]&cin);
 assign c[4]=g[3]|(p[3]&g[2])|(p[3]&p[2]&g[1])|
             (p[3]&p[2]&p[1]&g[0])|(p[3]&p[2]&p[1]&p[0]&cin);
 assign sum=p^c[3:0]; assign cout=c[4];
endmodule

// Four 4-bit blocks form a 16-bit hierarchical adder.
module adder16(input wire [15:0] a,b, input wire cin,
 output wire [15:0] sum, output wire cout);
 wire [4:0] c; assign c[0]=cin; assign cout=c[4];
 genvar k;
 generate for(k=0;k<4;k=k+1) begin:g
  ripple_adder #(.WIDTH(4)) u(a[k*4+:4],b[k*4+:4],c[k],sum[k*4+:4],c[k+1]);
 end endgenerate
endmodule
