`timescale 1ns / 1ps
module tb_instruction_memory;
 reg [5:0] addr;wire [31:0] rd;integer i;reg [31:0] expected;
 instruction_memory dut(addr,rd);
 initial begin
 for(i=0;i<64;i=i+1)begin addr=i;
 case(i)
 0:expected=32'h20010005;
 1:expected=32'h20020003;
 2:expected=32'h00221820;
 3:expected=32'h00222022;
 4:expected=32'h00222824;
 5:expected=32'h00223025;
 6:expected=32'hac030000;
 7:expected=32'h8c070000;
 default:expected=0;endcase
 #10;if(rd!==expected)$fatal(1,"ROM address %d",i);
 end
 $display("PASS tb_instruction_memory: all 64 initialized instruction words");$finish;
 end
endmodule
