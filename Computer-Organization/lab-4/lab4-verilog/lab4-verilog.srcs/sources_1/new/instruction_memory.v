`timescale 1ns / 1ps
module instruction_memory #(parameter INIT_FILE="imem.hex")(
 input wire [5:0] addr,output wire [31:0] rd);
 reg [31:0] mem[0:63];
 initial $readmemh(INIT_FILE,mem);
 assign rd=mem[addr];
endmodule
