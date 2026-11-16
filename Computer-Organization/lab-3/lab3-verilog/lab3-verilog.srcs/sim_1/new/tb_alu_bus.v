`timescale 1ns / 1ps
module tb_alu_bus;
 reg en;reg [3:0] op;reg [31:0] a,b;
 wire [31:0] f;wire zf,cf,of,sf,pf;integer i;
 alu_bus dut(en,op,a,b,f,zf,cf,of,sf,pf);
 initial begin
  a=4;b=13;en=0;op=0;#10;
  if(f!==32'bz) $fatal(1,"bus not released");
  en=1;
  for(i=0;i<8;i=i+1)begin op=i;#10;
   if(dut.select!==(8'b1<<i)) $fatal(1,"decoder not one-hot");
  end
  en=0;#10;if({f,zf,cf,of,sf,pf}!==37'bz) $fatal(1,"disable");
  en=1;op=15;#10;if(f!==0||dut.select!==0) $fatal(1,"reserved opcode");
  $display("PASS tb_alu_bus: disable/re-enable, 8 one-hot selections, reserved opcode");$finish;
 end
endmodule
