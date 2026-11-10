`timescale 1ns / 1ps
module tb_shift_register;
    reg clk=0;
    reg rst, dsl, dsr;
    reg [1:0] mode;
    reg [3:0] d;
    wire [3:0] q;
    integer m, value, serial_bit;
    reg [3:0] expected;
    shift_register dut(.clk(clk),.rst(rst),.mode(mode),.d(d),
                       .dsl(dsl),.dsr(dsr),.q(q));
    always #5 clk=~clk;
    task check;
        input [3:0] e;
        begin if(q !== e) $fatal(1,"Shift mode=%b q=%b expected=%b",mode,q,e); end
    endtask
    task step;
        input [1:0] op;
        input [3:0] data;
        input sl, sr;
        input [3:0] result;
        begin
            @(negedge clk); mode=op; d=data; dsl=sl; dsr=sr;
            @(posedge clk); #1; check(result);
        end
    endtask
    initial begin
        rst=1; mode=0; d=0; dsl=0; dsr=0;
        #1; check(0);
        @(negedge clk); rst=0;
        step(3,4'b1010,0,0,4'b1010);
        step(0,4'b0101,1,1,4'b1010);
        step(1,0,0,1,4'b1101);
        step(1,0,0,0,4'b0110);
        step(2,0,1,0,4'b1101);
        step(2,0,0,0,4'b1010);
        @(negedge clk); #2; rst=1; #1; check(0);
        @(negedge clk); rst=0;
        // Check every starting word, mode, and relevant serial-input value.
        for(value=0;value<16;value=value+1)
            for(m=0;m<4;m=m+1)
                for(serial_bit=0;serial_bit<2;serial_bit=serial_bit+1) begin
                    step(3,value,0,0,value);
                    case(m)
                        0: expected=value;
                        1: expected=(value>>1) | (serial_bit<<3);
                        2: expected=(value<<1) | serial_bit;
                        3: expected=15-value;
                    endcase
                    step(m,15-value,serial_bit,serial_bit,expected);
                end
        $display("PASS tb_shift_register: directed timing + 128 mode/data/serial cases");
        $finish;
    end
endmodule
