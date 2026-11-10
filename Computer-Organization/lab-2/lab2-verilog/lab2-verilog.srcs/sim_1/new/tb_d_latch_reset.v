`timescale 1ns / 1ps
module tb_d_latch_reset;
    reg d, en, rst;
    wire q, qn;
    d_latch_reset dut(.d(d),.en(en),.rst(rst),.q(q),.qn(qn));
    task check;
        input ed, een, erst, expected;
        begin
            d=ed; en=een; rst=erst; #10;
            if ({q,qn} !== {expected,~expected})
                $fatal(1,"Latch mismatch d=%b en=%b rst=%b q=%b",d,en,rst,q);
        end
    endtask
    initial begin
        check(1,0,1,0);
        check(1,0,0,0);
        check(1,1,0,1);
        check(0,1,0,0);
        check(1,1,0,1);
        check(0,0,0,1);
        check(1,0,1,0); // reset overrides hold
        check(1,1,1,0); // reset overrides data and enable
        check(1,1,0,1);
        $display("PASS tb_d_latch_reset: 9 transparency/hold/reset checks");
        $finish;
    end
endmodule
