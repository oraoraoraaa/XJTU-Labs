`timescale 1ns / 1ps
module tb_sr_latch;
    reg s, r;
    wire q, qn;
    sr_latch dut(.s(s), .r(r), .q(q), .qn(qn));
    task check;
        input es, er, eq, eqn;
        begin
            s = es; r = er; #10;
            if ({q, qn} !== {eq, eqn})
                $fatal(1, "SR mismatch S=%b R=%b Q=%b QN=%b", s,r,q,qn);
        end
    endtask
    initial begin
        check(0,1,0,1); // initialize by resetting
        check(0,0,0,1); // hold zero
        check(1,0,1,0); // set
        check(0,0,1,0); // hold one
        check(0,1,0,1); // reset
        check(1,1,0,0); // forbidden NOR-latch condition
        check(0,1,0,1); // leave forbidden condition through a defined reset
        check(0,0,0,1);
        $display("PASS tb_sr_latch: 8 input/hold/forbidden checks");
        $finish;
    end
endmodule
