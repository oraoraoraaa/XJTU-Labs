`timescale 1ns / 1ps
module tb_d_ff_control;
    reg clk=0;
    reg rst, en, d;
    wire q_async, q_sync, qn_async, qn_sync;
    d_ff_control #(.ASYNC_RESET(1)) u_async(
        .clk(clk),.rst(rst),.en(en),.d(d),.q(q_async),.qn(qn_async));
    d_ff_control #(.ASYNC_RESET(0)) u_sync(
        .clk(clk),.rst(rst),.en(en),.d(d),.q(q_sync),.qn(qn_sync));
    always #5 clk=~clk;
    task check;
        input ea, es;
        begin
            if ({q_async,q_sync,qn_async,qn_sync} !== {ea,es,~ea,~es})
                $fatal(1,"FF mismatch t=%0t async=%b sync=%b",$time,q_async,q_sync);
        end
    endtask
    initial begin
        rst=1; en=0; d=1;
        @(posedge clk); #1; check(0,0);
        @(negedge clk); rst=0; en=1;
        @(posedge clk); #1; check(1,1);
        @(negedge clk); en=0; d=0;
        @(posedge clk); #1; check(1,1); // disabled: hold
        @(negedge clk); #2; rst=1;
        #1; check(0,1); // at 33 ns, between rising edges
        @(posedge clk); #1; check(0,0); // at 36 ns, sync catches up
        @(negedge clk); rst=0; en=1; d=1;
        @(posedge clk); #1; check(1,1);
        @(negedge clk); rst=1; en=1; d=1;
        @(posedge clk); #1; check(0,0); // reset priority
        @(negedge clk); rst=0; en=0;
        @(posedge clk); #1; check(0,0);
        @(negedge clk); en=1;
        @(posedge clk); #1; check(1,1);
        #4;
        $display("PASS tb_d_ff_control: 9 enable/reset/timing checks");
        $finish;
    end
endmodule
