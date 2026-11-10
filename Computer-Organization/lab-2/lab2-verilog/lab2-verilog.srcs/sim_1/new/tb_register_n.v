`timescale 1ns / 1ps
module tb_register_n;
    reg clk=0;
    reg oe;
    reg [7:0] d;
    wire [7:0] q;
    register_n dut(.clk(clk),.oe(oe),.d(d),.q(q));
    always #5 clk=~clk;
    task check;
        input [7:0] expected;
        begin
            if(q !== expected) $fatal(1,"Register q=%h expected=%h",q,expected);
        end
    endtask
    initial begin
        oe=0; d=8'h3c;
        @(posedge clk); #1; check(8'h3c);
        @(negedge clk); d=8'ha5; #1; check(8'h3c);
        @(posedge clk); #1; check(8'ha5);
        @(negedge clk); oe=1; #1; check(8'hzz);
        #1; oe=0; #1; check(8'ha5); // output restored before next clock
        @(negedge clk); oe=1; d=8'h5a;
        @(posedge clk); #1; check(8'hzz); // storage still captures data
        @(negedge clk); oe=0; #1; check(8'h5a);
        @(negedge clk); d=8'hff;
        @(posedge clk); #1; check(8'hff);
        #4;
        $display("PASS tb_register_n: 8 capture/hold/high-impedance checks");
        $finish;
    end
endmodule
