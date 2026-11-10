`timescale 1ns / 1ps
module tb_counter_mod99;
    reg clk=0;
    reg cr_n, pe_n, cep, cet;
    reg [6:0] d;
    wire [6:0] q;
    wire tc, feedback_clear;
    integer i, expected, wraps=0;
    counter_mod99 dut(.clk(clk),.cr_n(cr_n),.pe_n(pe_n),.cep(cep),.cet(cet),
                      .d(d),.q(q),.tc(tc),.feedback_clear(feedback_clear));
    always #5 clk=~clk;
    always @(posedge feedback_clear) wraps=wraps+1;
    task check;
        input [6:0] e;
        begin
            if(q !== e || tc !== (cet && (e==98)))
                $fatal(1,"Counter q=%d expected=%d tc=%b",q,e,tc);
        end
    endtask
    initial begin
        cr_n=0; pe_n=1; cep=1; cet=1; d=0;
        #1; check(0);
        @(negedge clk); cr_n=1;
        // Two complete natural cycles, without loading near the boundary.
        for(i=1;i<=198;i=i+1) begin
            @(posedge clk); #1; expected=i%99; check(expected);
        end
        @(negedge clk); cep=0;
        repeat(2) begin @(posedge clk); #1; check(0); end
        @(negedge clk); cep=1; cet=0;
        repeat(2) begin @(posedge clk); #1; check(0); end
        @(negedge clk); pe_n=0; d=42; // load overrides disabled counting
        @(posedge clk); #1; check(42);
        @(negedge clk); pe_n=1; cet=1;
        @(posedge clk); #1; check(43);
        @(negedge clk); pe_n=0; d=98;
        @(posedge clk); #1; check(98);
        @(negedge clk); pe_n=1;
        @(posedge clk); #1; check(0);
        @(negedge clk); pe_n=0; d=77;
        @(posedge clk); #1; check(77);
        @(negedge clk); #2; cr_n=0; #1; check(0);
        if(wraps !== 3) $fatal(1,"Expected 3 feedback-clear events, got %0d",wraps);
        #7;
        $display("PASS tb_counter_mod99: 198 natural counts, 3 feedback wraps, control checks");
        $finish;
    end
endmodule
