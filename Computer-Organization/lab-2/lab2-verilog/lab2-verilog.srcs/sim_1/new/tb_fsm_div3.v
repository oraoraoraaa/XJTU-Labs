`timescale 1ns / 1ps
module tb_fsm_div3;
    reg clk=0;
    reg rst;
    wire y_case, y_ring;
    integer i;
    reg [2:0] expected;
    fsm_div3 u_case(.clk(clk),.rst(rst),.y(y_case));
    ring_div3 u_ring(.clk(clk),.rst(rst),.y(y_ring));
    always #5 clk=~clk;
    task check;
        input [2:0] e;
        begin
            if(u_case.state !== e || u_ring.state !== e ||
               y_case !== (e==3'b001) || y_ring !== (e==3'b001))
                $fatal(1,"Divider state=%b ring=%b expected=%b",u_case.state,u_ring.state,e);
        end
    endtask
    initial begin
        rst=1; #1; check(3'b001);
        @(negedge clk); rst=0;
        expected=3'b001;
        for(i=0;i<12;i=i+1) begin
            expected={expected[1:0],expected[2]};
            @(posedge clk); #1; check(expected);
        end
        @(negedge clk); #2; rst=1; #1; check(3'b001);
        @(negedge clk); rst=0;
        @(posedge clk); #1; check(3'b010);
        #4;
        $display("PASS tb_fsm_div3: case/ring equivalence, 12 transitions, asynchronous reset");
        $finish;
    end
endmodule
