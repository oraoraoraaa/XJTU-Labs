`timescale 1ns / 1ps
module fsm_div3(input wire clk, rst, output wire y);
    reg [2:0] state, nextstate;
    always @(posedge clk or posedge rst)
        if (rst) state <= 3'b001;
        else state <= nextstate;
    always @(*) begin
        case (state)
            3'b001: nextstate = 3'b010;
            3'b010: nextstate = 3'b100;
            3'b100: nextstate = 3'b001;
            default: nextstate = 3'b001;
        endcase
    end
    assign y = (state == 3'b001);
endmodule

// Separate asynchronous preset/clear controls demonstrate slide 20.
// Clear has priority if both controls are asserted.
module d_ff_set_reset(input wire clk, aset, arst, d, output reg q);
    always @(posedge clk or posedge aset or posedge arst)
        if (arst) q <= 1'b0;
        else if (aset) q <= 1'b1;
        else q <= d;
endmodule

module ring_div3(input wire clk, rst, output wire y);
    wire [2:0] state;
    d_ff_set_reset u0(.clk(clk), .aset(rst), .arst(1'b0),
                      .d(state[2]), .q(state[0]));
    d_ff_set_reset u1(.clk(clk), .aset(1'b0), .arst(rst),
                      .d(state[0]), .q(state[1]));
    d_ff_set_reset u2(.clk(clk), .aset(1'b0), .arst(rst),
                      .d(state[1]), .q(state[2]));
    assign y = state[0];
endmodule
