`timescale 1ns / 1ps
// Moore FSM: IDLE=00, SEEN_ZERO=01, FOUND=10. Output depends on state.
module sequence_01(input wire clk, rst, a, output wire y);
    localparam IDLE = 2'b00, SEEN_ZERO = 2'b01, FOUND = 2'b10;
    reg [1:0] state, nextstate;
    always @(posedge clk or posedge rst)
        if (rst) state <= IDLE;
        else state <= nextstate;
    always @(*) begin
        case (state)
            IDLE:      nextstate = a ? IDLE : SEEN_ZERO;
            SEEN_ZERO: nextstate = a ? FOUND : SEEN_ZERO;
            FOUND:     nextstate = a ? IDLE : SEEN_ZERO;
            default:   nextstate = IDLE;
        endcase
    end
    assign y = (state == FOUND);
endmodule

// Equivalent D-input equations from slide 22: S'={S0*A, ~A}, Y=S1.
module sequence_01_equations(input wire clk, rst, a, output wire y);
    reg [1:0] state;
    always @(posedge clk or posedge rst)
        if (rst) state <= 2'b00;
        else state <= {state[0] & a, ~a};
    assign y = state[1];
endmodule
