`timescale 1ns / 1ps
// 74161-style controls: active-low asynchronous clear and synchronous load.
module binary_counter #(parameter WIDTH = 7)(
    input wire clk, cr_n, pe_n, cep, cet,
    input wire [WIDTH-1:0] d,
    output reg [WIDTH-1:0] q,
    output wire tc
);
    always @(posedge clk or negedge cr_n) begin
        if (!cr_n) q <= {WIDTH{1'b0}};
        else if (!pe_n) q <= d;
        else if (cep && cet) q <= q + 1'b1;
    end
    assign tc = cet && (&q);
endmodule

// Required feedback-clear implementation: decode decimal 99, then clear.
// The decoded clear is a delta-cycle pulse in zero-delay RTL simulation.
module counter_mod99(
    input wire clk, cr_n, pe_n, cep, cet,
    input wire [6:0] d,
    output wire [6:0] q,
    output wire tc, feedback_clear
);
    wire internal_cr_n;
    wire unused_binary_tc;
    assign feedback_clear = (q == 7'd99);
    assign internal_cr_n = cr_n && !feedback_clear;
    binary_counter #(.WIDTH(7)) u_counter (
        .clk(clk), .cr_n(internal_cr_n), .pe_n(pe_n),
        .cep(cep), .cet(cet), .d(d), .q(q), .tc(unused_binary_tc)
    );
    // Terminal-count indication for modulo 99 (last stable count = 98).
    assign tc = cet && (q == 7'd98);
endmodule
