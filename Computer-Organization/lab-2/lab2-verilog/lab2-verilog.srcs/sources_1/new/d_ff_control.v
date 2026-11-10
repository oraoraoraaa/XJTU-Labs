`timescale 1ns / 1ps
// Reset has priority over enable. ASYNC_RESET=0 selects synchronous reset.
module d_ff_control #(parameter ASYNC_RESET = 1)(
    input wire clk, rst, en, d, output reg q, output wire qn
);
    generate
        if (ASYNC_RESET) begin : g_async
            always @(posedge clk or posedge rst)
                if (rst) q <= 1'b0;
                else if (en) q <= d;
        end else begin : g_sync
            always @(posedge clk)
                if (rst) q <= 1'b0;
                else if (en) q <= d;
        end
    endgenerate
    assign qn = ~q;
endmodule
