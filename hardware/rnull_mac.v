// ============================================================================
// Module  : rnull_mac
// Project : rnull-framework
// Author  : Hakim Guettal
//
// Multiply-accumulate cell with a 1-bit "null" tag on each operand.
//
//   acc <= acc (+) (w (.) x)
//
// * If either operand is tagged null the synapse is skipped: the multiplier
//   inputs are forced to zero (operand isolation) and the accumulator
//   registers are not enabled, so they hold their value.
// * acc_null is 1 until the first non-null product arrives.  It lets the
//   consumer distinguish "sum of real products that happens to be 0" from
//   "nothing was accumulated".
// * The accumulator has GUARD extra bits, so at least 2**GUARD worst-case
//   products can be summed without overflow.  acc_ovf is a sticky flag that
//   is set if the signed sum ever wraps.
//
// Parameters
//   WIDTH : operand width in bits (signed, two's complement), >= 2
//   GUARD : accumulator guard bits, >= 1
//
// Notes
//   * The enable on the accumulator is a synchronous clock enable.  Whether
//     it becomes an integrated clock-gating cell is a synthesis decision.
//   * This file has no associated area, timing or power results.  The
//     fraction of skipped cycles is an activity figure, not a power figure.
//   * rnull/hw_model.py is the bit-accurate Python reference for this module.
// ============================================================================
module rnull_mac #(
    parameter WIDTH = 16,
    parameter GUARD = 8
)(
    input  wire                              clk,
    input  wire                              rst_n,      // asynchronous, active low
    input  wire                              clear_acc,  // synchronous clear
    input  wire signed [WIDTH-1:0]           w_val,
    input  wire                              w_null,     // 1: weight is null
    input  wire signed [WIDTH-1:0]           x_val,
    input  wire                              x_null,     // 1: input is null
    output reg  signed [2*WIDTH+GUARD-1:0]   acc_val,
    output reg                               acc_null,   // 1: nothing accumulated yet
    output reg                               acc_ovf     // sticky signed-overflow flag
);

    localparam ACC_W = 2*WIDTH + GUARD;

    // A synapse is skipped when its weight or its input is null.
    wire syn_null = w_null | x_null;

    // Operand isolation: keep the multiplier inputs quiet on skipped cycles.
    wire signed [WIDTH-1:0]   gated_w = syn_null ? {WIDTH{1'b0}} : w_val;
    wire signed [WIDTH-1:0]   gated_x = syn_null ? {WIDTH{1'b0}} : x_val;
    wire signed [2*WIDTH-1:0] prod    = gated_w * gated_x;

    // Sign-extend the product to the accumulator width and add.
    wire signed [ACC_W-1:0] prod_ext = {{GUARD{prod[2*WIDTH-1]}}, prod};
    wire signed [ACC_W-1:0] acc_base = acc_null ? {ACC_W{1'b0}} : acc_val;
    wire signed [ACC_W-1:0] acc_sum  = acc_base + prod_ext;

    // Signed overflow: operands share a sign and the sum has the other sign.
    wire add_ovf = (acc_base[ACC_W-1] == prod_ext[ACC_W-1]) &&
                   (acc_sum[ACC_W-1]  != acc_base[ACC_W-1]);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_val  <= {ACC_W{1'b0}};
            acc_null <= 1'b1;
            acc_ovf  <= 1'b0;
        end else if (clear_acc) begin
            acc_val  <= {ACC_W{1'b0}};
            acc_null <= 1'b1;
            acc_ovf  <= 1'b0;
        end else if (!syn_null) begin
            acc_val  <= acc_sum;
            acc_null <= 1'b0;
            acc_ovf  <= acc_ovf | add_ovf;
        end
        // syn_null == 1: registers hold (acc (+) null = acc).
    end

endmodule
