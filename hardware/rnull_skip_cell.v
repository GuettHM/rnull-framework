// ============================================================================
// Module  : rnull_skip_cell
// Project : rnull-framework
// Author  : Hakim Guettal
//
// One pipeline stage of a chained accumulator that uses the *directed*
// coupling  w |> x  instead of the absorbing coupling used by rnull_mac:
//
//   z_out <= z_in (+) (w |> x)
//
//   x null            -> the branch is null and contributes nothing
//   w null, x real    -> the branch is x itself (the weight acts as 1)
//   w real, x real    -> the branch is the product w * x
//
// The output is null only if z_in is null and the branch is null.
// The multiplier operands are forced to zero whenever the product is not
// needed (operand isolation).
//
// This cell is fully synchronous.  It does not implement pruning: a pruned
// synapse must contribute nothing, which is what rnull_mac does.
//
// Parameters
//   DATA_WIDTH : operand width in bits (signed), >= 2
//   ACC_WIDTH  : accumulator width in bits, > 2*DATA_WIDTH
// ============================================================================
`timescale 1ns / 1ps

module rnull_skip_cell #(
    parameter DATA_WIDTH = 16,
    parameter ACC_WIDTH  = 2*DATA_WIDTH + 8
)(
    input  wire                         clk,
    input  wire                         rst_n,      // asynchronous, active low
    input  wire signed [DATA_WIDTH-1:0] v_x,
    input  wire                         m_x,        // 1: input is null
    input  wire signed [DATA_WIDTH-1:0] v_w,
    input  wire                         m_w,        // 1: weight is null
    input  wire signed [ACC_WIDTH-1:0]  v_acc_in,
    input  wire                         m_acc_in,   // 1: upstream sum is null
    output reg  signed [ACC_WIDTH-1:0]  v_acc_out,
    output reg                          m_acc_out,
    output wire                         mult_idle,  // 1: multiplier operands held at 0
    output wire                         skip_wire   // 1: input forwarded with weight 1
);

    assign mult_idle = m_x | m_w;
    assign skip_wire = (~m_x) & m_w;

    wire signed [DATA_WIDTH-1:0]   gated_x = mult_idle ? {DATA_WIDTH{1'b0}} : v_x;
    wire signed [DATA_WIDTH-1:0]   gated_w = mult_idle ? {DATA_WIDTH{1'b0}} : v_w;
    wire signed [2*DATA_WIDTH-1:0] product = gated_x * gated_w;

    // Signed assignments sign-extend to the accumulator width.
    wire signed [ACC_WIDTH-1:0] product_ext = product;
    wire signed [ACC_WIDTH-1:0] x_ext       = v_x;
    wire signed [ACC_WIDTH-1:0] zero        = {ACC_WIDTH{1'b0}};

    wire signed [ACC_WIDTH-1:0] branch   = m_x ? zero : (m_w ? x_ext : product_ext);
    wire signed [ACC_WIDTH-1:0] acc_base = m_acc_in ? zero : v_acc_in;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_acc_out <= {ACC_WIDTH{1'b0}};
            m_acc_out <= 1'b1;
        end else begin
            v_acc_out <= acc_base + branch;
            m_acc_out <= m_acc_in & m_x;
        end
    end

endmodule
