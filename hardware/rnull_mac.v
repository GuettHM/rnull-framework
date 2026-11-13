// ============================================================================
// Module: rnull_mac
// Project: Causal Transparency Algebra (R_null) & null-Net Framework (v1.0.1)
// Author: Hakim Guettal | DOI: 10.5281/zenodo.24175268
// Description: Synthesizable Multiply-Accumulate (MAC) unit with 1-Bit Causal
//              Tagging (is_null) and Zero-Power Clock/Operand Gating Bypass.
// ============================================================================
module rnull_mac #(
    parameter WIDTH = 16
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     clear_acc,
    input  wire signed [WIDTH-1:0]  w_val,
    input  wire                     w_null,     // 1'b1 <=> W_ij = null
    input  wire signed [WIDTH-1:0]  x_val,
    input  wire                     x_null,     // 1'b1 <=> x_j  = null
    output reg  signed [2*WIDTH-1:0] acc_val,
    output reg                      acc_null    // 1'b1 <=> Acc_t = null (OOD Abstention)
);

    // Absorbing Synaptic Coupling (w_null | x_null):
    // If either weight or input is null, the synapse abstains (syn_null = 1'b1).
    wire syn_null = w_null | x_null;

    // Operand Isolation / Gating:
    // Prevents combinatorial multiplier switching when syn_null == 1'b1.
    wire signed [WIDTH-1:0]   gated_w = syn_null ? {WIDTH{1'b0}} : w_val;
    wire signed [WIDTH-1:0]   gated_x = syn_null ? {WIDTH{1'b0}} : x_val;
    wire signed [2*WIDTH-1:0] prod    = gated_w * gated_x;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_val  <= {(2*WIDTH){1'b0}};
            acc_null <= 1'b1; // Starts in epistemic abstention state (null)
        end else if (clear_acc) begin
            acc_val  <= {(2*WIDTH){1'b0}};
            acc_null <= 1'b1;
        end else if (!syn_null) begin
            // Active synapse: Acc_t = Acc_{t-1} (+) prod
            acc_val  <= acc_null ? prod : (acc_val + prod);
            acc_null <= 1'b0;
        end
        // Zero-Power Bypass: when syn_null == 1'b1, registers hold state without switching:
        // Acc_t = Acc_{t-1} (+) null = Acc_{t-1}
    end

endmodule
