// =============================================================================
// Module: null_synapse_mac
// Description: Asynchronous Tri-State / Clock-Gated Hardware Implementation
//              of Causal Transparency Algebra (R_null) Synaptic Cell.
// Author: Hakim Guettal
// =============================================================================
`timescale 1ns / 1ps

module null_synapse_mac #(
    parameter DATA_WIDTH = 16
)(
    input  wire                         clk,
    input  wire                         rst_n,
    // Pre-synaptic input x_j = (v_x, m_x)
    input  wire signed [DATA_WIDTH-1:0] v_x,
    input  wire                         m_x,       // 1 = null (transparent), 0 = real
    // Synaptic weight w_ij = (v_w, m_w)
    input  wire signed [DATA_WIDTH-1:0] v_w,
    input  wire                         m_w,       // 1 = null (passive wire skip), 0 = real
    // Upstream accumulator input z_in = (v_acc_in, m_acc_in)
    input  wire signed [2*DATA_WIDTH-1:0] v_acc_in,
    input  wire                           m_acc_in,
    // Post-synaptic accumulator output z_out = (v_acc_out, m_acc_out)
    output reg  signed [2*DATA_WIDTH-1:0] v_acc_out,
    output reg                            m_acc_out,
    // Hardware telemetry flags
    output wire                           mac_power_gated,
    output wire                           passive_wire_bypass
);

    // Branch transparency & passive wire logic
    // If input is null (m_x == 1), branch collapses to null -> MAC is power-gated!
    // If weight is null (m_w == 1) and input is real (m_x == 0), Axiom 1 (w (*) x = x)
    // routes v_x directly as a passive wire without multiplier switching!
    assign mac_power_gated     = m_x | m_w;
    assign passive_wire_bypass = (~m_x) & m_w;

    // Gated operands prevent dynamic switching power (P = alpha * C * V^2 * f = 0)
    wire signed [DATA_WIDTH-1:0] gated_vx = mac_power_gated ? {DATA_WIDTH{1'b0}} : v_x;
    wire signed [DATA_WIDTH-1:0] gated_vw = mac_power_gated ? {DATA_WIDTH{1'b0}} : v_w;

    wire signed [2*DATA_WIDTH-1:0] mult_res = gated_vx * gated_vw;

    // Branch value selection:
    // - If m_x == 1: branch is null (contributes 0 to real sum under oplus)
    // - If m_w == 1 and m_x == 0: passive wire bypass (contributes sign-extended v_x)
    // - Otherwise: standard product v_w * v_x
    wire signed [2*DATA_WIDTH-1:0] branch_val =
        m_x ? {(2*DATA_WIDTH){1'b0}} :
        (m_w ? {{DATA_WIDTH{v_x[DATA_WIDTH-1]}}, v_x} : mult_res);

    wire branch_is_null = m_x;

    // Axiom 1 oplus accumulation: z_out = z_in (+) branch
    wire signed [2*DATA_WIDTH-1:0] acc_base = m_acc_in ? {(2*DATA_WIDTH){1'b0}} : v_acc_in;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_acc_out <= {(2*DATA_WIDTH){1'b0}};
            m_acc_out <= 1'b1; // Reset into pure epistemic abstention (null)
        end else begin
            v_acc_out <= acc_base + branch_val;
            m_acc_out <= m_acc_in & branch_is_null;
        end
    end

endmodule
