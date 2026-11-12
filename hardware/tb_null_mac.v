// =============================================================================
// Testbench: tb_null_mac
// Description: Verifies Zero-Power Bypass, Passive Wire Skip (w=null), and
//              Epistemic Abstention in the R_null Hardware MAC Cell.
// Author: Hakim Guettal
// =============================================================================
`timescale 1ns / 1ps

module tb_null_mac;
    reg clk = 0;
    reg rst_n = 0;
    reg signed [15:0] v_x, v_w;
    reg m_x, m_w;
    reg signed [31:0] v_acc_in;
    reg m_acc_in;

    wire signed [31:0] v_acc_out;
    wire m_acc_out;
    wire mac_power_gated;
    wire passive_wire_bypass;

    null_synapse_mac #(.DATA_WIDTH(16)) dut (
        .clk(clk),
        .rst_n(rst_n),
        .v_x(v_x), .m_x(m_x),
        .v_w(v_w), .m_w(m_w),
        .v_acc_in(v_acc_in), .m_acc_in(m_acc_in),
        .v_acc_out(v_acc_out), .m_acc_out(m_acc_out),
        .mac_power_gated(mac_power_gated),
        .passive_wire_bypass(passive_wire_bypass)
    );

    always #5 clk = ~clk;

    initial begin
        rst_n = 0;
        v_x = 0; m_x = 1;
        v_w = 0; m_w = 1;
        v_acc_in = 0; m_acc_in = 1;
        #15 rst_n = 1;

        // Case 1: Epistemic Abstention (Input = null, Acc_in = null -> Out = null)
        v_x = 16'd25; m_x = 1'b1;
        v_w = 16'd4;  m_w = 1'b0;
        v_acc_in = 32'd0; m_acc_in = 1'b1;
        #10;
        $display("[CASE 1 - Input Null] m_acc_out=%b (Expected 1), power_gated=%b", m_acc_out, mac_power_gated);

        // Case 2: Passive Wire Skip-Connection (w = null, x = 12 -> Out = 12, multiplier gated!)
        v_x = 16'd12; m_x = 1'b0;
        v_w = 16'd99; m_w = 1'b1;
        v_acc_in = 32'd0; m_acc_in = 1'b1;
        #10;
        $display("[CASE 2 - Weight Null Skip] v_acc_out=%0d (Expected 12), passive_wire=%b, power_gated=%b",
                 v_acc_out, passive_wire_bypass, mac_power_gated);

        // Case 3: Active Real MAC (w = 5, x = 7, acc_in = 12 -> Out = 47)
        v_x = 16'd7;  m_x = 1'b0;
        v_w = 16'd5;  m_w = 1'b0;
        v_acc_in = 32'd12; m_acc_in = 1'b0;
        #10;
        $display("[CASE 3 - Active MAC] v_acc_out=%0d (Expected 47), m_acc_out=%b", v_acc_out, m_acc_out);
        $finish;
    end
endmodule
