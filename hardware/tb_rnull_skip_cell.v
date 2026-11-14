// ============================================================================
// Testbench : tb_rnull_skip_cell  (self-checking)
//
// Stimulus changes on the falling clock edge and outputs are sampled after
// the rising edge.
//
// Run (Icarus Verilog):
//   iverilog -g2005 -o tb_rnull_skip_cell.vvp hardware/rnull_skip_cell.v hardware/tb_rnull_skip_cell.v
//   vvp tb_rnull_skip_cell.vvp
// The last line printed is "TB_RNULL_SKIP_CELL: PASS" or "... FAIL".
// ============================================================================
`timescale 1ns / 1ps

module tb_rnull_skip_cell;

    localparam DATA_WIDTH = 16;
    localparam ACC_WIDTH  = 40;

    reg clk;
    reg rst_n;
    reg  signed [DATA_WIDTH-1:0] v_x;
    reg  signed [DATA_WIDTH-1:0] v_w;
    reg                          m_x;
    reg                          m_w;
    reg  signed [ACC_WIDTH-1:0]  v_acc_in;
    reg                          m_acc_in;
    wire signed [ACC_WIDTH-1:0]  v_acc_out;
    wire                         m_acc_out;
    wire                         mult_idle;
    wire                         skip_wire;

    integer errors;

    rnull_skip_cell #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) dut (
        .clk(clk), .rst_n(rst_n),
        .v_x(v_x), .m_x(m_x),
        .v_w(v_w), .m_w(m_w),
        .v_acc_in(v_acc_in), .m_acc_in(m_acc_in),
        .v_acc_out(v_acc_out), .m_acc_out(m_acc_out),
        .mult_idle(mult_idle), .skip_wire(skip_wire)
    );

    always #5 clk = ~clk;

    task apply;
        input signed [DATA_WIDTH-1:0] x;
        input                         xn;
        input signed [DATA_WIDTH-1:0] w;
        input                         wn;
        input signed [ACC_WIDTH-1:0]  acc;
        input                         accn;
        input signed [ACC_WIDTH-1:0]  exp_val;
        input                         exp_null;
        input                         exp_idle;
        input                         exp_skip;
        input [8*32-1:0]              label;
        begin
            @(negedge clk);
            v_x = x;  m_x = xn;
            v_w = w;  m_w = wn;
            v_acc_in = acc;  m_acc_in = accn;
            @(posedge clk);
            #1;
            if (m_acc_out !== exp_null || (!exp_null && v_acc_out !== exp_val) ||
                mult_idle !== exp_idle || skip_wire !== exp_skip) begin
                errors = errors + 1;
                $display("ERROR [%0s] v_acc_out=%0d (exp %0d) m_acc_out=%b (exp %b) idle=%b skip=%b",
                         label, v_acc_out, exp_val, m_acc_out, exp_null, mult_idle, skip_wire);
            end
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        v_x = 0;  m_x = 1'b1;
        v_w = 0;  m_w = 1'b1;
        v_acc_in = 0;  m_acc_in = 1'b1;
        errors = 0;

        repeat (2) @(negedge clk);
        if (m_acc_out !== 1'b1) begin
            errors = errors + 1;
            $display("ERROR reset: m_acc_out=%b (exp 1)", m_acc_out);
        end
        rst_n = 1'b1;

        //    x        xn    w        wn    acc_in   accn  expected  null  idle  skip
        apply(16'sd25, 1'b1, 16'sd4,  1'b0, 40'sd0,  1'b1, 40'sd0,   1'b1, 1'b1, 1'b0, "null input, null sum");
        apply(16'sd25, 1'b1, 16'sd4,  1'b0, 40'sd7,  1'b0, 40'sd7,   1'b0, 1'b1, 1'b0, "null input, real sum");
        apply(16'sd12, 1'b0, 16'sd99, 1'b1, 40'sd0,  1'b1, 40'sd12,  1'b0, 1'b1, 1'b1, "null weight: skip wire");
        apply(-16'sd12, 1'b0, 16'sd99, 1'b1, 40'sd2, 1'b0, -40'sd10, 1'b0, 1'b1, 1'b1, "skip wire, negative x");
        apply(16'sd7,  1'b0, 16'sd5,  1'b0, 40'sd12, 1'b0, 40'sd47,  1'b0, 1'b0, 1'b0, "real product");
        apply(16'sd7,  1'b0, -16'sd5, 1'b0, 40'sd0,  1'b1, -40'sd35, 1'b0, 1'b0, 1'b0, "product, null sum");
        apply(16'sd0,  1'b0, 16'sd5,  1'b0, 40'sd0,  1'b1, 40'sd0,   1'b0, 1'b0, 1'b0, "real zero is not null");

        if (errors == 0) $display("TB_RNULL_SKIP_CELL: PASS");
        else             $display("TB_RNULL_SKIP_CELL: FAIL");
        $finish;
    end

endmodule
