// ============================================================================
// Testbench : tb_rnull_mac  (self-checking)
//
// Drives rnull_mac with directed and random stimulus and compares every
// cycle against a behavioural reference kept in the testbench.  A second,
// narrow instance (WIDTH=4, GUARD=1) checks the overflow flag.
//
// Stimulus changes on the falling clock edge and outputs are sampled after
// the rising edge, so there are no races with the design's registers.
//
// Run (Icarus Verilog):
//   iverilog -g2005 -o tb_rnull_mac.vvp hardware/rnull_mac.v hardware/tb_rnull_mac.v
//   vvp tb_rnull_mac.vvp
// The last line printed is "TB_RNULL_MAC: PASS" or "TB_RNULL_MAC: FAIL".
// ============================================================================
`timescale 1ns / 1ps

module tb_rnull_mac;

    localparam WIDTH = 16;
    localparam GUARD = 8;
    localparam ACC_W = 2*WIDTH + GUARD;

    reg clk;
    reg rst_n;

    // ---------------- main instance ----------------
    reg                     clear_acc;
    reg  signed [WIDTH-1:0] w_val;
    reg  signed [WIDTH-1:0] x_val;
    reg                     w_null;
    reg                     x_null;
    wire signed [ACC_W-1:0] acc_val;
    wire                    acc_null;
    wire                    acc_ovf;

    rnull_mac #(.WIDTH(WIDTH), .GUARD(GUARD)) dut (
        .clk(clk), .rst_n(rst_n), .clear_acc(clear_acc),
        .w_val(w_val), .w_null(w_null),
        .x_val(x_val), .x_null(x_null),
        .acc_val(acc_val), .acc_null(acc_null), .acc_ovf(acc_ovf)
    );

    // ---------------- narrow instance for the overflow flag ----------------
    reg               s_clear;
    reg  signed [3:0] s_w;
    reg  signed [3:0] s_x;
    reg               s_w_null;
    reg               s_x_null;
    wire signed [8:0] s_acc;
    wire              s_null;
    wire              s_ovf;

    rnull_mac #(.WIDTH(4), .GUARD(1)) dut_small (
        .clk(clk), .rst_n(rst_n), .clear_acc(s_clear),
        .w_val(s_w), .w_null(s_w_null),
        .x_val(s_x), .x_null(s_x_null),
        .acc_val(s_acc), .acc_null(s_null), .acc_ovf(s_ovf)
    );

    // ---------------- reference model and bookkeeping ----------------
    reg signed [ACC_W-1:0] ref_acc;
    reg                    ref_null;
    integer                errors;
    integer                checks;
    integer                i;

    always #5 clk = ~clk;

    task check;
        input [8*32-1:0] label;
        begin
            checks = checks + 1;
            if (acc_null !== ref_null || acc_val !== ref_acc || acc_ovf !== 1'b0) begin
                errors = errors + 1;
                $display("ERROR [%0s] t=%0t acc_val=%0d (exp %0d) acc_null=%b (exp %b) acc_ovf=%b",
                         label, $time, acc_val, ref_acc, acc_null, ref_null, acc_ovf);
            end
        end
    endtask

    // Apply one operand pair for one clock cycle and check the result.
    task step;
        input signed [WIDTH-1:0] w;
        input                    wn;
        input signed [WIDTH-1:0] x;
        input                    xn;
        input [8*32-1:0]         label;
        begin
            @(negedge clk);
            clear_acc = 1'b0;
            w_val = w;  w_null = wn;
            x_val = x;  x_null = xn;
            if (!(wn | xn)) begin
                if (ref_null) ref_acc = w * x;
                else          ref_acc = ref_acc + w * x;
                ref_null = 1'b0;
            end
            @(posedge clk);
            #1;
            check(label);
        end
    endtask

    task clear;
        begin
            @(negedge clk);
            clear_acc = 1'b1;
            w_null = 1'b1;
            x_null = 1'b1;
            ref_acc  = {ACC_W{1'b0}};
            ref_null = 1'b1;
            @(posedge clk);
            #1;
            check("clear");
            @(negedge clk);
            clear_acc = 1'b0;
        end
    endtask

    task small_step;
        input signed [3:0] w;
        input signed [3:0] x;
        begin
            @(negedge clk);
            s_clear = 1'b0;
            s_w = w;  s_w_null = 1'b0;
            s_x = x;  s_x_null = 1'b0;
            @(posedge clk);
            #1;
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        clear_acc = 1'b0;
        w_val = 0;  x_val = 0;  w_null = 1'b1;  x_null = 1'b1;
        s_clear = 1'b0;
        s_w = 0;  s_x = 0;  s_w_null = 1'b1;  s_x_null = 1'b1;
        ref_acc = {ACC_W{1'b0}};
        ref_null = 1'b1;
        errors = 0;
        checks = 0;

        // ---- reset ----
        repeat (2) @(negedge clk);
        rst_n = 1'b1;
        @(posedge clk);
        #1;
        check("after reset");

        // ---- directed cases ----
        step(16'sd25,  1'b0, 16'sd4,   1'b1, "null input");
        step(16'sd25,  1'b1, 16'sd4,   1'b0, "null weight");
        step(16'sd25,  1'b1, 16'sd4,   1'b1, "both null");
        step(16'sd0,   1'b0, 16'sd123, 1'b0, "real zero product");   // acc = 0, not null
        step(16'sd5,   1'b0, 16'sd7,   1'b0, "accumulate 35");
        step(-16'sd3,  1'b0, 16'sd9,   1'b0, "negative product");    // 35 - 27 = 8
        step(16'sd99,  1'b0, 16'sd99,  1'b1, "hold on null");
        if (acc_val !== 40'sd8) begin
            errors = errors + 1;
            $display("ERROR directed sum: acc_val=%0d (exp 8)", acc_val);
        end
        clear;
        // 16'sh8000 = -32768, the most negative operand
        step(16'sh8000, 1'b0, 16'sh8000, 1'b0, "max positive product");
        step(16'sh8000, 1'b0,  16'sd32767, 1'b0, "max negative product");
        clear;

        // ---- random cases (about half of the cycles are skipped) ----
        for (i = 0; i < 2000; i = i + 1) begin
            step($random, $random, $random, $random, "random");
        end

        // ---- overflow flag on the narrow instance ----
        // 9-bit accumulator, range [-256, 255]; each product is (-8)*(-8) = 64  (4'sh8 = -8).
        small_step(4'sh8, 4'sh8);
        small_step(4'sh8, 4'sh8);
        small_step(4'sh8, 4'sh8);
        if (s_acc !== 9'sd192 || s_ovf !== 1'b0 || s_null !== 1'b0) begin
            errors = errors + 1;
            $display("ERROR small: before overflow s_acc=%0d s_ovf=%b s_null=%b", s_acc, s_ovf, s_null);
        end
        small_step(4'sh8, 4'sh8);
        if (s_ovf !== 1'b1) begin
            errors = errors + 1;
            $display("ERROR small: overflow not flagged, s_acc=%0d", s_acc);
        end
        small_step(4'sd1, 4'sd1);
        if (s_ovf !== 1'b1) begin
            errors = errors + 1;
            $display("ERROR small: overflow flag is not sticky");
        end
        @(negedge clk);
        s_clear = 1'b1;
        @(posedge clk);
        #1;
        if (s_ovf !== 1'b0 || s_null !== 1'b1) begin
            errors = errors + 1;
            $display("ERROR small: clear did not reset flags");
        end

        $display("tb_rnull_mac: %0d checks, %0d errors", checks, errors);
        if (errors == 0) $display("TB_RNULL_MAC: PASS");
        else             $display("TB_RNULL_MAC: FAIL");
        $finish;
    end

endmodule
