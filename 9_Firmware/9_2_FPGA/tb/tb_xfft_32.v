`timescale 1ns / 1ps

/**
 * tb_xfft_32.v
 *
 * Testbench for xfft_32 AXI-Stream FFT wrapper.
 * Verifies the wrapper correctly interfaces with fft_engine via AXI-Stream.
 *
 * Test Groups:
 *   1. Impulse response (all output bins = input amplitude)
 *   2. DC input (bin 0 = A*N, rest ~= 0)
 *   3. Single tone detection
 *   4. AXI-Stream handshake correctness (tvalid, tlast, tready)
 *   5. Back-to-back transforms (no state leakage)
 */

module tb_xfft_32;

// ============================================================================
// PARAMETERS
// ============================================================================
localparam N         = 32;
localparam CLK_PERIOD = 10;

// ============================================================================
// SIGNALS
// ============================================================================
reg         aclk, aresetn;
reg  [7:0]  cfg_tdata;
reg         cfg_tvalid;
wire        cfg_tready;
reg  [31:0] din_tdata;
reg         din_tvalid;
reg         din_tlast;
wire [31:0] dout_tdata;
wire        dout_tvalid;
wire        dout_tlast;
reg         dout_tready;

// ============================================================================
// DUT
// ============================================================================
xfft_32 dut (
    .aclk(aclk),
    .aresetn(aresetn),
    .s_axis_config_tdata(cfg_tdata),
    .s_axis_config_tvalid(cfg_tvalid),
    .s_axis_config_tready(cfg_tready),
    .s_axis_data_tdata(din_tdata),
    .s_axis_data_tvalid(din_tvalid),
    .s_axis_data_tlast(din_tlast),
    .m_axis_data_tdata(dout_tdata),
    .m_axis_data_tvalid(dout_tvalid),
    .m_axis_data_tlast(dout_tlast),
    .m_axis_data_tready(dout_tready)
);

// ============================================================================
// CLOCK
// ============================================================================
initial aclk = 0;
always #(CLK_PERIOD/2) aclk = ~aclk;

// ============================================================================
// PASS/FAIL TRACKING
// ============================================================================
integer pass_count, fail_count;

task check;
    input cond;
    input [512*8-1:0] label;
    begin
        if (cond) begin
            $display("  [PASS] %0s", label);
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] %0s", label);
            fail_count = fail_count + 1;
        end
    end
endtask

// ============================================================================
// OUTPUT CAPTURE
// ============================================================================
reg signed [15:0] out_re [0:N-1];
reg signed [15:0] out_im [0:N-1];
integer out_idx;
reg got_tlast;
integer tlast_count;

// ============================================================================
// HELPER TASKS
// ============================================================================

task do_reset;
    begin
        aresetn    = 0;
        cfg_tdata  = 0;
        cfg_tvalid = 0;
        din_tdata  = 0;
        din_tvalid = 0;
        din_tlast  = 0;
        dout_tready = 1;
        repeat(5) @(posedge aclk);
        aresetn = 1;
        repeat(2) @(posedge aclk);
    end
endtask

// Send config (forward FFT: tdata[0]=1)
// Waits for cfg_tready (wrapper in S_IDLE) before sending
task send_config;
    input [7:0] cfg;
    integer wait_cnt;
    begin
        // Wait for wrapper to be ready (S_IDLE)
        wait_cnt = 0;
        while (!cfg_tready && wait_cnt < 5000) begin
            @(posedge aclk);
            wait_cnt = wait_cnt + 1;
        end
        cfg_tdata  = cfg;
        cfg_tvalid = 1;
        @(posedge aclk);
        cfg_tvalid = 0;
        cfg_tdata  = 0;
    end
endtask

// Feed N samples: each sample is {im[15:0], re[15:0]}
// in_re_arr and in_im_arr must be pre-loaded
reg signed [15:0] feed_re [0:N-1];
reg signed [15:0] feed_im [0:N-1];

task feed_data;
    integer i;
    begin
        for (i = 0; i < N; i = i + 1) begin
            din_tdata  = {feed_im[i], feed_re[i]};
            din_tvalid = 1;
            din_tlast  = (i == N - 1) ? 1 : 0;
            @(posedge aclk);
        end
        din_tvalid = 0;
        din_tlast  = 0;
        din_tdata  = 0;
    end
endtask

// Capture N output samples
task capture_output;
    integer timeout;
    begin
        out_idx    = 0;
        got_tlast  = 0;
        tlast_count = 0;
        timeout    = 0;
        while (out_idx < N && timeout < 5000) begin
            @(posedge aclk);
            if (dout_tvalid && dout_tready) begin
                out_re[out_idx] = dout_tdata[15:0];
                out_im[out_idx] = dout_tdata[31:16];
                if (dout_tlast) begin
                    got_tlast = 1;
                    tlast_count = tlast_count + 1;
                end
                out_idx = out_idx + 1;
            end
            timeout = timeout + 1;
        end
    end
endtask

// ============================================================================
// VCD
// ============================================================================
initial begin
    $dumpfile("tb_xfft_32.vcd");
    $dumpvars(0, tb_xfft_32);
end

// ============================================================================
// MAIN TEST
// ============================================================================
integer i;
reg signed [31:0] err;
integer max_err;
integer max_mag_bin;
reg signed [31:0] max_mag, mag;
real angle;

initial begin
    pass_count = 0;
    fail_count = 0;

    $display("============================================================");
    $display("  xfft_32 AXI-Stream Wrapper Testbench");
    $display("============================================================");

    do_reset;

    // ================================================================
    // TEST 1: Impulse Response
    // ================================================================
    $display("");
    $display("--- Test 1: Impulse Response ---");

    for (i = 0; i < N; i = i + 1) begin
        feed_re[i] = (i == 0) ? 16'sd1000 : 16'sd0;
        feed_im[i] = 16'sd0;
    end

    send_config(8'h01);  // Forward FFT
    feed_data;
    capture_output;

    check(out_idx == N, "Received N output samples");
    check(got_tlast == 1, "Got tlast on output");

    max_err = 0;
    for (i = 0; i < N; i = i + 1) begin
        err = out_re[i] - 1000;
        if (err < 0) err = -err;
        if (err > max_err) max_err = err;
        err = out_im[i];
        if (err < 0) err = -err;
        if (err > max_err) max_err = err;
    end
    $display("  Impulse max error: %0d", max_err);
    check(max_err < 10, "Impulse: all bins ~= 1000");

    // ================================================================
    // TEST 2: DC Input
    // ================================================================
    $display("");
    $display("--- Test 2: DC Input ---");

    for (i = 0; i < N; i = i + 1) begin
        feed_re[i] = 16'sd100;
        feed_im[i] = 16'sd0;
    end

    send_config(8'h01);
    feed_data;
    capture_output;

    $display("  DC bin[0] = %0d + j%0d (expect ~3200)", out_re[0], out_im[0]);
    check(out_re[0] >= 3100 && out_re[0] <= 3300, "DC: bin 0 ~= 3200 (5% tol)");

    max_err = 0;
    for (i = 1; i < N; i = i + 1) begin
        err = out_re[i]; if (err < 0) err = -err;
        if (err > max_err) max_err = err;
        err = out_im[i]; if (err < 0) err = -err;
        if (err > max_err) max_err = err;
    end
    $display("  DC max non-DC: %0d", max_err);
    check(max_err < 25, "DC: non-DC bins ~= 0");

    // ================================================================
    // TEST 3: Single Tone (bin 4)
    // ================================================================
    $display("");
    $display("--- Test 3: Single Tone (bin 4) ---");

    for (i = 0; i < N; i = i + 1) begin
        angle = 6.28318530718 * 4.0 * i / 32.0;
        feed_re[i] = $rtoi($cos(angle) * 1000.0);
        feed_im[i] = 16'sd0;
    end

    send_config(8'h01);
    feed_data;
    capture_output;

    max_mag = 0;
    max_mag_bin = 0;
    for (i = 0; i < N; i = i + 1) begin
        mag = out_re[i] * out_re[i] + out_im[i] * out_im[i];
        if (mag > max_mag) begin
            max_mag = mag;
            max_mag_bin = i;
        end
    end
    $display("  Tone peak bin: %0d (expect 4 or 28)", max_mag_bin);
    check(max_mag_bin == 4 || max_mag_bin == 28, "Tone: peak at bin 4 or 28");

    // ================================================================
    // TEST 4: Back-to-back transforms
    // ================================================================
    $display("");
    $display("--- Test 4: Back-to-Back Transforms ---");

    // First: impulse
    for (i = 0; i < N; i = i + 1) begin
        feed_re[i] = (i == 0) ? 16'sd500 : 16'sd0;
        feed_im[i] = 16'sd0;
    end
    send_config(8'h01);
    feed_data;
    capture_output;
    check(out_idx == N, "Back-to-back 1st: got N outputs");

    // Second: DC immediately after
    for (i = 0; i < N; i = i + 1) begin
        feed_re[i] = 16'sd50;
        feed_im[i] = 16'sd0;
    end
    send_config(8'h01);
    feed_data;
    capture_output;
    check(out_idx == N, "Back-to-back 2nd: got N outputs");
    $display("  2nd transform bin[0] = %0d (expect ~1600)", out_re[0]);
    check(out_re[0] >= 1500 && out_re[0] <= 1700, "Back-to-back 2nd: bin 0 ~= 1600");

    // ================================================================
    // TEST 5: Zero input
    // ================================================================
    $display("");
    $display("--- Test 5: Zero Input ---");

    for (i = 0; i < N; i = i + 1) begin
        feed_re[i] = 16'sd0;
        feed_im[i] = 16'sd0;
    end
    send_config(8'h01);
    feed_data;
    capture_output;

    max_err = 0;
    for (i = 0; i < N; i = i + 1) begin
        err = out_re[i]; if (err < 0) err = -err;
        if (err > max_err) max_err = err;
        err = out_im[i]; if (err < 0) err = -err;
        if (err > max_err) max_err = err;
    end
    check(max_err == 0, "Zero input: all outputs = 0");

    // ================================================================
    // SUMMARY
    // ================================================================
    $display("");
    $display("============================================================");
    $display("  RESULTS: %0d/%0d passed", pass_count, pass_count + fail_count);
    if (fail_count == 0)
        $display("  ALL TESTS PASSED");
    else
        $display("  SOME TESTS FAILED");
    $display("============================================================");

    $finish;
end

endmodule
