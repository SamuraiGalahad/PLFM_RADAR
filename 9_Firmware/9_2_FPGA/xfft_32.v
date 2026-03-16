`timescale 1ns / 1ps
// ============================================================================
// xfft_32.v — 32-point FFT with AXI-Stream interface
// ============================================================================
// Wraps the synthesizable fft_engine (radix-2 DIT) with the AXI-Stream port
// interface expected by doppler_processor.v.
//
// Port interface matches the Xilinx LogiCORE IP Fast Fourier Transform
// (AXI-Stream variant) as instantiated in doppler_processor.v.
//
// Data format: {Q[15:0], I[15:0]} packed 32-bit.
// Config tdata[0]: 1 = forward FFT, 0 = inverse FFT.
// ============================================================================

module xfft_32 (
    input  wire        aclk,
    input  wire        aresetn,

    // Configuration channel (AXI-Stream slave)
    input  wire [7:0]  s_axis_config_tdata,
    input  wire        s_axis_config_tvalid,
    output wire        s_axis_config_tready,

    // Data input channel (AXI-Stream slave)
    input  wire [31:0] s_axis_data_tdata,
    input  wire        s_axis_data_tvalid,
    input  wire        s_axis_data_tlast,

    // Data output channel (AXI-Stream master)
    output wire [31:0] m_axis_data_tdata,
    output wire        m_axis_data_tvalid,
    output wire        m_axis_data_tlast,
    input  wire        m_axis_data_tready
);

// ============================================================================
// PARAMETERS
// ============================================================================
localparam N     = 32;
localparam LOG2N = 5;

// ============================================================================
// INTERNAL SIGNALS
// ============================================================================

// FSM states
localparam [2:0] S_IDLE    = 3'd0,
                 S_CONFIG  = 3'd1,  // Latch config (fwd/inv)
                 S_FEED    = 3'd2,  // Feed input to FFT engine
                 S_WAIT    = 3'd3,  // Wait for FFT to complete
                 S_OUTPUT  = 3'd4;  // Stream output

reg [2:0] state;

// Configuration
reg inverse_reg;

// Input buffering
reg signed [15:0] in_buf_re [0:N-1];
reg signed [15:0] in_buf_im [0:N-1];
reg [5:0] in_count;    // 0..31 for loading, extra bit for overflow check

// Output buffering
reg signed [15:0] out_buf_re [0:N-1];
reg signed [15:0] out_buf_im [0:N-1];
reg [5:0] out_count;
reg [5:0] out_total;   // counts how many outputs captured from engine

// FFT engine interface
reg fft_start;
reg fft_inverse;
reg signed [15:0] fft_din_re, fft_din_im;
reg fft_din_valid;
wire signed [15:0] fft_dout_re, fft_dout_im;
wire fft_dout_valid;
wire fft_busy;
wire fft_done;

// Feed counter for streaming into engine
reg [5:0] feed_count;

// ============================================================================
// FFT ENGINE INSTANCE
// ============================================================================
fft_engine #(
    .N(N),
    .LOG2N(LOG2N),
    .DATA_W(16),
    .INTERNAL_W(32),
    .TWIDDLE_W(16),
    .TWIDDLE_FILE("fft_twiddle_32.mem")
) fft_core (
    .clk(aclk),
    .reset_n(aresetn),
    .start(fft_start),
    .inverse(fft_inverse),
    .din_re(fft_din_re),
    .din_im(fft_din_im),
    .din_valid(fft_din_valid),
    .dout_re(fft_dout_re),
    .dout_im(fft_dout_im),
    .dout_valid(fft_dout_valid),
    .busy(fft_busy),
    .done(fft_done)
);

// ============================================================================
// AXI-STREAM OUTPUTS
// ============================================================================

// Config is accepted when idle
assign s_axis_config_tready = (state == S_IDLE);

// Output data: {Q, I} packed
assign m_axis_data_tdata  = {out_buf_im[out_count[4:0]], out_buf_re[out_count[4:0]]};
assign m_axis_data_tvalid = (state == S_OUTPUT) && (out_count < N);
assign m_axis_data_tlast  = (state == S_OUTPUT) && (out_count == N - 1);

// ============================================================================
// BUFFER WRITE LOGIC — separate always block, NO async reset
// Allows Vivado to infer distributed RAM instead of dissolving into registers.
// ============================================================================
// Input buffer write enable
reg in_buf_we;
reg [4:0] in_buf_waddr;
reg signed [15:0] in_buf_wdata_re, in_buf_wdata_im;

// Output buffer write enable
reg out_buf_we;
reg [4:0] out_buf_waddr;
reg signed [15:0] out_buf_wdata_re, out_buf_wdata_im;

always @(posedge aclk) begin
    if (in_buf_we) begin
        in_buf_re[in_buf_waddr] <= in_buf_wdata_re;
        in_buf_im[in_buf_waddr] <= in_buf_wdata_im;
    end
    if (out_buf_we) begin
        out_buf_re[out_buf_waddr] <= out_buf_wdata_re;
        out_buf_im[out_buf_waddr] <= out_buf_wdata_im;
    end
end

// ============================================================================
// MAIN FSM
// ============================================================================
always @(posedge aclk or negedge aresetn) begin
    if (!aresetn) begin
        state        <= S_IDLE;
        inverse_reg  <= 1'b0;
        in_count     <= 0;
        out_count    <= 0;
        out_total    <= 0;
        feed_count   <= 0;
        fft_start    <= 1'b0;
        fft_inverse  <= 1'b0;
        fft_din_re   <= 0;
        fft_din_im   <= 0;
        fft_din_valid <= 1'b0;
        in_buf_we    <= 1'b0;
        in_buf_waddr <= 0;
        in_buf_wdata_re <= 0;
        in_buf_wdata_im <= 0;
        out_buf_we   <= 1'b0;
        out_buf_waddr <= 0;
        out_buf_wdata_re <= 0;
        out_buf_wdata_im <= 0;
    end else begin
        // Defaults
        fft_start     <= 1'b0;
        fft_din_valid <= 1'b0;
        in_buf_we     <= 1'b0;
        out_buf_we    <= 1'b0;

        case (state)

        // ================================================================
        S_IDLE: begin
            in_count <= 0;
            if (s_axis_config_tvalid) begin
                // Config tdata[0]: 1=forward, 0=inverse
                // fft_engine: inverse=0 means forward, inverse=1 means inverse
                inverse_reg <= ~s_axis_config_tdata[0];
                state       <= S_FEED;
                in_count    <= 0;
                feed_count  <= 0;
            end
        end

        // ================================================================
        // S_FEED: Buffer all N inputs first, then start engine.
        // ================================================================
        S_FEED: begin
            if (in_count < N) begin
                // Still accepting input data
                if (s_axis_data_tvalid) begin
                    in_buf_we       <= 1'b1;
                    in_buf_waddr    <= in_count[4:0];
                    in_buf_wdata_re <= s_axis_data_tdata[15:0];
                    in_buf_wdata_im <= s_axis_data_tdata[31:16];
                    in_count <= in_count + 1;
                end
            end else if (feed_count == 0) begin
                // All N inputs buffered, start the FFT engine
                fft_start   <= 1'b1;
                fft_inverse <= inverse_reg;
                feed_count  <= 0;
                state       <= S_WAIT;
                out_total   <= 0;
            end
        end

        // ================================================================
        // S_WAIT: Feed buffered data to engine, then wait for output
        // ================================================================
        S_WAIT: begin
            if (feed_count < N) begin
                fft_din_re   <= in_buf_re[feed_count[4:0]];
                fft_din_im   <= in_buf_im[feed_count[4:0]];
                fft_din_valid <= 1'b1;
                feed_count   <= feed_count + 1;
            end

            // Capture engine outputs
            if (fft_dout_valid && out_total < N) begin
                out_buf_we       <= 1'b1;
                out_buf_waddr    <= out_total[4:0];
                out_buf_wdata_re <= fft_dout_re;
                out_buf_wdata_im <= fft_dout_im;
                out_total <= out_total + 1;
            end

            // Engine done
            if (fft_done) begin
                state     <= S_OUTPUT;
                out_count <= 0;
            end
        end

        // ================================================================
        // S_OUTPUT: Stream buffered results via AXI-Stream master
        // ================================================================
        S_OUTPUT: begin
            if (m_axis_data_tready || !m_axis_data_tvalid) begin
                if (out_count < N) begin
                    // m_axis_data_tdata driven combinationally from out_buf
                    if (m_axis_data_tready) begin
                        out_count <= out_count + 1;
                    end
                end
                if (out_count >= N - 1 && m_axis_data_tready) begin
                    state <= S_IDLE;
                end
            end
        end

        default: state <= S_IDLE;

        endcase
    end
end

// ============================================================================
// MEMORY INIT (simulation only)
// ============================================================================
`ifdef SIMULATION
integer init_k;
initial begin
    for (init_k = 0; init_k < N; init_k = init_k + 1) begin
        in_buf_re[init_k]  = 0;
        in_buf_im[init_k]  = 0;
        out_buf_re[init_k] = 0;
        out_buf_im[init_k] = 0;
    end
end
`endif

endmodule
