`timescale 1ns / 1ps

/**
 * fft_engine.v
 *
 * Synthesizable parameterized radix-2 DIT FFT/IFFT engine.
 * Iterative single-butterfly architecture with quarter-wave twiddle ROM.
 *
 * Architecture:
 *   - LOAD:    Accept N input samples, store bit-reversed in BRAM
 *   - COMPUTE: LOG2N stages x N/2 butterflies, 2-cycle pipeline:
 *              BF_READ:  Present BRAM addresses, capture twiddle
 *              BF_CALC:  BRAM data valid; butterfly compute + writeback
 *   - OUTPUT:  Stream N results (1/N scaling for IFFT)
 *
 * Data memory uses xpm_memory_tdpram (Xilinx Parameterized Macros) for
 * guaranteed BRAM mapping in synthesis.  Under `ifdef SIMULATION, a
 * behavioral Verilog-2001 model replaces the XPM so the design compiles
 * with Icarus Verilog or any non-Xilinx simulator.
 *
 * Clock domain: single clock (clk), active-low async reset (reset_n).
 */

module fft_engine #(
    parameter N            = 1024,
    parameter LOG2N        = 10,
    parameter DATA_W       = 16,
    parameter INTERNAL_W   = 32,
    parameter TWIDDLE_W    = 16,
    parameter TWIDDLE_FILE = "fft_twiddle_1024.mem"
)(
    input wire clk,
    input wire reset_n,

    // Control
    input wire start,
    input wire inverse,

    // Data input
    input wire signed [DATA_W-1:0] din_re,
    input wire signed [DATA_W-1:0] din_im,
    input wire din_valid,

    // Data output
    output reg signed [DATA_W-1:0] dout_re,
    output reg signed [DATA_W-1:0] dout_im,
    output reg dout_valid,

    // Status
    output wire busy,
    output reg  done
);

// ============================================================================
// SAFE WIDTH CONSTANTS
// ============================================================================
localparam [LOG2N:0] FFT_N         = N;
localparam [LOG2N:0] FFT_N_HALF    = N / 2;
localparam [LOG2N:0] FFT_N_QTR     = N / 4;
localparam [LOG2N:0] FFT_N_HALF_M1 = N / 2 - 1;
localparam [LOG2N:0] FFT_N_M1      = N - 1;

// ============================================================================
// STATES
// ============================================================================
localparam [2:0] ST_IDLE    = 3'd0,
                 ST_LOAD    = 3'd1,
                 ST_BF_READ = 3'd2,
                 ST_BF_CALC = 3'd3,
                 ST_OUTPUT  = 3'd4,
                 ST_DONE    = 3'd5;

reg [2:0] state;
assign busy = (state != ST_IDLE);

// ============================================================================
// DATA MEMORY DECLARATIONS
// ============================================================================

// BRAM read data (registered outputs from port blocks)
reg signed [INTERNAL_W-1:0] mem_rdata_a_re, mem_rdata_a_im;
reg signed [INTERNAL_W-1:0] mem_rdata_b_re, mem_rdata_b_im;

// ============================================================================
// TWIDDLE ROM
// ============================================================================
localparam TW_QUARTER = N / 4;
localparam TW_ADDR_W  = LOG2N - 2;

(* rom_style = "block" *) reg signed [TWIDDLE_W-1:0] cos_rom [0:TW_QUARTER-1];

initial begin
    $readmemh(TWIDDLE_FILE, cos_rom);
end

// ============================================================================
// BIT-REVERSE
// ============================================================================
function [LOG2N-1:0] bit_reverse;
    input [LOG2N-1:0] val;
    integer b;
    begin
        bit_reverse = 0;
        for (b = 0; b < LOG2N; b = b + 1)
            bit_reverse[LOG2N-1-b] = val[b];
    end
endfunction

// ============================================================================
// COUNTERS AND PIPELINE REGISTERS
// ============================================================================
reg [LOG2N-1:0] load_count;
reg [LOG2N:0]   out_count;
reg [LOG2N-1:0] bfly_count;
reg [3:0]       stage;

// Registered values (captured in BF_READ, used in BF_CALC)
reg signed [TWIDDLE_W-1:0]  rd_tw_cos, rd_tw_sin;
reg [LOG2N-1:0] rd_addr_even, rd_addr_odd;
reg rd_inverse;

// Half and twiddle stride
reg [LOG2N-1:0] half_reg;
reg [LOG2N-1:0] tw_stride_reg;

// ============================================================================
// BUTTERFLY ADDRESS COMPUTATION (combinational)
// ============================================================================
reg [LOG2N-1:0] bf_addr_even;
reg [LOG2N-1:0] bf_addr_odd;
reg [LOG2N-1:0] bf_tw_idx;

always @(*) begin : bf_addr_calc
    reg [LOG2N-1:0] half_val;
    reg [LOG2N-1:0] idx_val;
    reg [LOG2N-1:0] grp_val;

    half_val  = half_reg;
    idx_val   = bfly_count & (half_val - 1);
    grp_val   = (bfly_count - idx_val);

    bf_addr_even = (grp_val << 1) | idx_val;
    bf_addr_odd  = bf_addr_even + half_val;

    bf_tw_idx = idx_val * tw_stride_reg;
end

// ============================================================================
// TWIDDLE LOOKUP (combinational)
// ============================================================================
reg signed [TWIDDLE_W-1:0] tw_cos_lookup;
reg signed [TWIDDLE_W-1:0] tw_sin_lookup;

always @(*) begin : tw_lookup
    reg [LOG2N-1:0] k;
    reg [LOG2N-1:0] rom_idx;

    k = bf_tw_idx;
    tw_cos_lookup = 0;
    tw_sin_lookup = 0;

    if (k == 0) begin
        tw_cos_lookup = cos_rom[0];
        tw_sin_lookup = {TWIDDLE_W{1'b0}};
    end else if (k == FFT_N_QTR[LOG2N-1:0]) begin
        tw_cos_lookup = {TWIDDLE_W{1'b0}};
        tw_sin_lookup = cos_rom[0];
    end else if (k < FFT_N_QTR[LOG2N-1:0]) begin
        tw_cos_lookup = cos_rom[k[TW_ADDR_W-1:0]];
        rom_idx = FFT_N_QTR[LOG2N-1:0] - k;
        tw_sin_lookup = cos_rom[rom_idx[TW_ADDR_W-1:0]];
    end else begin
        rom_idx = k - FFT_N_QTR[LOG2N-1:0];
        tw_sin_lookup = cos_rom[rom_idx[TW_ADDR_W-1:0]];
        rom_idx = FFT_N_HALF[LOG2N-1:0] - k;
        tw_cos_lookup = -cos_rom[rom_idx[TW_ADDR_W-1:0]];
    end
end

// ============================================================================
// SATURATION
// ============================================================================
function signed [DATA_W-1:0] saturate;
    input signed [INTERNAL_W-1:0] val;
    reg signed [INTERNAL_W-1:0] max_pos;
    reg signed [INTERNAL_W-1:0] max_neg;
    begin
        max_pos = (1 << (DATA_W - 1)) - 1;
        max_neg = -(1 << (DATA_W - 1));
        if (val > max_pos)
            saturate = max_pos[DATA_W-1:0];
        else if (val < max_neg)
            saturate = max_neg[DATA_W-1:0];
        else
            saturate = val[DATA_W-1:0];
    end
endfunction

// ============================================================================
// BUTTERFLY COMPUTATION (combinational, for BF_CALC write data)
// ============================================================================
reg signed [INTERNAL_W-1:0] bf_t_re, bf_t_im;
reg signed [INTERNAL_W-1:0] bf_sum_re, bf_sum_im;
reg signed [INTERNAL_W-1:0] bf_dif_re, bf_dif_im;

always @(*) begin : bf_compute
    if (!rd_inverse) begin
        bf_t_re = (mem_rdata_b_re * rd_tw_cos + mem_rdata_b_im * rd_tw_sin) >>> (TWIDDLE_W - 1);
        bf_t_im = (mem_rdata_b_im * rd_tw_cos - mem_rdata_b_re * rd_tw_sin) >>> (TWIDDLE_W - 1);
    end else begin
        bf_t_re = (mem_rdata_b_re * rd_tw_cos - mem_rdata_b_im * rd_tw_sin) >>> (TWIDDLE_W - 1);
        bf_t_im = (mem_rdata_b_im * rd_tw_cos + mem_rdata_b_re * rd_tw_sin) >>> (TWIDDLE_W - 1);
    end
    bf_sum_re = mem_rdata_a_re + bf_t_re;
    bf_sum_im = mem_rdata_a_im + bf_t_im;
    bf_dif_re = mem_rdata_a_re - bf_t_re;
    bf_dif_im = mem_rdata_a_im - bf_t_im;
end

// ============================================================================
// BRAM PORT ADDRESS / WE / WDATA — combinational mux (registered signals)
// ============================================================================
// Drives port A and port B control signals from FSM state.
// These are registered (via NBA) so they are stable at the next posedge
// when the BRAM template blocks sample them. This avoids any NBA race.
// ============================================================================
reg                          bram_we_a;
reg  [LOG2N-1:0]             bram_addr_a;
reg  signed [INTERNAL_W-1:0] bram_wdata_a_re;
reg  signed [INTERNAL_W-1:0] bram_wdata_a_im;

reg                          bram_we_b;
reg  [LOG2N-1:0]             bram_addr_b;
reg  signed [INTERNAL_W-1:0] bram_wdata_b_re;
reg  signed [INTERNAL_W-1:0] bram_wdata_b_im;

always @(*) begin : bram_port_mux
    // Port A defaults
    bram_we_a       = 1'b0;
    bram_addr_a     = 0;
    bram_wdata_a_re = 0;
    bram_wdata_a_im = 0;

    // Port B defaults
    bram_we_b       = 1'b0;
    bram_addr_b     = 0;
    bram_wdata_b_re = 0;
    bram_wdata_b_im = 0;

    case (state)
    ST_LOAD: begin
        bram_we_a       = din_valid;
        bram_addr_a     = bit_reverse(load_count);
        bram_wdata_a_re = {{(INTERNAL_W-DATA_W){din_re[DATA_W-1]}}, din_re};
        bram_wdata_a_im = {{(INTERNAL_W-DATA_W){din_im[DATA_W-1]}}, din_im};
    end
    ST_BF_READ: begin
        bram_addr_a = bf_addr_even;
        bram_addr_b = bf_addr_odd;
    end
    ST_BF_CALC: begin
        bram_we_a       = 1'b1;
        bram_addr_a     = rd_addr_even;
        bram_wdata_a_re = bf_sum_re;
        bram_wdata_a_im = bf_sum_im;

        bram_we_b       = 1'b1;
        bram_addr_b     = rd_addr_odd;
        bram_wdata_b_re = bf_dif_re;
        bram_wdata_b_im = bf_dif_im;
    end
    ST_OUTPUT: begin
        bram_addr_a = out_count[LOG2N-1:0];
    end
    default: begin
        // keep defaults
    end
    endcase
end

// ============================================================================
// DATA MEMORY — True Dual-Port BRAM
// ============================================================================
// For synthesis: xpm_memory_tdpram (Xilinx Parameterized Macros)
// For simulation: behavioral Verilog-2001 model (Icarus-compatible)
// ============================================================================

// XPM read-data wires (directly assigned to rdata regs below)
wire [INTERNAL_W-1:0] xpm_douta_re, xpm_doutb_re;
wire [INTERNAL_W-1:0] xpm_douta_im, xpm_doutb_im;

always @(*) begin
    mem_rdata_a_re = $signed(xpm_douta_re);
    mem_rdata_a_im = $signed(xpm_douta_im);
    mem_rdata_b_re = $signed(xpm_doutb_re);
    mem_rdata_b_im = $signed(xpm_doutb_im);
end

`ifndef FFT_XPM_BRAM
// ----------------------------------------------------------------------------
// Default: behavioral TDP model (works with Icarus Verilog -g2001)
// For Vivado synthesis, define FFT_XPM_BRAM to use xpm_memory_tdpram.
// ----------------------------------------------------------------------------
reg [INTERNAL_W-1:0] sim_mem_re [0:N-1];
reg [INTERNAL_W-1:0] sim_mem_im [0:N-1];

// Port A
reg [INTERNAL_W-1:0] sim_douta_re, sim_douta_im;
always @(posedge clk) begin
    if (bram_we_a) begin
        sim_mem_re[bram_addr_a] <= bram_wdata_a_re;
        sim_mem_im[bram_addr_a] <= bram_wdata_a_im;
    end
    sim_douta_re <= sim_mem_re[bram_addr_a];
    sim_douta_im <= sim_mem_im[bram_addr_a];
end
assign xpm_douta_re = sim_douta_re;
assign xpm_douta_im = sim_douta_im;

// Port B
reg [INTERNAL_W-1:0] sim_doutb_re, sim_doutb_im;
always @(posedge clk) begin
    if (bram_we_b) begin
        sim_mem_re[bram_addr_b] <= bram_wdata_b_re;
        sim_mem_im[bram_addr_b] <= bram_wdata_b_im;
    end
    sim_doutb_re <= sim_mem_re[bram_addr_b];
    sim_doutb_im <= sim_mem_im[bram_addr_b];
end
assign xpm_doutb_re = sim_doutb_re;
assign xpm_doutb_im = sim_doutb_im;

integer init_i;
initial begin
    for (init_i = 0; init_i < N; init_i = init_i + 1) begin
        sim_mem_re[init_i] = 0;
        sim_mem_im[init_i] = 0;
    end
end

`else
// ----------------------------------------------------------------------------
// Synthesis: xpm_memory_tdpram — guaranteed BRAM mapping
// Enabled when FFT_XPM_BRAM is defined (e.g. in Vivado TCL script).
// ----------------------------------------------------------------------------
// Note: Vivado auto-finds XPM library; no `include needed.
// Two instances: one for real, one for imaginary.
// WRITE_MODE = "write_first" matches the behavioral TDP template.
// READ_LATENCY = 1 (registered output).
// ----------------------------------------------------------------------------

xpm_memory_tdpram #(
    .ADDR_WIDTH_A        (LOG2N),
    .ADDR_WIDTH_B        (LOG2N),
    .AUTO_SLEEP_TIME     (0),
    .BYTE_WRITE_WIDTH_A  (INTERNAL_W),
    .BYTE_WRITE_WIDTH_B  (INTERNAL_W),
    .CASCADE_HEIGHT      (0),
    .CLOCKING_MODE       ("common_clock"),
    .ECC_BIT_RANGE       ("7:0"),
    .ECC_MODE            ("no_ecc"),
    .ECC_TYPE            ("none"),
    .IGNORE_INIT_SYNTH   (0),
    .MEMORY_INIT_FILE    ("none"),
    .MEMORY_INIT_PARAM   ("0"),
    .MEMORY_OPTIMIZATION ("true"),
    .MEMORY_PRIMITIVE     ("block"),
    .MEMORY_SIZE         (N * INTERNAL_W),
    .MESSAGE_CONTROL     (0),
    .RAM_DECOMP          ("auto"),
    .READ_DATA_WIDTH_A   (INTERNAL_W),
    .READ_DATA_WIDTH_B   (INTERNAL_W),
    .READ_LATENCY_A      (1),
    .READ_LATENCY_B      (1),
    .READ_RESET_VALUE_A  ("0"),
    .READ_RESET_VALUE_B  ("0"),
    .RST_MODE_A          ("SYNC"),
    .RST_MODE_B          ("SYNC"),
    .SIM_ASSERT_CHK      (0),
    .USE_EMBEDDED_CONSTRAINT (0),
    .USE_MEM_INIT        (1),
    .USE_MEM_INIT_MMI    (0),
    .WAKEUP_TIME         ("disable_sleep"),
    .WRITE_DATA_WIDTH_A  (INTERNAL_W),
    .WRITE_DATA_WIDTH_B  (INTERNAL_W),
    .WRITE_MODE_A        ("read_first"),
    .WRITE_MODE_B        ("read_first"),
    .WRITE_PROTECT       (1)
) u_bram_re (
    .clka            (clk),
    .clkb            (clk),
    .rsta            (1'b0),
    .rstb            (1'b0),
    .ena             (1'b1),
    .enb             (1'b1),
    .regcea          (1'b1),
    .regceb          (1'b1),
    .addra           (bram_addr_a),
    .addrb           (bram_addr_b),
    .dina            (bram_wdata_a_re),
    .dinb            (bram_wdata_b_re),
    .wea             (bram_we_a),
    .web             (bram_we_b),
    .douta           (xpm_douta_re),
    .doutb           (xpm_doutb_re),
    .injectdbiterra  (1'b0),
    .injectdbiterrb  (1'b0),
    .injectsbiterra  (1'b0),
    .injectsbiterrb  (1'b0),
    .sbiterra        (),
    .sbiterrb        (),
    .dbiterra        (),
    .dbiterrb        (),
    .sleep           (1'b0)
);

xpm_memory_tdpram #(
    .ADDR_WIDTH_A        (LOG2N),
    .ADDR_WIDTH_B        (LOG2N),
    .AUTO_SLEEP_TIME     (0),
    .BYTE_WRITE_WIDTH_A  (INTERNAL_W),
    .BYTE_WRITE_WIDTH_B  (INTERNAL_W),
    .CASCADE_HEIGHT      (0),
    .CLOCKING_MODE       ("common_clock"),
    .ECC_BIT_RANGE       ("7:0"),
    .ECC_MODE            ("no_ecc"),
    .ECC_TYPE            ("none"),
    .IGNORE_INIT_SYNTH   (0),
    .MEMORY_INIT_FILE    ("none"),
    .MEMORY_INIT_PARAM   ("0"),
    .MEMORY_OPTIMIZATION ("true"),
    .MEMORY_PRIMITIVE     ("block"),
    .MEMORY_SIZE         (N * INTERNAL_W),
    .MESSAGE_CONTROL     (0),
    .RAM_DECOMP          ("auto"),
    .READ_DATA_WIDTH_A   (INTERNAL_W),
    .READ_DATA_WIDTH_B   (INTERNAL_W),
    .READ_LATENCY_A      (1),
    .READ_LATENCY_B      (1),
    .READ_RESET_VALUE_A  ("0"),
    .READ_RESET_VALUE_B  ("0"),
    .RST_MODE_A          ("SYNC"),
    .RST_MODE_B          ("SYNC"),
    .SIM_ASSERT_CHK      (0),
    .USE_EMBEDDED_CONSTRAINT (0),
    .USE_MEM_INIT        (1),
    .USE_MEM_INIT_MMI    (0),
    .WAKEUP_TIME         ("disable_sleep"),
    .WRITE_DATA_WIDTH_A  (INTERNAL_W),
    .WRITE_DATA_WIDTH_B  (INTERNAL_W),
    .WRITE_MODE_A        ("read_first"),
    .WRITE_MODE_B        ("read_first"),
    .WRITE_PROTECT       (1)
) u_bram_im (
    .clka            (clk),
    .clkb            (clk),
    .rsta            (1'b0),
    .rstb            (1'b0),
    .ena             (1'b1),
    .enb             (1'b1),
    .regcea          (1'b1),
    .regceb          (1'b1),
    .addra           (bram_addr_a),
    .addrb           (bram_addr_b),
    .dina            (bram_wdata_a_im),
    .dinb            (bram_wdata_b_im),
    .wea             (bram_we_a),
    .web             (bram_we_b),
    .douta           (xpm_douta_im),
    .doutb           (xpm_doutb_im),
    .injectdbiterra  (1'b0),
    .injectdbiterrb  (1'b0),
    .injectsbiterra  (1'b0),
    .injectsbiterrb  (1'b0),
    .sbiterra        (),
    .sbiterrb        (),
    .dbiterra        (),
    .dbiterrb        (),
    .sleep           (1'b0)
);

`endif

// ============================================================================
// OUTPUT PIPELINE
// ============================================================================
reg out_pipe_valid;
reg out_pipe_inverse;

always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        out_pipe_valid   <= 1'b0;
        out_pipe_inverse <= 1'b0;
    end else begin
        out_pipe_valid   <= (state == ST_OUTPUT) && (out_count <= FFT_N_M1[LOG2N-1:0]);
        out_pipe_inverse <= inverse;
    end
end

// ============================================================================
// MAIN FSM
// ============================================================================
always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        state          <= ST_IDLE;
        load_count     <= 0;
        out_count      <= 0;
        bfly_count     <= 0;
        stage          <= 0;
        half_reg       <= 1;
        tw_stride_reg  <= FFT_N_HALF[LOG2N-1:0];
        dout_re        <= 0;
        dout_im        <= 0;
        dout_valid     <= 0;
        done           <= 0;
        rd_tw_cos      <= 0;
        rd_tw_sin      <= 0;
        rd_addr_even   <= 0;
        rd_addr_odd    <= 0;
        rd_inverse     <= 0;
    end else begin
        dout_valid <= 1'b0;
        done       <= 1'b0;

        case (state)

        ST_IDLE: begin
            if (start) begin
                state      <= ST_LOAD;
                load_count <= 0;
            end
        end

        ST_LOAD: begin
            if (din_valid) begin
                if (load_count == FFT_N_M1[LOG2N-1:0]) begin
                    state         <= ST_BF_READ;
                    stage         <= 0;
                    bfly_count    <= 0;
                    half_reg      <= 1;
                    tw_stride_reg <= FFT_N_HALF[LOG2N-1:0];
                end else begin
                    load_count <= load_count + 1;
                end
            end
        end

        ST_BF_READ: begin
            rd_tw_cos    <= tw_cos_lookup;
            rd_tw_sin    <= tw_sin_lookup;
            rd_addr_even <= bf_addr_even;
            rd_addr_odd  <= bf_addr_odd;
            rd_inverse   <= inverse;
            state        <= ST_BF_CALC;
        end

        ST_BF_CALC: begin
            if (bfly_count == FFT_N_HALF_M1[LOG2N-1:0]) begin
                bfly_count <= 0;
                if (stage == LOG2N - 1) begin
                    state     <= ST_OUTPUT;
                    out_count <= 0;
                end else begin
                    stage         <= stage + 1;
                    half_reg      <= half_reg << 1;
                    tw_stride_reg <= tw_stride_reg >> 1;
                    state         <= ST_BF_READ;
                end
            end else begin
                bfly_count <= bfly_count + 1;
                state      <= ST_BF_READ;
            end
        end

        ST_OUTPUT: begin
            if (out_count <= FFT_N_M1[LOG2N-1:0]) begin
                out_count <= out_count + 1;
            end

            if (out_pipe_valid) begin
                if (out_pipe_inverse) begin
                    dout_re <= saturate(mem_rdata_a_re >>> LOG2N);
                    dout_im <= saturate(mem_rdata_a_im >>> LOG2N);
                end else begin
                    dout_re <= saturate(mem_rdata_a_re);
                    dout_im <= saturate(mem_rdata_a_im);
                end
                dout_valid <= 1'b1;
            end

            if (out_count > FFT_N_M1[LOG2N-1:0] && !out_pipe_valid) begin
                state <= ST_DONE;
            end
        end

        ST_DONE: begin
            done  <= 1'b1;
            state <= ST_IDLE;
        end

        default: state <= ST_IDLE;
        endcase
    end
end

endmodule
