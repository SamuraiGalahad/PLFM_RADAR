`timescale 1ns / 1ps

module doppler_processor_optimized #(
    parameter DOPPLER_FFT_SIZE = 32,
    parameter RANGE_BINS = 64,
    parameter CHIRPS_PER_FRAME = 32,
    parameter WINDOW_TYPE = 0,            // 0=Hamming, 1=Rectangular
    parameter DATA_WIDTH = 16
)(
    input wire clk,
    input wire reset_n,
    input wire [31:0] range_data,
    input wire data_valid,
    input wire new_chirp_frame,
    output reg [31:0] doppler_output,
    output reg doppler_valid,
    output reg [4:0] doppler_bin,
    output reg [5:0] range_bin,
    output wire processing_active,
    output wire frame_complete,
    output reg [3:0] status
);

// ==============================================
// Window Coefficients (Simple Implementation)
// ==============================================
reg [DATA_WIDTH-1:0] window_coeff [0:31];

// Generate window coefficients
integer w;
initial begin
    if (WINDOW_TYPE == 0) begin
        // Pre-calculated Hamming window (Q15 format)
        window_coeff[0]  = 16'h0800; window_coeff[1]  = 16'h0862;
        window_coeff[2]  = 16'h09CB; window_coeff[3]  = 16'h0C3B;
        window_coeff[4]  = 16'h0FB2; window_coeff[5]  = 16'h142F;
        window_coeff[6]  = 16'h19B2; window_coeff[7]  = 16'h2039;
        window_coeff[8]  = 16'h27C4; window_coeff[9]  = 16'h3050;
        window_coeff[10] = 16'h39DB; window_coeff[11] = 16'h4462;
        window_coeff[12] = 16'h4FE3; window_coeff[13] = 16'h5C5A;
        window_coeff[14] = 16'h69C4; window_coeff[15] = 16'h781D;
        window_coeff[16] = 16'h7FFF; // Peak
        window_coeff[17] = 16'h781D; window_coeff[18] = 16'h69C4;
        window_coeff[19] = 16'h5C5A; window_coeff[20] = 16'h4FE3;
        window_coeff[21] = 16'h4462; window_coeff[22] = 16'h39DB;
        window_coeff[23] = 16'h3050; window_coeff[24] = 16'h27C4;
        window_coeff[25] = 16'h2039; window_coeff[26] = 16'h19B2;
        window_coeff[27] = 16'h142F; window_coeff[28] = 16'h0FB2;
        window_coeff[29] = 16'h0C3B; window_coeff[30] = 16'h09CB;
        window_coeff[31] = 16'h0862;
    end else begin
        // Rectangular window (all ones)
        for (w = 0; w < 32; w = w + 1) begin
            window_coeff[w] = 16'h7FFF;
        end
    end
end

// ==============================================
// Memory Declaration - FIXED SIZE
// ==============================================
localparam MEM_DEPTH = RANGE_BINS * CHIRPS_PER_FRAME;
(* ram_style = "block" *) reg [DATA_WIDTH-1:0] doppler_i_mem [0:MEM_DEPTH-1];
(* ram_style = "block" *) reg [DATA_WIDTH-1:0] doppler_q_mem [0:MEM_DEPTH-1];

// ==============================================
// Control Registers
// ==============================================
reg [5:0] write_range_bin;     // Changed to match RANGE_BINS width
reg [4:0] write_chirp_index;   // Changed to match CHIRPS_PER_FRAME width
reg [5:0] read_range_bin;
reg [4:0] read_doppler_index;  // Changed name for clarity
reg frame_buffer_full;
reg [9:0] chirps_received;     // Enough for up to 1024 chirps
reg [1:0] chirp_state;         // Track chirp accumulation state


// ==============================================
// FFT Interface
// ==============================================
reg fft_start;
wire fft_ready;
reg [DATA_WIDTH-1:0] fft_input_i;
reg [DATA_WIDTH-1:0] fft_input_q;
reg signed [31:0] mult_i, mult_q;  // 32-bit to avoid overflow

reg fft_input_valid;
reg fft_input_last;
wire [DATA_WIDTH-1:0] fft_output_i;
wire [DATA_WIDTH-1:0] fft_output_q;
wire fft_output_valid;
wire fft_output_last;

// ==============================================
// Addressing 
// ==============================================
wire [10:0] mem_write_addr;
wire [10:0] mem_read_addr;

// Proper address calculation using parameters
assign mem_write_addr = (write_chirp_index * RANGE_BINS) + write_range_bin;
assign mem_read_addr = (read_doppler_index * RANGE_BINS) + read_range_bin;

// Alternative organization (choose one):
// If you want range-major organization (all chirps for one range bin together):
// assign mem_write_addr = (write_range_bin * CHIRPS_PER_FRAME) + write_chirp_index;
// assign mem_read_addr = (read_range_bin * CHIRPS_PER_FRAME) + read_doppler_index;

// ==============================================
// State Machine
// ==============================================
reg [2:0] state;
localparam S_IDLE       = 3'b000;
localparam S_ACCUMULATE = 3'b001;
localparam S_LOAD_FFT   = 3'b010;
localparam S_FFT_WAIT   = 3'b011;
localparam S_OUTPUT     = 3'b100;

// Frame sync detection
reg new_chirp_frame_d1;
always @(posedge clk or negedge reset_n) begin
    if (!reset_n) new_chirp_frame_d1 <= 0;
    else new_chirp_frame_d1 <= new_chirp_frame;
end
wire frame_start_pulse = new_chirp_frame & ~new_chirp_frame_d1;

// ==============================================
// Main State Machine - FIXED
// ==============================================
reg [5:0] fft_sample_counter;
reg [9:0] processing_timeout;

// Memory write enable and data signals (extracted for BRAM inference)
reg mem_we;
reg [10:0] mem_waddr_r;
reg [DATA_WIDTH-1:0] mem_wdata_i, mem_wdata_q;

// Memory read data (registered for BRAM read latency)
reg [DATA_WIDTH-1:0] mem_rdata_i, mem_rdata_q;

// ----------------------------------------------------------
// Separate always block for memory writes — NO async reset
// in sensitivity list, so Vivado can infer Block RAM.
// ----------------------------------------------------------
always @(posedge clk) begin
    if (mem_we) begin
        doppler_i_mem[mem_waddr_r] <= mem_wdata_i;
        doppler_q_mem[mem_waddr_r] <= mem_wdata_q;
    end
    // Registered read — address driven by mem_read_addr from FSM
    mem_rdata_i <= doppler_i_mem[mem_read_addr];
    mem_rdata_q <= doppler_q_mem[mem_read_addr];
end

// ----------------------------------------------------------
// Main FSM — async reset for control registers only.
// Memory arrays are NOT touched here.
// ----------------------------------------------------------
always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        state <= S_IDLE;
        write_range_bin <= 0;
        write_chirp_index <= 0;
        read_range_bin <= 0;
        read_doppler_index <= 0;
        frame_buffer_full <= 0;
        doppler_valid <= 0;
        fft_start <= 0;
        fft_input_valid <= 0;
        fft_input_last <= 0;
        fft_sample_counter <= 0;
        processing_timeout <= 0;
        status <= 0;
        chirps_received <= 0;
        chirp_state <= 0;
        mem_we <= 0;
        mem_waddr_r <= 0;
        mem_wdata_i <= 0;
        mem_wdata_q <= 0;
        mult_i <= 0;
        mult_q <= 0;
        fft_input_i <= 0;
        fft_input_q <= 0;
        doppler_output <= 0;
        doppler_bin <= 0;
    end else begin
        doppler_valid <= 0;
        fft_input_valid <= 0;
        fft_input_last <= 0;
        mem_we <= 0;
        
        if (processing_timeout > 0) begin
            processing_timeout <= processing_timeout - 1;
        end
        
        case (state)
            S_IDLE: begin
                if (frame_start_pulse) begin
                    // Start new frame
                    write_chirp_index <= 0;
                    write_range_bin <= 0;
                    frame_buffer_full <= 0;
                    chirps_received <= 0;
                end
                
                if (data_valid && !frame_buffer_full) begin
                    state <= S_ACCUMULATE;
                    write_range_bin <= 0;
                end
            end
            
            S_ACCUMULATE: begin
                if (data_valid) begin
                    // Drive memory write signals (actual write in separate block)
                    mem_we      <= 1;
                    mem_waddr_r <= mem_write_addr;
                    mem_wdata_i <= range_data[15:0];
                    mem_wdata_q <= range_data[31:16];
                    
                    // Increment range bin
                    if (write_range_bin < RANGE_BINS - 1) begin
                        write_range_bin <= write_range_bin + 1;
                    end else begin
                        // Completed one chirp
                        write_range_bin <= 0;
                        write_chirp_index <= write_chirp_index + 1;
                        chirps_received <= chirps_received + 1;
                        
                        // Check if frame is complete
                        if (write_chirp_index >= CHIRPS_PER_FRAME - 1) begin
                            frame_buffer_full <= 1;
                            chirp_state <= 0;
                            state <= S_LOAD_FFT;
                            read_range_bin <= 0;
                            read_doppler_index <= 0;
                            fft_sample_counter <= 0;
                            fft_start <= 1;
                        end
                    end
                end 
            end
            
            S_LOAD_FFT: begin
                fft_start <= 0;
                
                if (fft_sample_counter < DOPPLER_FFT_SIZE) begin
                    // Use registered read data (one cycle latency from BRAM)
                    mult_i <= $signed(mem_rdata_i) *
                                   $signed(window_coeff[read_doppler_index]);
                    mult_q <= $signed(mem_rdata_q) *
                                   $signed(window_coeff[read_doppler_index]);
                    
                    // Round instead of truncate
                    fft_input_i <= (mult_i + (1 << 14)) >>> 15;
                    fft_input_q <= (mult_q + (1 << 14)) >>> 15;
                    
                    fft_input_valid <= 1;
                    
                    if (fft_sample_counter == DOPPLER_FFT_SIZE - 1) begin
                        fft_input_last <= 1;
                    end
                    
                    // Increment chirp index for next sample
                    read_doppler_index <= read_doppler_index + 1;
                    fft_sample_counter <= fft_sample_counter + 1;
                end else begin
                    state <= S_FFT_WAIT;
                    fft_sample_counter <= 0;
                    processing_timeout <= 100;
                end
            end
            
            S_FFT_WAIT: begin
                if (fft_output_valid) begin
                    doppler_output <= {fft_output_q[15:0], fft_output_i[15:0]};
                    doppler_bin <= fft_sample_counter;
                    range_bin <= read_range_bin;
                    doppler_valid <= 1;
                    
                    fft_sample_counter <= fft_sample_counter + 1;
                    
                    if (fft_output_last) begin
                        state <= S_OUTPUT;
                        fft_sample_counter <= 0;
                    end
                end
                
                if (processing_timeout == 0) begin
                    state <= S_OUTPUT;
                end
            end
            
            S_OUTPUT: begin
                if (read_range_bin < RANGE_BINS - 1) begin
                    read_range_bin <= read_range_bin + 1;
                    read_doppler_index <= 0;
                    state <= S_LOAD_FFT;
                    fft_start <= 1;
                end else begin
                    state <= S_IDLE;
                    frame_buffer_full <= 0;
                end
            end
            
        endcase
        
        status <= {state, frame_buffer_full};
    end
end

// ==============================================
// FFT Module
// ==============================================
xfft_32 fft_inst (
    .aclk(clk),
    .aresetn(reset_n),
    .s_axis_config_tdata(8'h01),
    .s_axis_config_tvalid(fft_start),
    .s_axis_config_tready(fft_ready),
    .s_axis_data_tdata({fft_input_q, fft_input_i}),
    .s_axis_data_tvalid(fft_input_valid),
    .s_axis_data_tlast(fft_input_last),
    .m_axis_data_tdata({fft_output_q, fft_output_i}),
    .m_axis_data_tvalid(fft_output_valid),
    .m_axis_data_tlast(fft_output_last),
    .m_axis_data_tready(1'b1)
);

// ==============================================
// Status Outputs
// ==============================================
assign processing_active = (state != S_IDLE);
assign frame_complete = (state == S_IDLE && frame_buffer_full == 0);


endmodule