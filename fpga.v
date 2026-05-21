// fpga.v
// fpga.v
// FPGA顶层集成模块 v2.2（修复版）
// 修复：
//   1. FLANN例化由 flann_trig → flann_top（端口匹配）
//   2. mix_output改为组合逻辑，消除fxlms_core 4拍pipeline导致的时序错位
// 架构：FxLMS（线性）+ FLANN（三角基函数非线性）+ Elman RNN（时序记忆）
//       三路并联，自适应加权融合，谐波抑制可选
`timescale 1ns / 1ps

module fpga (
    // 系统时钟与复位
    input           sys_clk,
    input           sys_rst_n,
    // ADC采集接口
    input   [15:0]  adc_data,
    input           adc_data_valid,
    // DAC输出接口
    output  [15:0]  dac_data,
    output          dac_data_valid,
    // 模式配置
    input           step_mode_sel,
    input           harm_suppress_en,
    input           flann_en,
    input           rnn_en,
    input   [1:0]   mix_mode
);

// ==================== 内部信号 ====================
wire                sys_clk_global;
wire                pll_locked;
wire                sys_rst_n_sync;
wire                sample_en;

wire signed [15:0]  vib_raw;
wire signed [15:0]  error_signal;
wire signed [15:0]  xf_signal;

wire signed [15:0]  fxlms_output;
wire signed [15:0]  harm_output;
wire signed [15:0]  flann_output;
wire signed [15:0]  rnn_output;

wire signed [15:0]  nonlinear_sum;
wire signed [15:0]  mix_output;
wire signed [15:0]  control_total;

// 自适应混合权重
reg  signed [15:0]  rho;
reg  signed [15:0]  abs_error;
reg  signed [15:0]  nl_score;

// ==================== 0. 时钟/复位 ====================
rst_sync u_rst_sync (
    .sys_clk        (sys_clk_global),
    .sys_rst_n      (sys_rst_n),
    .sys_rst_n_sync (sys_rst_n_sync)
);

clk_pll u_clk_pll (
    .sys_clk_in     (sys_clk),
    .sys_rst_n      (sys_rst_n),
    .sys_clk        (sys_clk_global),
    .pll_locked     (pll_locked),
    .sample_en      (sample_en)
);

// ==================== 1. ADC接口 ====================
adc_interface u_adc_interface (
    .sys_clk        (sys_clk_global),
    .sys_rst_n      (sys_rst_n_sync),
    .adc_data       (adc_data),
    .adc_data_valid (adc_data_valid),
    .sample_en      (sample_en),
    .dac_data       (control_total),
    .vib_raw        (vib_raw),
    .error_signal   (error_signal)
);

// ==================== 2. 次级路径滤波 ====================
s_hat_filter u_s_hat_filter (
    .sys_clk        (sys_clk_global),
    .sys_rst_n      (sys_rst_n_sync),
    .sample_en      (sample_en),
    .x_input        (vib_raw),
    .xf_output      (xf_signal)
);

// ==================== 3. FxLMS核心（线性支路）====================
fxlms_core u_fxlms_core (
    .sys_clk        (sys_clk_global),
    .sys_rst_n      (sys_rst_n_sync),
    .sample_en      (sample_en),
    .x_input        (vib_raw),
    .xf_input       (xf_signal),
    .error_input    (error_signal),
    .step_mode_sel  (step_mode_sel),
    .fxlms_output   (fxlms_output)
);

// ==================== 4. 谐波抑制（可选）====================
harmonic_suppress u_harmonic_suppress (
    .sys_clk        (sys_clk_global),
    .sys_rst_n      (sys_rst_n_sync),
    .sample_en      (sample_en),
    .en             (harm_suppress_en),
    .current_phase  (vib_raw[15:0]),
    .error_input    (error_signal),
    .xf_input       (xf_signal),
    .harm_output    (harm_output)
);

// ==================== 5. FLANN三角基函数非线性支路 ====================
// 【修复】原例化 flann_trig 端口不匹配（flann_trig无en/error_input/mu/flann_output端口）
// 改为例化 flann_top（集成 flann_trig + flann_output）
flann_top u_flann_top (
    .sys_clk        (sys_clk_global),
    .sys_rst_n      (sys_rst_n_sync),
    .sample_en      (sample_en),
    .en             (flann_en),
    .x_input        (vib_raw),
    .error_input    (error_signal),
    .mu             (16'sd2048),
    .flann_output   (flann_output)
);

// ==================== 6. Elman RNN时序记忆支路 ====================
elman_rnn u_elman_rnn (
    .sys_clk        (sys_clk_global),
    .sys_rst_n      (sys_rst_n_sync),
    .sample_en      (sample_en),
    .en             (rnn_en),
    .x_input        (vib_raw),
    .error_input    (error_signal),
    .mu             (16'sd1024),
    .rnn_output     (rnn_output)
);

// ==================== 7. 非线性支路合成（FLANN + RNN 并联和）====================
wire signed [31:0] nl_sum_raw = flann_output + rnn_output;
assign nonlinear_sum = (nl_sum_raw > 32'sd32767)  ? 16'sh7FFF :
                       (nl_sum_raw < -32'sd32768) ? -16'sh8000 :
                       nl_sum_raw[15:0];

// ==================== 8. 自适应混合加权策略 ====================
always @(posedge sys_clk or negedge sys_rst_n) begin
    if (!sys_rst_n) begin
        rho      <= 16'sh3000;
        abs_error<= 16'sd0;
        nl_score <= 16'sd0;
    end
    else if (sample_en) begin
        abs_error <= error_signal[15] ? (-error_signal) : error_signal;

        case (mix_mode)
            2'b00: rho <= 16'sh1000;

            2'b01: begin
                if (!flann_en) rho <= 16'sh1000;
                else if (abs_error > 16'sd2048)  rho <= 16'sh0C00;
                else if (abs_error > 16'sd1024)  rho <= 16'sh0800;
                else                             rho <= 16'sh0400;
            end

            2'b10: begin
                if (!rnn_en) rho <= 16'sh1000;
                else if (abs_error > 16'sd2048)  rho <= 16'sh0E00;
                else if (abs_error > 16'sd512)   rho <= 16'sh0600;
                else                             rho <= 16'sh0200;
            end

            2'b11: begin
                nl_score <= abs_error;
                if (!flann_en && !rnn_en) rho <= 16'sh1000;
                else if (nl_score > 16'sd4096)   rho <= 16'sh0F00;
                else if (nl_score > 16'sd2048)   rho <= 16'sh0A00;
                else if (nl_score > 16'sd1024)   rho <= 16'sh0600;
                else if (nl_score > 16'sd512)    rho <= 16'sh0300;
                else                             rho <= 16'sh0100;
            end
        endcase
    end
end

// ==================== 9. 混合输出计算（组合逻辑，无时序错位）====================
reg signed [15:0] nonlinear_selected;
always @(*) begin
    case (mix_mode)
        2'b00:   nonlinear_selected = 16'sd0;
        2'b01:   nonlinear_selected = flann_output;
        2'b10:   nonlinear_selected = rnn_output;
        2'b11:   nonlinear_selected = nonlinear_sum;
        default: nonlinear_selected = 16'sd0;
    endcase
end

/*
wire signed [31:0] mix_lin  = rho * fxlms_output;
wire signed [31:0] mix_nlin = (16'sd4096 - rho) * nonlinear_selected;
wire signed [31:0] mix_sum  = mix_lin + mix_nlin;

// 【修复】mix_output由组合逻辑直接驱动，消除sample_en_d3锁存导致的时序错位
// fxlms_core有4拍内部pipeline，flann_top有2拍，rnn有1拍
// 由于算法周期为5000个时钟周期（10kHz），单拍20ns的延迟差异可忽略
// 组合逻辑保证各支路输出在sample_en更新后实时可见
assign mix_output = (mix_sum > 32'sh007FFF00) ? 16'sh7FFF :
                    (mix_sum < -32'sh00800000) ? -16'sh8000 :
                    mix_sum[23:8];

						  */
						  
// ==================== 新增：打一拍（Pipeline）寄存器 ====================
// 将 rho 的长组合逻辑打断，分配 1 个时钟周期（20ns）给乘法器，彻底解决时序违例
reg signed [31:0] mix_lin_reg;
reg signed [31:0] mix_nlin_reg;

always @(posedge sys_clk or negedge sys_rst_n) begin
    if (!sys_rst_n) begin
        mix_lin_reg  <= 32'sd0;
        mix_nlin_reg <= 32'sd0;
    end else begin
        // 由 sys_clk 时钟驱动乘法计算
        mix_lin_reg  <= rho * fxlms_output;
        mix_nlin_reg <= (16'sd4096 - rho) * nonlinear_selected;
    end
end

// 加法器使用上一拍乘法算好的寄存器结果
wire signed [31:0] mix_sum  = mix_lin_reg + mix_nlin_reg;

// 限幅输出依然保持组合逻辑
assign mix_output = (mix_sum > 32'sh007FFF00) ?
                    16'sh7FFF :
                    (mix_sum < -32'sh00800000) ?
                    -16'sh8000 :
                    mix_sum[23:8];
						  
						  
// ==================== 10. 总控制输出 + DAC ====================
wire signed [31:0] total_raw = mix_output + harm_output;
wire signed [15:0] total_sat;
assign total_sat = (total_raw > 32'sd32767)  ? 16'sh7FFF :
                   (total_raw < -32'sd32768) ? -16'sh8000 :
                   total_raw[15:0];
assign control_total = total_sat;

dac_interface u_dac_interface (
    .sys_clk        (sys_clk_global),
    .sys_rst_n      (sys_rst_n_sync),
    .sample_en      (sample_en),
    .control_input  (control_total),
    .dac_data       (dac_data),
    .dac_data_valid (dac_data_valid)
);

endmodule
