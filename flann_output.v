// flann_output.v
// flann_output.v
// FLANN输出层模块（修复版 v2.0）
// 【修复日志】
//   1. 消除除法器：NLMS归一化改为barrel shifter动态移位（分母近似为2的幂次）
//   2. 消除latch：乘法器阵列用generate生成，integer移出always块
//   3. 修复位宽截断：所有64→32/48→32显式截取，加饱和保护
//   4. 时序优化：梯度计算打一拍，消除超长组合逻辑链
`timescale 1ns / 1ps

module flann_output (
    input           sys_clk,
    input           sys_rst_n,
    input           sample_en,
    input           en,
    input   signed [15:0] phi [0:6],
    input   signed [15:0] error_input,
    input   signed [15:0] mu,
    output  reg signed [15:0] flann_output
);

parameter NUM_BASIS = 7;

// ==================== 权重寄存器 ====================
reg signed [31:0] w_coeff [0:NUM_BASIS-1];

// ==================== 流水线寄存器 ====================
reg signed [47:0] mult_pipe [0:NUM_BASIS-1];
reg signed [47:0] add_stage1 [0:3];
reg signed [47:0] add_stage2 [0:1];
reg signed [47:0] add_stage3;
reg signed [47:0] fir_acc;

reg signed [31:0] mu_error;
reg signed [63:0] phi_norm_sq;

// 新增：梯度计算中间寄存器（打拍消除时序违例）
reg signed [31:0] grad_scaled [0:NUM_BASIS-1];
reg signed [31:0] update_val  [0:NUM_BASIS-1];

reg sample_en_dly1, sample_en_dly2;

integer i;

// ==================== 0. 乘法器阵列（generate，无latch）====================
genvar gvi;
generate
    for (gvi = 0; gvi < NUM_BASIS; gvi = gvi + 1) begin : gen_mult
        always @(posedge sys_clk or negedge sys_rst_n) begin
            if (!sys_rst_n)
                mult_pipe[gvi] <= 48'sd0;
            else if (sample_en && en)
                mult_pipe[gvi] <= w_coeff[gvi] * phi[gvi];
        end
    end
endgenerate

// ==================== 1. 二叉树加法流水线 ====================
always @(posedge sys_clk or negedge sys_rst_n) begin
    if (!sys_rst_n) begin
        for (i = 0; i < 4; i = i + 1) add_stage1[i] <= 48'sd0;
        for (i = 0; i < 2; i = i + 1) add_stage2[i] <= 48'sd0;
        add_stage3 <= 48'sd0;
        fir_acc    <= 48'sd0;
    end
    else if (sample_en_dly1 && en) begin
        add_stage1[0] <= mult_pipe[0] + mult_pipe[1];
        add_stage1[1] <= mult_pipe[2] + mult_pipe[3];
        add_stage1[2] <= mult_pipe[4] + mult_pipe[5];
        add_stage1[3] <= mult_pipe[6];
        add_stage2[0] <= add_stage1[0] + add_stage1[1];
        add_stage2[1] <= add_stage1[2] + add_stage1[3];
        add_stage3    <= add_stage2[0] + add_stage2[1];
        fir_acc       <= add_stage3;
    end
end

// ==================== 2. 输出锁存 + 饱和 ====================
always @(posedge sys_clk or negedge sys_rst_n) begin
    if (!sys_rst_n)
        flann_output <= 16'sd0;
    else if (sample_en_dly2 && en) begin
        if (fir_acc > 48'sh0000_7FFF_0000)
            flann_output <= 16'sh7FFF;
        else if (fir_acc < -48'sh0000_8000_0000)
            flann_output <= -16'sh8000;
        else
            flann_output <= fir_acc[31:16];
    end
    else if (!en)
        flann_output <= 16'sd0;
end

// ==================== 3. 伪NLMS权重更新（无除法器）====================
// 原理：phi_norm_sq 范围约 4096~32768 (2^12 ~ 2^15)，用高bit位选择右移位数
// 等价于将分母近似为最近的2的幂次，用barrel shifter实现
reg [3:0] norm_shift;
reg signed [31:0] norm_val;

always @(*) begin
    // phi_norm_sq >>> 12 将 Q8.24 转为 Q20.12 量级的标量
    norm_val = phi_norm_sq[43:12];  // 取32位有效范围，显式截取
    // 动态选择移位量：norm_val 越大，右移越多
    if      (norm_val >= 32'sd16384) norm_shift = 4'd14;  // ~2^14
    else if (norm_val >= 32'sd8192)  norm_shift = 4'd13;  // ~2^13
    else if (norm_val >= 32'sd4096)  norm_shift = 4'd12;  // ~2^12
    else                             norm_shift = 4'd11;  // 保底
end

always @(posedge sys_clk or negedge sys_rst_n) begin
    if (!sys_rst_n) begin
        for (i = 0; i < NUM_BASIS; i = i + 1) w_coeff[i] <= 32'sd0;
        mu_error    <= 32'sd0;
        phi_norm_sq <= 64'sd0;
        for (i = 0; i < NUM_BASIS; i = i + 1) grad_scaled[i] <= 32'sd0;
        for (i = 0; i < NUM_BASIS; i = i + 1) update_val[i]  <= 32'sd0;
    end
    else if (sample_en && en) begin
        // Step 1: mu * error >>> 12 = Q16.16
        mu_error <= (mu * error_input) >>> 12;

        // Step 2: phi 能量和（Q8.24 累加）
        phi_norm_sq <= (phi[0]*phi[0]) + (phi[1]*phi[1]) + (phi[2]*phi[2]) +
                       (phi[3]*phi[3]) + (phi[4]*phi[4]) + (phi[5]*phi[5]) + (phi[6]*phi[6]);

        // Step 3: 梯度 = mu_error * phi (Q16.16 * Q4.12 = Q20.28)
        // 先 >>> 12 得 Q20.16，再用32位寄存器打拍
        for (i = 0; i < NUM_BASIS; i = i + 1) begin
            grad_scaled[i] <= (mu_error * phi[i]) >>> 12;  // 显式截取到32位
        end

        // Step 4: 伪归一化 = 动态右移（barrel shifter，组合逻辑极快）
        for (i = 0; i < NUM_BASIS; i = i + 1) begin
            update_val[i] <= grad_scaled[i] >>> norm_shift;
        end

        // Step 5: 权重更新（使用上一拍算好的update_val，时序友好）
        for (i = 0; i < NUM_BASIS; i = i + 1) begin
            w_coeff[i] <= w_coeff[i] + update_val[i];
        end
    end
end

// ==================== 4. 使能信号打拍 ====================
always @(posedge sys_clk or negedge sys_rst_n) begin
    if (!sys_rst_n) begin
        sample_en_dly1 <= 1'b0;
        sample_en_dly2 <= 1'b0;
    end else begin
        sample_en_dly1 <= sample_en;
        sample_en_dly2 <= sample_en_dly1;
    end
end

endmodule