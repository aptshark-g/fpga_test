// elman_rnn.v
// elman_rnn.v
// Elman RNN 轻量化循环神经网络模块（修复版 v2.0）
// 【修复日志】
//   1. tanh_addr 32→8位显式截取，消除Warning 10230
//   2. grad_raw 64→32位显式截取，消除Warning 10230
//   3. 隐藏层net_h计算增加显式饱和，防止地址越界
`timescale 1ns / 1ps

module elman_rnn (
    input           sys_clk,
    input           sys_rst_n,
    input           sample_en,
    input           en,
    input   signed [15:0] x_input,
    input   signed [15:0] error_input,
    input   signed [15:0] mu,
    output  reg signed [15:0] rnn_output
);

parameter HIDDEN_NUM = 4;
parameter NUM_LUT = 256;

reg signed [15:0] tanh_lut [0:NUM_LUT-1];
initial begin
    $readmemh("tanh_lut.hex", tanh_lut);
end

reg signed [15:0] W_xh [0:HIDDEN_NUM-1];
reg signed [15:0] W_hh [0:HIDDEN_NUM-1][0:HIDDEN_NUM-1];
reg signed [15:0] b_h [0:HIDDEN_NUM-1];
reg signed [15:0] b_y;

reg signed [31:0] W_hy [0:HIDDEN_NUM-1];
reg signed [15:0] h_state [0:HIDDEN_NUM-1];

reg signed [31:0] net_h [0:HIDDEN_NUM-1];
reg [7:0] tanh_addr [0:HIDDEN_NUM-1];
reg signed [15:0] h_new [0:HIDDEN_NUM-1];

reg signed [47:0] out_mult [0:HIDDEN_NUM-1];
reg signed [47:0] out_acc;

// 【新增】梯度计算中间寄存器（消除64位截断）
reg signed [31:0] grad_temp [0:HIDDEN_NUM-1];

integer i, j;

// ==================== 预训练权重初始化 ====================
initial begin
    W_xh[0] = 16'sd512;  W_xh[1] = 16'sd1024;
    W_xh[2] = 16'sd256;  W_xh[3] = 16'sd768;
    for (i = 0; i < HIDDEN_NUM; i = i + 1) begin
        for (j = 0; j < HIDDEN_NUM; j = j + 1) begin
            if (i == j) W_hh[i][j] = 16'sh0C00;
            else if ((i+1)%HIDDEN_NUM == j) W_hh[i][j] = 16'sd256;
            else W_hh[i][j] = 16'sd64;
        end
    end
    b_h[0] = 16'sd0; b_h[1] = 16'sd0; b_h[2] = 16'sd0; b_h[3] = 16'sd0;
    b_y = 16'sd0;
    W_hy[0] = 32'sh0000_0400; W_hy[1] = 32'sh0000_0200;
    W_hy[2] = 32'sh0000_0600; W_hy[3] = 32'sh0000_0300;
end

// ==================== 1. 隐藏层组合逻辑 ====================
// 【修复】Quartus不支持表达式后直接位选择，用临时变量中转
reg signed [31:0] net_h_scaled;
reg [31:0] addr_temp;

always @(*) begin
    for (i = 0; i < HIDDEN_NUM; i = i + 1) begin
        net_h[i] = (W_xh[i] * x_input) + (b_h[i] <<< 12);
        for (j = 0; j < HIDDEN_NUM; j = j + 1) begin
            net_h[i] = net_h[i] + (W_hh[i][j] * h_state[j]);
        end
        
        // 分步计算，每一步都赋值给单独的变量
        net_h_scaled = net_h[i] >>> 12;
        
        if (net_h_scaled > 32'sd8192)
            tanh_addr[i] = 8'd255;
        else if (net_h_scaled < -32'sd8192)
            tanh_addr[i] = 8'd0;
        else begin
            addr_temp = net_h_scaled + 32'sd8192;
            addr_temp = addr_temp >>> 6;
            tanh_addr[i] = addr_temp[7:0]; // 只能对单独的变量名进行位选择
        end
        
        h_new[i] = tanh_lut[tanh_addr[i]];
    end
end

// ==================== 2. 隐藏状态更新 ====================
always @(posedge sys_clk or negedge sys_rst_n) begin
    if (!sys_rst_n) begin
        for (i = 0; i < HIDDEN_NUM; i = i + 1) h_state[i] <= 16'sd0;
    end
    else if (sample_en && en) begin
        for (i = 0; i < HIDDEN_NUM; i = i + 1) h_state[i] <= h_new[i];
    end
end

// ==================== 3. 输出层组合逻辑 ====================
always @(*) begin
    out_acc = 48'sd0;
    for (i = 0; i < HIDDEN_NUM; i = i + 1) begin
        out_mult[i] = W_hy[i] * h_state[i];
        out_acc = out_acc + out_mult[i];
    end
    out_acc = out_acc + ({32'sd0, b_y} <<< 16);
end

// ==================== 4. 输出锁存 + 饱和 ====================
always @(posedge sys_clk or negedge sys_rst_n) begin
    if (!sys_rst_n)
        rnn_output <= 16'sd0;
    else if (sample_en && en) begin
        if (out_acc > 48'sh0000_7FFF_0000)
            rnn_output <= 16'sh7FFF;
        else if (out_acc < -48'sh0000_8000_0000)
            rnn_output <= -16'sh8000;
        else
            rnn_output <= out_acc[31:16];
    end
    else if (!en)
        rnn_output <= 16'sd0;
end

// ==================== 5. 输出层权重更新（消除64位截断）====================
always @(posedge sys_clk or negedge sys_rst_n) begin
    if (!sys_rst_n) begin
        W_hy[0] <= 32'sh0000_0400; W_hy[1] <= 32'sh0000_0200;
        W_hy[2] <= 32'sh0000_0600; W_hy[3] <= 32'sh0000_0300;
        for (i = 0; i < HIDDEN_NUM; i = i + 1) grad_temp[i] <= 32'sd0;
    end
    else if (sample_en && en) begin
        for (i = 0; i < HIDDEN_NUM; i = i + 1) begin
            // 【修复】分步计算：先算 mu*error (Q4.12*Q4.12=Q8.24) >>> 12 = Q16.16
            // 再乘 h_state (Q4.12) >>> 16 = Q16.16，全程32位安全
            grad_temp[i] <= ((mu * error_input) >>> 12) * h_state[i] >>> 16;
            W_hy[i] <= W_hy[i] + grad_temp[i];
        end
    end
end

endmodule