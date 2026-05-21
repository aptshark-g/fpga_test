// flann_top.v
// flann_top.v
// FLANN顶层集成模块
// 将flann_trig（三角基函数生成）和flann_output（加权和+NLMS更新）级联
// 【修复】原例化名flann_basis与实际模块名flann_trig不匹配，已修正
`timescale 1ns / 1ps

module flann_top (
    input           sys_clk,
    input           sys_rst_n,
    input           sample_en,
    input           en,                     // FLANN使能
    input   signed [15:0] x_input,          // 原始输入信号，Q4.12
    input   signed [15:0] error_input,      // 误差信号，Q4.12
    input   signed [15:0] mu,               // 学习步长，Q4.12
    output  signed [15:0] flann_output      // FLANN控制输出，Q4.12
);

// 内部连线
wire signed [15:0] phi [0:6];

// ==================== 基函数生成 ====================
// 【修复】原flann_basis → flann_trig，与实际模块名一致
flann_trig u_flann_trig (
    .sys_clk    (sys_clk),
    .sys_rst_n  (sys_rst_n),
    .sample_en  (sample_en),
    .x_input    (x_input),
    .phi        (phi)
);

// ==================== 输出层加权和更新 ====================
flann_output u_flann_output (
    .sys_clk        (sys_clk),
    .sys_rst_n      (sys_rst_n),
    .sample_en      (sample_en),
    .en             (en),
    .phi            (phi),
    .error_input    (error_input),
    .mu             (mu),
    .flann_output   (flann_output)
);

endmodule