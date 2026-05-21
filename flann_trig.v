// flann_trig.v
// flann_trig.v
// FLANN三角基函数非线性扩展模块（修复版 v2.0）
// 【修复日志】
//   1. LUT深度改为128，与7位idx索引匹配，消除Warning 10027
//   2. 显式截取x_k到8位地址，消除Warning 10230
//   3. cos_addr加法显式限制在8位，防止溢出回绕
`timescale 1ns / 1ps

module flann_trig (
    input           sys_clk,
    input           sys_rst_n,
    input           sample_en,
    input   signed [15:0] x_input,
    output  reg signed [15:0] phi [0:6]
);

parameter ORDER = 3;
parameter NUM_BASIS = 2*ORDER + 1;
parameter LUT_DEPTH = 128;   // 【修复】改为128，与7位idx匹配
parameter LUT_HALF = 128;
parameter LUT_QTR = 64;

reg signed [15:0] sin_lut [0:LUT_DEPTH-1];
initial begin
    $readmemh("flann_sin_lut.hex", sin_lut);
end

wire signed [31:0] x_ext;
assign x_ext = {{16{x_input[15]}}, x_input};

reg signed [31:0] x_k [1:3];
reg [7:0] sin_addr_raw [1:3];   // 8位原始地址
reg [7:0] cos_addr_raw [1:3];   // 8位原始地址
reg [6:0] sin_addr [1:3];       // 7位查表地址（0-127）
reg [6:0] cos_addr [1:3];       // 7位查表地址（0-127）
reg       sin_sign [1:3];       // 符号位
reg       cos_sign [1:3];       // 符号位

integer k;

always @(*) begin
    for (k = 1; k <= 3; k = k + 1) begin
        x_k[k] = x_ext * k;
        // 【修复】显式截取到8位，消除32→8截断警告
        sin_addr_raw[k] = x_k[k][12:5];
        // 【修复】cos_addr = sin_addr + 64，显式取8位防止溢出
        cos_addr_raw[k] = (sin_addr_raw[k] + 8'd64) & 8'hFF;

        // 符号：最高位
        sin_sign[k] = sin_addr_raw[k][7];
        cos_sign[k] = cos_addr_raw[k][7];
        // 索引：低7位
        sin_addr[k] = sin_addr_raw[k][6:0];
        cos_addr[k] = cos_addr_raw[k][6:0];
    end
end

always @(posedge sys_clk or negedge sys_rst_n) begin
    if (!sys_rst_n) begin
        for (k = 0; k < NUM_BASIS; k = k + 1) phi[k] <= 16'sd0;
    end
    else if (sample_en) begin
        // 1倍频
        phi[0] <= sin_sign[1] ? (-sin_lut[sin_addr[1]]) : sin_lut[sin_addr[1]];
        phi[1] <= cos_sign[1] ? (-sin_lut[cos_addr[1]]) : sin_lut[cos_addr[1]];
        // 2倍频
        phi[2] <= sin_sign[2] ? (-sin_lut[sin_addr[2]]) : sin_lut[sin_addr[2]];
        phi[3] <= cos_sign[2] ? (-sin_lut[cos_addr[2]]) : sin_lut[cos_addr[2]];
        // 3倍频
        phi[4] <= sin_sign[3] ? (-sin_lut[sin_addr[3]]) : sin_lut[sin_addr[3]];
        phi[5] <= cos_sign[3] ? (-sin_lut[cos_addr[3]]) : sin_lut[cos_addr[3]];
        // 直通
        phi[6] <= x_input;
    end
end

endmodule