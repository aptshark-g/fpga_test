// 时钟管理模块：PLL全局时钟+10kHz采样使能生成
`timescale 1ns / 1ps

module clk_pll(
    input           sys_clk_in,     // 外部输入50MHz时钟
    input           sys_rst_n,      // 外部复位，低电平有效
    output          sys_clk,        // PLL输出全局50MHz时钟
    output          pll_locked,     // PLL锁定信号，高电平=PLL稳定
    output  reg     sample_en       // 10kHz采样使能，单时钟周期高脉冲
);

// ==================== PLL IP例化 ====================
clk_pll_ip u_clk_pll_ip(
    .refclk     (sys_clk_in),   // 外部参考时钟输入
    .rst        (~sys_rst_n),   // PLL复位是高电平有效，外部低电平复位取反
    .outclk_0   (sys_clk),      // 输出50MHz全局时钟
    .locked     (pll_locked)    // PLL锁定信号
);

// ==================== 10kHz采样使能生成 ====================
// 50MHz系统时钟 → 10kHz，分频系数5000
parameter   DIV_CNT_MAX = 16'd4999;
reg [15:0]  div_cnt;

// 【修复】异步复位、同步使能的规范写法
always @(posedge sys_clk or negedge sys_rst_n) begin
    // 1. 仅异步复位信号触发复位
    if(!sys_rst_n) begin
        div_cnt     <= 16'd0;
        sample_en   <= 1'b0;
    end
    // 2. PLL未锁定时，保持复位状态（同步逻辑）
    else if(!pll_locked) begin
        div_cnt     <= 16'd0;
        sample_en   <= 1'b0;
    end
    // 3. PLL锁定后，正常计数
    else begin
        if(div_cnt == DIV_CNT_MAX) begin
            div_cnt     <= 16'd0;
            sample_en   <= 1'b1;
        end else begin
            div_cnt     <= div_cnt + 16'd1;
            sample_en   <= 1'b0;
        end
    end
end

endmodule