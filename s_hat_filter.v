// s_hat_filter.v
// s_hat_filter.v
// 次级路径S_hat滤波模块
// 功能：4拍延迟+固定增益滤波，生成FxLMS所需的Filtered-x信号
`timescale 1ns / 1ps

module s_hat_filter(
    input           sys_clk,
    input           sys_rst_n,
    input           sample_en,
    input   signed [15:0] x_input,  // 原始参考信号，Q4.12
    output  reg signed [15:0] xf_output // 滤波后xf信号，Q4.12
);

// 4级延迟抽头，匹配MATLAB的delay=4
reg signed [15:0] x_delay [3:0];
// 【修复】参数显式声明signed，避免工具按无符号处理
parameter signed [15:0] S_HAT_GAIN = 16'sd819; // -0.2*4096 = -819，Q4.12

// 延迟链实现
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        x_delay[0] <= 16'sd0;
        x_delay[1] <= 16'sd0;
        x_delay[2] <= 16'sd0;
        x_delay[3] <= 16'sd0;
        xf_output <= 16'sd0;
    end
    else if(sample_en) begin
        // 4拍延迟移位
        x_delay[0] <= x_input;
        x_delay[1] <= x_delay[0];
        x_delay[2] <= x_delay[1];
        x_delay[3] <= x_delay[2];
        // 次级路径滤波：xf = s_hat_gain * x_delay[3]
        xf_output <= (x_delay[3] * S_HAT_GAIN) >>> 12;
    end
end

endmodule
