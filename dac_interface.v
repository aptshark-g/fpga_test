// DAC输出接口模块
// 功能：控制信号限幅、格式转换，输出到DAC芯片
`timescale 1ns / 1ps

module dac_interface(
    input           sys_clk,
    input           sys_rst_n,
    input           sample_en,
    input   signed [15:0] control_input, // 总控制信号，Q4.12
    output  reg [15:0]  dac_data,        // DAC16位输出数据
    output  reg         dac_data_valid   // DAC数据有效
);

// 输出限幅：±10V对应DAC满量程±32767
parameter   DAC_MAX = 16'sd32767;
parameter DAC_MIN = 16'sh8000;  // 16位有符号数-32768的补码

always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        dac_data <= 16'd0;
        dac_data_valid <= 1'b0;
    end
    else if(sample_en) begin
        // 限幅处理
        if(control_input > DAC_MAX) begin
            dac_data <= DAC_MAX;
        end
        else if(control_input < DAC_MIN) begin
            dac_data <= DAC_MIN;
        end
        else begin
            dac_data <= control_input;
        end
        dac_data_valid <= 1'b1;
    end
    else begin
        dac_data_valid <= 1'b0;
    end
end

endmodule