// harmonic_suppress.v
// harmonic_suppress.v
// 窄带谐波抑制模块（修复版 v2.0）
// 【修复日志】
//   1. 64位grad_full改为分步32位计算，消除Warning 10230
//   2. 权重更新拆为两步流水线（error*x_harm → *步长），利用5000周期余量
//   3. 所有中间变量显式声明位宽
`timescale 1ns / 1ps

module harmonic_suppress(
    input           sys_clk,
    input           sys_rst_n,
    input           sample_en,
    input           en,
    input   signed [15:0]  current_phase,
    input   signed [15:0] error_input,
    input   signed [15:0] xf_input,
    output  reg signed [15:0] harm_output
);

reg signed [15:0] sin_lut [0:87];
initial begin
    $readmemh("sin_lut.hex", sin_lut);
end

reg signed [31:0]  w_harm [3:0];
reg signed [15:0]  x_harm [3:0];
reg [17:0] phase_2x_reg, phase_3x_reg;
reg [15:0] phase_2x, phase_3x;
reg [1:0]  quad_2x, quad_3x;
reg [6:0]  addr_2x, addr_3x;
reg signed [15:0] sin2, cos2, sin3, cos3;
reg signed [49:0] harm_acc;

integer i;

// 【新增】分步梯度计算寄存器（消除64位截断）
reg signed [31:0] grad_step1 [3:0];  // error * x_harm (Q8.24 → Q8.8)
reg signed [15:0] harm_step;         // 步长增益 2147 的量化值

// ==================== 时序逻辑：相位计算 ====================
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        phase_2x_reg <= 18'd0;
        phase_3x_reg <= 18'd0;
        phase_2x <= 16'd0;
        phase_3x <= 16'd0;
    end
    else if(sample_en && en) begin
        phase_2x_reg <= current_phase * 18'd2;
        phase_3x_reg <= current_phase * 18'd3;
        phase_2x <= phase_2x_reg[16:1];
        phase_3x <= phase_3x_reg[17:2];
    end
end

// ==================== 时序逻辑：LUT 查表 ====================
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        quad_2x <= 2'b00; quad_3x <= 2'b00;
        addr_2x <= 7'd0;  addr_3x <= 7'd0;
        sin2 <= 16'sd0; cos2 <= 16'sd0;
        sin3 <= 16'sd0; cos3 <= 16'sd0;
    end
    else if(sample_en && en) begin
        quad_2x <= phase_2x[15:14];
        addr_2x <= phase_2x[13:7];
        case(quad_2x)
            2'b00: begin sin2 <=  sin_lut[addr_2x]; cos2 <=  sin_lut[7'd87-addr_2x]; end
            2'b01: begin sin2 <=  sin_lut[7'd87-addr_2x]; cos2 <= -sin_lut[addr_2x]; end
            2'b10: begin sin2 <= -sin_lut[addr_2x]; cos2 <= -sin_lut[7'd87-addr_2x]; end
            2'b11: begin sin2 <= -sin_lut[7'd87-addr_2x]; cos2 <=  sin_lut[addr_2x]; end
        endcase

        quad_3x <= phase_3x[15:14];
        addr_3x <= phase_3x[13:7];
        case(quad_3x)
            2'b00: begin sin3 <=  sin_lut[addr_3x]; cos3 <=  sin_lut[7'd87-addr_3x]; end
            2'b01: begin sin3 <=  sin_lut[7'd87-addr_3x]; cos3 <= -sin_lut[addr_3x]; end
            2'b10: begin sin3 <= -sin_lut[addr_3x]; cos3 <= -sin_lut[7'd87-addr_3x]; end
            2'b11: begin sin3 <= -sin_lut[7'd87-addr_3x]; cos3 <=  sin_lut[addr_3x]; end
        endcase
    end
end

// ==================== 时序逻辑：输出与更新 ====================
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        harm_output <= 16'sd0;
        x_harm[0] <= 16'sd0; x_harm[1] <= 16'sd0;
        x_harm[2] <= 16'sd0; x_harm[3] <= 16'sd0;
        harm_acc <= 50'sd0;
    end
    else if(sample_en && en) begin
        x_harm[0] <= sin2; x_harm[1] <= cos2;
        x_harm[2] <= sin3; x_harm[3] <= cos3;
        harm_acc <= (w_harm[0]*x_harm[0]) + (w_harm[1]*x_harm[1]) +
                    (w_harm[2]*x_harm[2]) + (w_harm[3]*x_harm[3]);
        if ((harm_acc >>> 12) > 16'sh7FFF)
            harm_output <= 16'sh7FFF;
        else if ((harm_acc >>> 12) < -16'sh8000)
            harm_output <= -16'sh8000;
        else
            harm_output <= harm_acc[27:12];
    end
    else if(!en) begin
        harm_output <= 16'sd0;
    end
end

// ==================== 权重更新（分步32位，消除64位截断）====================
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        for (i = 0; i < 4; i = i + 1) w_harm[i] <= 32'sd0;
        for (i = 0; i < 4; i = i + 1) grad_step1[i] <= 32'sd0;
        harm_step <= 16'sd2147;
    end
    else if(sample_en && en) begin
        // Step1: error(Q4.12) * x_harm(Q4.12) = Q8.24 → >>> 16 得 Q8.8 (32位安全)
        for(i=0; i<4; i=i+1) begin
            grad_step1[i] <= (error_input * x_harm[i]) >>> 16;
        end
        // Step2: 上一拍的grad_step1 * harm_step(2147) >>> 6
        // Q8.8 * Q12.0 = Q20.8 → >>> 6 = Q20.2，再截取到32位
        for(i=0; i<4; i=i+1) begin
            w_harm[i] <= w_harm[i] - ((grad_step1[i] * harm_step) >>> 6);
        end
    end
end

endmodule