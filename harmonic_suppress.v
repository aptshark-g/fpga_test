// 窄带谐波抑制模块 (语法修复版)
`timescale 1ns / 1ps

module harmonic_suppress(
    input           sys_clk,
    input           sys_rst_n,
    input           sample_en,
    input           en,                 
    input   [15:0]  current_phase,      
    input   signed [15:0] error_input,  
    input   signed [15:0] xf_input,     
    output  reg signed [15:0] harm_output 
);

// ==================== 1. LUT 定义 (修正深度为88，匹配hex文件) ====================
reg signed [15:0] sin_lut [0:87]; // 88点深度
initial begin
    $readmemh("sin_lut.hex", sin_lut);
end

// ==================== 2. 内部信号定义 (全部在always块外声明) ====================
reg signed [31:0]  w_harm [3:0];
reg signed [15:0]  x_harm [3:0];
reg [17:0] phase_2x_reg, phase_3x_reg; // 扩展位宽防止溢出
reg [15:0] phase_2x, phase_3x;
reg [1:0]  quad_2x, quad_3x;
reg [6:0]  addr_2x, addr_3x; // 7位地址适配88深度
reg signed [15:0] sin2, cos2, sin3, cos3;
reg signed [49:0] harm_acc; // 全位宽累加变量

integer i;

// ==================== 3. 时序逻辑：相位计算 ====================
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
        phase_2x <= phase_2x_reg[16:1]; // 显式截取
        phase_3x <= phase_3x_reg[17:2];
    end
end

// ==================== 4. 时序逻辑：LUT 查表 ====================
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        quad_2x <= 2'b00;
        quad_3x <= 2'b00;
        addr_2x <= 7'd0;
        addr_3x <= 7'd0;
        sin2 <= 16'sd0;
        cos2 <= 16'sd0;
        sin3 <= 16'sd0;
        cos3 <= 16'sd0;
    end 
    else if(sample_en && en) begin
        // 2倍频查表
        quad_2x <= phase_2x[15:14];
        addr_2x <= phase_2x[13:7];
        
        case(quad_2x)
            2'b00: begin sin2 <=  sin_lut[addr_2x]; cos2 <=  sin_lut[7'd87-addr_2x]; end
            2'b01: begin sin2 <=  sin_lut[7'd87-addr_2x]; cos2 <= -sin_lut[addr_2x]; end
            2'b10: begin sin2 <= -sin_lut[addr_2x]; cos2 <= -sin_lut[7'd87-addr_2x]; end
            2'b11: begin sin2 <= -sin_lut[7'd87-addr_2x]; cos2 <=  sin_lut[addr_2x]; end
        endcase

        // 3倍频查表
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

// ==================== 5. 时序逻辑：输出与更新 ====================
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        harm_output <= 16'sd0;
        x_harm[0] <= 16'sd0;
        x_harm[1] <= 16'sd0;
        x_harm[2] <= 16'sd0;
        x_harm[3] <= 16'sd0;
        harm_acc <= 50'sd0;
    end
    else if(sample_en && en) begin
        x_harm[0] <= sin2;
        x_harm[1] <= cos2;
        x_harm[2] <= sin3;
        x_harm[3] <= cos3;
        
        // 全位宽累加
        harm_acc <= (w_harm[0] * x_harm[0]) + (w_harm[1] * x_harm[1]) + 
                    (w_harm[2] * x_harm[2]) + (w_harm[3] * x_harm[3]);
        
        // 显式饱和输出
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

// ==================== 6. 权重更新 ====================
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        for (i = 0; i < 4; i = i + 1) begin
            w_harm[i] <= 32'sd0;
        end
    end
    else if(sample_en && en) begin
        for(i=0; i<4; i=i+1) begin
            w_harm[i] <= w_harm[i] - ((32'sd2147 * error_input * x_harm[i]) >>> 22);
        end
    end
end

endmodule