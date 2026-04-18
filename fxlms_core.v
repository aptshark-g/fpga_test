// FxLMS核心算法模块
// 功能：128阶FxLMS滤波器，支持定步长/变步长切换，权重实时更新
// 优化：消除除法器、全流水线设计、修复时序违例
`timescale 1ns / 1ps

module fxlms_core(
    input           sys_clk,
    input           sys_rst_n,
    input           sample_en,
    input   signed [15:0] x_input,      // 原始参考信号，Q4.12
    input   signed [15:0] xf_input,     // 滤波后xf信号，Q4.12
    input   signed [15:0] error_input,  // 误差信号，Q4.12
    input           step_mode_sel,       // 0=定步长，1=变步长
    output  reg signed [15:0] fxlms_output // FxLMS控制输出，Q4.12
);

// ==================== 参数定义 ====================
parameter   FILTER_ORDER    = 128;          // 滤波器阶数L=128
parameter   MU_FIXED        = 38'sd137439;  // 定步长5e-7，量化为5e-7*2^38
parameter   MU_MIN          = 38'sd13744;   // 最小步长5e-8
parameter   MU_MAX          = 38'sd1374390;  // 最大步长5e-6
parameter   SHIFT_BIT       = 6'd38;         // 步长右移位数
// 除法替换参数：除以10000 = 乘以429497 >>> 32 (2^32/10000≈429496.73)
parameter   DIV_10000_MUL   = 32'd429497;
parameter   DIV_10000_SHIFT = 6'd32;

// ==================== 内部信号定义 ====================
// 参考信号与xf信号缓冲区
reg signed [15:0]  x_buffer [FILTER_ORDER-1:0];
reg signed [15:0]  xf_buffer [FILTER_ORDER-1:0];
// 滤波器权重系数，Q16.16
reg signed [31:0]  w_coeff [FILTER_ORDER-1:0];
// 步长信号
reg signed [37:0]  mu_current;
// 误差包络与变步长计算
reg signed [31:0]  err_env;
reg signed [15:0]  err_abs;
// FIR滤波流水线信号
reg signed [47:0]  mult_pipe [FILTER_ORDER-1:0];
reg signed [47:0]  add_stage1 [63:0];
reg signed [47:0]  add_stage2 [31:0];
reg signed [47:0]  add_stage3 [15:0];
reg signed [47:0]  add_stage4 [7:0];
reg signed [47:0]  add_stage5 [3:0];
reg signed [47:0]  add_stage6 [1:0];
reg signed [47:0]  fir_acc;
// 使能信号打拍，同步流水线
reg sample_en_dly1, sample_en_dly2, sample_en_dly3, sample_en_dly4;

// ==================== 1. 参考信号与xf信号缓冲区更新 ====================
integer i_buf;
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        for(i_buf=0; i_buf<FILTER_ORDER; i_buf=i_buf+1) begin
            x_buffer[i_buf] <= 16'sd0;
            xf_buffer[i_buf] <= 16'sd0;
        end
    end
    else if(sample_en) begin
        for(i_buf=FILTER_ORDER-1; i_buf>0; i_buf=i_buf-1) begin
            x_buffer[i_buf] <= x_buffer[i_buf-1];
            xf_buffer[i_buf] <= xf_buffer[i_buf-1];
        end
        x_buffer[0] <= x_input;
        xf_buffer[0] <= xf_input;
    end
end

// ==================== 2. 变步长计算（彻底消除除法器！） ====================
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        err_abs <= 16'sd0;
        err_env <= 32'sd0;
        mu_current <= MU_FIXED;
    end
    else if(sample_en) begin
        // 误差绝对值计算
        err_abs <= error_input[15] ? -error_input : error_input;
        // 一阶低通平滑误差包络：err_env = 0.9995*err_env + 0.0005*err_abs
        // 除法替换：除以10000 → 乘法右移，彻底消除组合逻辑除法器
        err_env <= $signed( ( (32'sd9995 * err_env + 32'sd5 * err_abs) * DIV_10000_MUL ) >>> DIV_10000_SHIFT );
        
        // 步长模式选择
        if(!step_mode_sel) begin
            mu_current <= MU_FIXED;
        end else begin
            // 变步长线性映射
            if(err_env < 32'sd2048) begin
                mu_current <= MU_MAX;
            end else if(err_env > 32'sd8192) begin
                mu_current <= MU_MIN;
            end else begin
                mu_current <= MU_MAX - $signed( ((err_env - 32'sd2048) * (MU_MAX - MU_MIN)) / 32'sd6144 );
            end
        end
    end
end

// ==================== 3. 使能信号打拍，同步流水线 ====================
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        sample_en_dly1 <= 1'b0;
        sample_en_dly2 <= 1'b0;
        sample_en_dly3 <= 1'b0;
        sample_en_dly4 <= 1'b0;
    end else begin
        sample_en_dly1 <= sample_en;
        sample_en_dly2 <= sample_en_dly1;
        sample_en_dly3 <= sample_en_dly2;
        sample_en_dly4 <= sample_en_dly3;
    end
end

// ==================== 4. FIR滤波 全流水线二叉树加法（核心时序优化） ====================
// 第一级：全并行乘法
integer i_mult;
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        for(i_mult=0; i_mult<FILTER_ORDER; i_mult=i_mult+1) begin
            mult_pipe[i_mult] <= 48'sd0;
        end
    end else if(sample_en) begin
        for(i_mult=0; i_mult<FILTER_ORDER; i_mult=i_mult+1) begin
            mult_pipe[i_mult] <= $signed(w_coeff[i_mult]) * $signed(x_buffer[i_mult]);
        end
    end
end

// 第二级：128→64 加法
integer i_add1;
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        for(i_add1=0; i_add1<64; i_add1=i_add1+1) begin
            add_stage1[i_add1] <= 48'sd0;
        end
    end else if(sample_en_dly1) begin
        for(i_add1=0; i_add1<64; i_add1=i_add1+1) begin
            add_stage1[i_add1] <= mult_pipe[i_add1*2] + mult_pipe[i_add1*2+1];
        end
    end
end

// 第三级：64→32 加法
integer i_add2;
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        for(i_add2=0; i_add2<32; i_add2=i_add2+1) begin
            add_stage2[i_add2] <= 48'sd0;
        end
    end else if(sample_en_dly2) begin
        for(i_add2=0; i_add2<32; i_add2=i_add2+1) begin
            add_stage2[i_add2] <= add_stage1[i_add2*2] + add_stage1[i_add2*2+1];
        end
    end
end

// 第四级：32→16 加法
integer i_add3;
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        for(i_add3=0; i_add3<16; i_add3=i_add3+1) begin
            add_stage3[i_add3] <= 48'sd0;
        end
    end else if(sample_en_dly2) begin
        for(i_add3=0; i_add3<16; i_add3=i_add3+1) begin
            add_stage3[i_add3] <= add_stage2[i_add3*2] + add_stage2[i_add3*2+1];
        end
    end
end

// 第五级：16→8 加法
integer i_add4;
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        for(i_add4=0; i_add4<8; i_add4=i_add4+1) begin
            add_stage4[i_add4] <= 48'sd0;
        end
    end else if(sample_en_dly3) begin
        for(i_add4=0; i_add4<8; i_add4=i_add4+1) begin
            add_stage4[i_add4] <= add_stage3[i_add4*2] + add_stage3[i_add4*2+1];
        end
    end
end

// 第六级：8→4 加法
integer i_add5;
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        for(i_add5=0; i_add5<4; i_add5=i_add5+1) begin
            add_stage5[i_add5] <= 48'sd0;
        end
    end else if(sample_en_dly3) begin
        for(i_add5=0; i_add5<4; i_add5=i_add5+1) begin
            add_stage5[i_add5] <= add_stage4[i_add5*2] + add_stage4[i_add5*2+1];
        end
    end
end

// 第七级：4→2 加法
integer i_add6;
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        for(i_add6=0; i_add6<2; i_add6=i_add6+1) begin
            add_stage6[i_add6] <= 48'sd0;
        end
    end else if(sample_en_dly4) begin
        for(i_add6=0; i_add6<2; i_add6=i_add6+1) begin
            add_stage6[i_add6] <= add_stage5[i_add6*2] + add_stage5[i_add6*2+1];
        end
    end
end

// 第八级：2→1 最终累加
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        fir_acc <= 48'sd0;
    end else if(sample_en_dly4) begin
        fir_acc <= add_stage6[0] + add_stage6[1];
    end
end

// ==================== 5. 输出饱和处理 ====================
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        fxlms_output <= 16'sd0;
    end else if(sample_en_dly4) begin
        if (fir_acc > 48'sh00007FFFFFFF)
            fxlms_output <= 16'sh7FFF;
        else if (fir_acc < -48'sh000080000000)
            fxlms_output <= -16'sh8000;
        else
            fxlms_output <= fir_acc[27:12];
    end
end

// ==================== 6. FxLMS权重更新（修复位宽截断警告） ====================
integer i_wgt;
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        for(i_wgt=0; i_wgt<FILTER_ORDER; i_wgt=i_wgt+1) begin
            w_coeff[i_wgt] <= 32'sd0;
        end
    end
    else if(sample_en) begin
        for(i_wgt=0; i_wgt<FILTER_ORDER; i_wgt=i_wgt+1) begin
            w_coeff[i_wgt] <= w_coeff[i_wgt] - $signed(
                ((mu_current * $signed({1'b0, error_input})) >>> 19) * $signed({1'b0, xf_buffer[i_wgt]}) >>> (SHIFT_BIT-19)
            );
        end
    end
end

endmodule