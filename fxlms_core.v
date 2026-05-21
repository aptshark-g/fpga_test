// fxlms_core.v
// fxlms_core.v
// FxLMS核心算法模块
// 功能：128阶FxLMS滤波器，支持定步长/变步长切换，权重实时更新
// 优化：消除除法器、全流水线设计、修复时序违例
// 【修复日志】
//   1. 所有parameter显式声明signed，避免无符号运算错误
//   2. 变步长分支的/6144除法替换为乘法右移（*279620 >>> 32）
//   3. DIV_10000_MUL显式声明signed，修复逻辑右移bug
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
// 【修复】显式signed声明，避免工具按无符号处理
parameter signed [37:0] MU_FIXED        = 38'sd137439;  // 定步长5e-7，量化为5e-7*2^38
parameter signed [37:0] MU_MIN          = 38'sd13744;   // 最小步长5e-8
parameter signed [37:0] MU_MAX          = 38'sd1374390;  // 最大步长5e-6
parameter   SHIFT_BIT       = 6'd38;         // 步长右移位数
// 除法替换参数：除以10000 = 乘以429497 >>> 32 (2^32/10000≈429496.73)
// 【修复】显式signed声明，避免无符号乘法导致逻辑右移
parameter signed [31:0] DIV_10000_MUL   = 32'sd429497;
parameter   DIV_10000_SHIFT = 6'd32;
// 【新增】除法替换参数：除以6144 = 乘以279620 >>> 32 (2^32/6144≈279620.27)
parameter signed [31:0] DIV_6144_MUL    = 32'sd279620;
parameter   DIV_6144_SHIFT  = 6'd32;

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
//reg signed [47:0]  mult_pipe [FILTER_ORDER-1:0];
//reg signed [47:0]  add_stage1 [63:0];
//reg signed [47:0]  add_stage2 [31:0];
//reg signed [47:0]  add_stage3 [15:0];
//reg signed [47:0]  add_stage4 [7:0];
//reg signed [47:0]  add_stage5 [3:0];
//reg signed [47:0]  add_stage6 [1:0];
//reg signed [47:0]  fir_acc;
// 使能信号打拍，同步流水线
//reg sample_en_dly1, sample_en_dly2, sample_en_dly3, sample_en_dly4;

// ==================== 状态机与复用信号（替换原流水线信号）====================
localparam IDLE       = 2'd0;
localparam FIR_CALC   = 2'd1;
localparam WGT_UPDATE = 2'd2;
localparam DONE       = 2'd3;

reg [1:0] state;
reg [6:0] mac_cnt;            // 0~127计数器
reg signed [47:0] fir_acc;    // 唯一复用的乘加累加器
integer i_wgt_init;           // 权重初始化索引



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
        // 除法替换：除以10000 → 乘法右移
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
                // 【修复】原/6144除法替换为乘法右移，消除组合逻辑除法器
                mu_current <= MU_MAX - $signed(
                    (( (err_env - 32'sd2048) * (MU_MAX - MU_MIN) ) * DIV_6144_MUL) >>> DIV_6144_SHIFT
                );
            end
        end
    end
end

/*
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

// ==================== 4. FIR滤波流水线（128阶，7级二叉树加法） ====================
integer i_fir;
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        for(i_fir=0; i_fir<FILTER_ORDER; i_fir=i_fir+1) mult_pipe[i_fir] <= 48'sd0;
        for(i_fir=0; i_fir<64; i_fir=i_fir+1) add_stage1[i_fir] <= 48'sd0;
        for(i_fir=0; i_fir<32; i_fir=i_fir+1) add_stage2[i_fir] <= 48'sd0;
        for(i_fir=0; i_fir<16; i_fir=i_fir+1) add_stage3[i_fir] <= 48'sd0;
        for(i_fir=0; i_fir<8; i_fir=i_fir+1) add_stage4[i_fir] <= 48'sd0;
        for(i_fir=0; i_fir<4; i_fir=i_fir+1) add_stage5[i_fir] <= 48'sd0;
        for(i_fir=0; i_fir<2; i_fir=i_fir+1) add_stage6[i_fir] <= 48'sd0;
        fir_acc <= 48'sd0;
    end
    else if(sample_en) begin
        // 第1级：128个乘法器并行
        for(i_fir=0; i_fir<FILTER_ORDER; i_fir=i_fir+1) begin
            mult_pipe[i_fir] <= w_coeff[i_fir] * x_buffer[i_fir];  // Q16.16 * Q4.12 = Q20.28
        end
        
        // 第2级：64个加法器
        for(i_fir=0; i_fir<64; i_fir=i_fir+1) begin
            add_stage1[i_fir] <= mult_pipe[i_fir*2] + mult_pipe[i_fir*2+1];
        end
        
        // 第3级：32个加法器
        for(i_fir=0; i_fir<32; i_fir=i_fir+1) begin
            add_stage2[i_fir] <= add_stage1[i_fir*2] + add_stage1[i_fir*2+1];
        end
        
        // 第4级：16个加法器
        for(i_fir=0; i_fir<16; i_fir=i_fir+1) begin
            add_stage3[i_fir] <= add_stage2[i_fir*2] + add_stage2[i_fir*2+1];
        end
        
        // 第5级：8个加法器
        for(i_fir=0; i_fir<8; i_fir=i_fir+1) begin
            add_stage4[i_fir] <= add_stage3[i_fir*2] + add_stage3[i_fir*2+1];
        end
        
        // 第6级：4个加法器
        for(i_fir=0; i_fir<4; i_fir=i_fir+1) begin
            add_stage5[i_fir] <= add_stage4[i_fir*2] + add_stage4[i_fir*2+1];
        end
        
        // 第7级：2个加法器
        for(i_fir=0; i_fir<2; i_fir=i_fir+1) begin
            add_stage6[i_fir] <= add_stage5[i_fir*2] + add_stage5[i_fir*2+1];
        end
        
        // 最终累加
        fir_acc <= add_stage6[0] + add_stage6[1];
    end
end

// ==================== 5. FIR输出饱和（sample_en_dly4对齐） ====================
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        fxlms_output <= 16'sd0;
    end
    else if(sample_en_dly4) begin
        // fir_acc是Q20.28，右移16位得Q4.12
        if(fir_acc > 48'sh0000_7FFF_0000)
            fxlms_output <= 16'sh7FFF;
        else if(fir_acc < -48'sh0000_8000_0000)
            fxlms_output <= -16'sh8000;
        else
            fxlms_output <= fir_acc[31:16];
    end
end

// ==================== 6. 权重更新（LMS，128个并行更新） ====================
// 更新律：w[n+1] = w[n] - mu * e[n] * xf[n]
// 定点对齐：mu(Q?) * e(Q4.12) * xf(Q4.12)
integer i_wgt;
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        for(i_wgt=0; i_wgt<FILTER_ORDER; i_wgt=i_wgt+1) w_coeff[i_wgt] <= 32'sd0;
    end
    else if(sample_en) begin
        for(i_wgt=0; i_wgt<FILTER_ORDER; i_wgt=i_wgt+1) begin
            // 分步计算避免位宽溢出，显式截断
            w_coeff[i_wgt] <= w_coeff[i_wgt] - $signed(
                ((mu_current * $signed({1'b0, error_input})) >>> 19) * $signed({1'b0, xf_buffer[i_wgt]}) >>> (SHIFT_BIT-19)
            );
        end
    end
end

*/


// ==================== 3. 时序复用状态机（核心改造：省去 250+ DSP） ====================
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        state <= IDLE;
        mac_cnt <= 7'd0;
        fir_acc <= 48'sd0;
        fxlms_output <= 16'sd0;
        for(i_wgt_init=0; i_wgt_init<FILTER_ORDER; i_wgt_init=i_wgt_init+1) begin
            w_coeff[i_wgt_init] <= 32'sd0;
        end
    end 
    else begin
        case(state)
            IDLE: begin
                mac_cnt <= 7'd0;
                fir_acc <= 48'sd0; 
                // 等待 10kHz 的 sample_en 脉冲触发
                if(sample_en) begin
                    state <= FIR_CALC; 
                end
            end
            
            FIR_CALC: begin
                // 使用 1 个硬件乘法器，执行 128 次时分累加 (Q16.16 * Q4.12 = Q20.28)
                fir_acc <= fir_acc + w_coeff[mac_cnt] * x_buffer[mac_cnt];
                
                if(mac_cnt == FILTER_ORDER - 1) begin
                    mac_cnt <= 7'd0;
                    state <= WGT_UPDATE; // FIR计算完毕，去更新权重
                end else begin
                    mac_cnt <= mac_cnt + 1'b1;
                end
            end
            
            WGT_UPDATE: begin
                // 使用 1 个硬件乘法器，时分更新 128 个权重系数
                w_coeff[mac_cnt] <= w_coeff[mac_cnt] - $signed(
                    ((mu_current * $signed({1'b0, error_input})) >>> 19) * $signed({1'b0, xf_buffer[mac_cnt]}) >>> (SHIFT_BIT-19)
                );
                
                if(mac_cnt == FILTER_ORDER - 1) begin
                    mac_cnt <= 7'd0;
                    state <= DONE; // 权重更新完毕，进入输出赋值
                end else begin
                    mac_cnt <= mac_cnt + 1'b1;
                end
            end
            
            DONE: begin
                // FIR 输出饱和截断处理 (fir_acc 是 Q20.28，截取 Q4.12)
                if(fir_acc > 48'sh0000_7FFF_0000)
                    fxlms_output <= 16'sh7FFF;
                else if(fir_acc < -48'sh0000_8000_0000)
                    fxlms_output <= -16'sh8000;
                else
                    fxlms_output <= fir_acc[31:16];
                    
                state <= IDLE; // 回归空闲，等待下一个 sample_en
            end
            
            default: state <= IDLE;
        endcase
    end
end




endmodule