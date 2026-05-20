% MATLAB 到 Verilog 预训练权重量化脚本模板
% ============================================
% 用途：将 MATLAB 训练好的 Elman RNN 浮点权重，转换为 Verilog initial 块格式
% 定点格式：隐藏层权重 Q4.12（乘 4096 取整），输出层权重 Q16.16（乘 65536 取整）
% 
% 步骤：
% 1. 在 MATLAB 中训练 Elman RNN（输入：振动信号 x，输出：迟滞补偿量 y）
% 2. 提取 W_xh（4x1）、W_hh（4x4）、b_h（4x1）、W_hy（1x4）、b_y（1x1）
% 3. 运行此脚本生成 Verilog initial 块
% 4. 复制输出到 elman_rnn.v 的 initial 块中替换占位值

function generate_elman_weights(W_xh_float, W_hh_float, b_h_float, W_hy_float, b_y_float)
    % 量化参数
    Q12 = 4096;    % Q4.12 量化因子
    Q16 = 65536;   % Q16.16 量化因子
    
    H = 4;  % 隐藏层神经元数
    
    fprintf('\n// ==================== 预训练权重（由MATLAB量化生成）====================\n');
    fprintf('initial begin\n');
    
    % W_xh: 4 x 1
    fprintf('    // W_xh: 输入到隐藏层，Q4.12\n');
    for i = 1:H
        val = W_xh_float(i);
        q = round(val * Q12);
        if q > 32767, q = 32767; end
        if q < -32768, q = -32768; end
        if q >= 0
            fprintf('    W_xh[%d] = 16\'sd%d;     // %.6f\n', i-1, q, val);
        else
            fprintf('    W_xh[%d] = 16\'sd%d;     // %.6f\n', i-1, q, val);
        end
    end
    
    % W_hh: 4 x 4
    fprintf('\n    // W_hh: 隐藏层自反馈，Q4.12\n');
    for i = 1:H
        for j = 1:H
            val = W_hh_float(i,j);
            q = round(val * Q12);
            if q > 32767, q = 32767; end
            if q < -32768, q = -32768; end
            if q >= 0
                fprintf('    W_hh[%d][%d] = 16\'sd%d;  // %.6f\n', i-1, j-1, q, val);
            else
                fprintf('    W_hh[%d][%d] = 16\'sd%d;  // %.6f\n', i-1, j-1, q, val);
            end
        end
    end
    
    % b_h: 4 x 1
    fprintf('\n    // b_h: 隐藏层偏置，Q4.12\n');
    for i = 1:H
        val = b_h_float(i);
        q = round(val * Q12);
        if q > 32767, q = 32767; end
        if q < -32768, q = -32768; end
        fprintf('    b_h[%d] = 16\'sd%d;     // %.6f\n', i-1, q, val);
    end
    
    % b_y: 1 x 1
    fprintf('\n    // b_y: 输出偏置，Q4.12\n');
    val = b_y_float;
    q = round(val * Q12);
    if q > 32767, q = 32767; end
    if q < -32768, q = -32768; end
    fprintf('    b_y = 16\'sd%d;     // %.6f\n', q, val);
    
    % W_hy: 1 x 4，Q16.16
    fprintf('\n    // W_hy: 隐藏层到输出层，Q16.16\n');
    for i = 1:H
        val = W_hy_float(i);
        q = round(val * Q16);
        % Q16.16 范围：32位有符号 [-2147483648, 2147483647]
        if q > 2147483647, q = 2147483647; end
        if q < -2147483648, q = -2147483648; end
        if q >= 0
            fprintf('    W_hy[%d] = 32\'sh%08X;  // %.6f\n', i-1, q, val);
        else
            fprintf('    W_hy[%d] = 32\'sh%08X;  // %.6f\n', i-1, q + 4294967296, val);
        end
    end
    
    fprintf('end\n');
end

% ============================================
% 使用示例：
% 
% 假设你在 MATLAB 中训练后得到以下浮点权重：
% W_xh_float = [0.15; -0.08; 0.12; 0.05];
% W_hh_float = [0.80, 0.05, -0.02, 0.01; ...];  % 4x4
% b_h_float  = [0.0; 0.0; 0.0; 0.0];
% W_hy_float = [0.30, -0.15, 0.20, 0.10];
% b_y_float  = 0.0;
%
% 在 MATLAB 命令行运行：
% generate_elman_weights(W_xh_float, W_hh_float, b_h_float, W_hy_float, b_y_float);
%
% 将打印出的 initial 块复制到 elman_rnn.v 中替换原有 initial 块。
% ============================================
