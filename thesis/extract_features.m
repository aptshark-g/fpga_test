function features = extract_features(vibration, error, control, xf_buffer, k, fs, f_ref)
% 改进的特征提取函数，共19维，完全基于当前时刻及历史信息（无未来）
% 输入：
%   vibration  - 原始振动信号向量（全长，索引k为当前时刻）
%   error      - 误差信号向量（全长）
%   control    - 控制信号向量（全长）
%   xf_buffer  - 滤波后参考信号缓冲区（当前时刻的值为 xf_buffer(1)）
%   k          - 当前索引（>=1）
%   fs         - 采样率 (Hz)
%   f_ref      - 参考频率 (Hz)，已知为100Hz
% 输出：
%   features   - 1×19 特征向量

    % 初始化特征为零
    features = zeros(1, 19);
    
    % 确保历史长度足够
    if k < 200
        return;  % 前200步不提取有效特征，返回零
    end
    
    %% 1. 参考信号统计特征（8维）
    win_len = min(200, k);
    x_win = vibration(k-win_len+1:k);      % 最近200个点的振动
    features(1) = mean(x_win);              % 均值
    features(2) = var(x_win);               % 方差
    features(3) = max(x_win);               % 最大值
    features(4) = min(x_win);               % 最小值
    % 局部波动（相邻差值的绝对值平均）
    diff_x = abs(diff(x_win));
    features(5) = mean(diff_x);             % 平均变化率
    features(6) = max(diff_x);              % 最大变化率
    % 过零率（粗略估计频率）
    zero_cross = sum(x_win(1:end-1) .* x_win(2:end) < 0) / length(x_win);
    features(7) = zero_cross;               % 过零率
    % 峭度（反映冲击）
    features(8) = kurtosis(x_win);
    
    %% 2. 参考信号的正交分量（2维）——已知频率100Hz
    t = k / fs;
    features(9) = sin(2*pi*f_ref*t);
    features(10) = cos(2*pi*f_ref*t);
    
    %% 3. 误差信号统计特征（4维）
    win_len_e = min(100, k);
    e_win = error(k-win_len_e+1:k);
    features(11) = mean(e_win);             % 误差均值
    features(12) = var(e_win);              % 误差方差
    features(13) = e_win(end);              % 当前误差
    % 误差的斜率（最近5点线性拟合）
    if length(e_win) >= 5
        t_e = (1:5)' / fs;
        p = polyfit(t_e, e_win(end-4:end), 1);
        features(14) = p(1);                % 误差变化斜率
    else
        features(14) = 0;
    end
    
    %% 4. 控制信号特征（3维）
    win_len_c = min(50, k);
    c_win = control(k-win_len_c+1:k);
    features(15) = c_win(end);              % 当前控制量
    features(16) = mean(c_win);             % 近期控制均值
    if length(c_win) >= 2
        features(17) = c_win(end) - c_win(end-1);  % 控制增量
    end
    
    %% 5. 次级路径滤波参考信号特征（1维）
    % xf_buffer(1) 是当前时刻的滤波参考信号
    if ~isempty(xf_buffer)
        features(18) = xf_buffer(1);
    end
    
    %% 6. 频域特征（1维）——简化：最近256点FFT的主频幅值
    if k >= 256
        seg = vibration(k-255:k) .* hamming(256);
        Y = abs(fft(seg));
        Y = Y(1:129);
        [~, idx] = max(Y);
        freq_res = fs / 256;
        main_freq = (idx-1) * freq_res;
        % 只保留主频幅值（归一化）
        features(19) = Y(idx) / max(Y);
    else
        features(19) = 0;
    end
    
    % 处理可能的 NaN 或 Inf
    features(isnan(features)) = 0;
    features(isinf(features)) = 0;
end