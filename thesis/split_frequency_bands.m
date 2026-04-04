function [band_signals, filters, final_states] = split_frequency_bands(signal, fs, bands, order)
% split_frequency_bands - 将信号按指定频段拆分为多个子带信号
% 输入:
%   signal  : 输入信号向量 (N x 1)
%   fs      : 采样率 (Hz)
%   bands   : Kx2 矩阵，每行 [低截止频率, 高截止频率] (Hz)
%   order   : 滤波器阶数 (推荐 4~8)
% 输出:
%   band_signals : N x K 矩阵，每列为对应频段的滤波后信号
%   filters      : 结构体数组，包含每个频段滤波器的 b,a 系数
%   final_states : 每个滤波器最终的状态 (用于连续滤波)

    N = length(signal);
    K = size(bands, 1);
    band_signals = zeros(N, K);
    filters = cell(K, 1);
    final_states = cell(K, 1);
    
    % 设计每个频段的带通滤波器
    for k = 1:K
        Wn = bands(k, :) / (fs/2);
        % 确保截止频率在 (0,1) 范围内
        Wn(Wn <= 0) = 0.001;
        Wn(Wn >= 1) = 0.999;
        [b, a] = butter(order, Wn, 'bandpass');
        filters{k}.b = b;
        filters{k}.a = a;
        % 滤波 (零相位滤波用于离线分析，避免相位失真)
        % 注意：filtfilt 引入延迟但无相位失真，适合离线验证
        % 若需实时仿真，请改用 filter 并保持状态
        band_signals(:, k) = filtfilt(b, a, signal);
        % 如果需要实时滤波的状态，取消下面注释
        % [~, final_states{k}] = filter(b, a, signal);
    end
end