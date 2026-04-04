function y = generate_vibration(source, t, condition)
% 生成振动信号，支持三种模式：fixed / chirp / modulated
% 输入：
%   source    - VibrationSource 对象（提供幅值、频率等默认值）
%   t         - 当前时刻（标量）
%   condition - 结构体，包含字段：
%       .frequency_mode : 'fixed', 'chirp', 'modulated'
%       .amp            : 幅值（可选，若缺失则使用 source.amplitudes(1)）
%       以及各模式所需的其他参数（见代码）
    
    % 默认模式为固定频率
    if ~isfield(condition, 'frequency_mode')
        condition.frequency_mode = 'fixed';
    end
    
    % 确定幅值：优先使用 condition.amp，否则用 source.amplitudes(1)
    if isfield(condition, 'amp')
        A = condition.amp;
    else
        A = source.amplitudes(1);
    end
    
    switch condition.frequency_mode
        case 'fixed'
            % 原固定频率模式（多频叠加）
            y = 0;
            for i = 1:length(source.frequencies)
                f = source.frequencies(i);
                Ai = source.amplitudes(i);
                phi = source.phases(i);
                y = y + Ai * sin(2*pi*f*t + phi);
            end
            
        case 'chirp'
            % 线性扫频：频率从 freq_start 到 freq_end
            if ~isfield(condition, 'freq_start'), condition.freq_start = 50; end
            if ~isfield(condition, 'freq_end'), condition.freq_end = 100; end
            if ~isfield(condition, 'chirp_duration'), condition.chirp_duration = 10; end
            % 瞬时频率线性变化
            f_inst = condition.freq_start + (condition.freq_end - condition.freq_start) * t / condition.chirp_duration;
            % 相位 = 2π∫f dt
            phase = 2*pi * (condition.freq_start * t + (condition.freq_end - condition.freq_start) * t^2 / (2*condition.chirp_duration));
            y = A * sin(phase);
            
        case 'modulated'
            % 正弦调制频率：中心频率 freq_center，偏移 freq_dev，调制频率 mod_freq
            if ~isfield(condition, 'freq_center'), condition.freq_center = 75; end
            if ~isfield(condition, 'freq_dev'), condition.freq_dev = 25; end
            if ~isfield(condition, 'mod_freq'), condition.mod_freq = 0.5; end
            % 瞬时频率
            f_inst = condition.freq_center + condition.freq_dev * sin(2*pi*condition.mod_freq*t);
            % 数值积分得到相位（因为 t 是标量，可直接用公式，但需注意起始相位）
            % 对于正弦调制，相位有解析解：∫ sin(ω_m τ) dτ = -cos(ω_m t)/ω_m
            omega_c = 2*pi*condition.freq_center;
            omega_m = 2*pi*condition.mod_freq;
            beta = condition.freq_dev / condition.mod_freq;   % 调制指数
            phase = omega_c * t + beta * sin(omega_m * t);
            y = A * sin(phase);
    end
    
    % 添加噪声（可选）
    if isfield(condition, 'enable_noise') && condition.enable_noise
        noise_level = source.noise_level;
        y = y + noise_level * randn(size(t));
    end
end