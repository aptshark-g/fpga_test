function [linear_part, nonlinear_part] = separate_linear_nonlinear_by_freq(signal, fs, f0, delta_f)
% 基于窄带带通滤波的线性/非线性分离（稳健版）
    if nargin < 4, delta_f = 15; end
    if nargin < 3 || isempty(f0) || f0 == 0
        % 自动检测基频
        nfft = 4096;
        window = hamming(nfft);
        [S, f] = pwelch(signal, window, nfft/2, nfft, fs);
        S = sqrt(S);
        [~, idx] = max(S(2:end));
        f0 = f(idx+1);
        fprintf('自动检测基频: %.2f Hz\n', f0);
    end
    
    % 边界保护
    low_freq = max(f0 - delta_f, 1);          % 最低不低于1Hz
    high_freq = min(f0 + delta_f, fs/2 - 1);  % 最高不超过奈奎斯特频率-1
    if low_freq >= high_freq
        low_freq = f0 - 5;
        high_freq = f0 + 5;
    end
    
    Wn = [low_freq, high_freq] / (fs/2);
    Wn(Wn <= 0) = 0.001;
    Wn(Wn >= 1) = 0.999;
    
    % 降低阶数至4，提高数值稳定性
    [b, a] = butter(4, Wn, 'bandpass');
    
    % 检查稳定性
    if ~isstable(b,a)
        warning('滤波器不稳定，改用低阶滤波器');
        [b, a] = butter(2, Wn, 'bandpass');
    end
    
    % 使用 filtfilt 进行零相位滤波，如果失败则回退到 filter
    try
        linear_part = filtfilt(b, a, signal);
    catch
        warning('filtfilt 失败，改用 filter');
        linear_part = filter(b, a, signal);
    end
    
    % 检查 NaN
    if any(isnan(linear_part))
        warning('滤波产生了NaN，将线性部分置为全零，非线性部分为原始信号');
        linear_part = zeros(size(signal));
        nonlinear_part = signal;
    else
        nonlinear_part = signal - linear_part;
    end
end