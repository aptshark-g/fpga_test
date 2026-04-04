function [filters, states] = design_filterbank(bands, fs, order)
    % bands: Kx2矩阵，每行[low_cut, high_cut] (Hz)
    % order: 滤波器阶数（推荐4~6）
    % 输出: filters 结构体数组，每个包含 b,a；states 为初始状态
    num_bands = size(bands, 1);
    filters = cell(num_bands, 1);
    states = cell(num_bands, 1);
    for k = 1:num_bands
        Wn = bands(k,:) / (fs/2);
        [b, a] = butter(order, Wn, 'bandpass');
        filters{k}.b = b;
        filters{k}.a = a;
        states{k} = zeros(max(length(b), length(a))-1, 1);
    end
end