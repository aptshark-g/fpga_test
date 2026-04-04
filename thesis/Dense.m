classdef Dense < handle
    properties
        weights
        bias
        gradW
        gradb
        inputCache
    end
    methods
        function obj = Dense(inDim, outDim)
            obj.weights = randn(outDim, inDim) * sqrt(2 / inDim);
            obj.bias = zeros(outDim, 1);
        end
        function Y = forward(obj, X)
            obj.inputCache = X;
            Y = obj.weights * X + obj.bias;
        end
        function dX = backward(obj, dY, lr)
            obj.gradW = dY * obj.inputCache';
            obj.gradb = sum(dY, 2);
            dX = obj.weights' * dY;
            obj.weights = obj.weights - lr * obj.gradW;
            obj.bias = obj.bias - lr * obj.gradb;
        end
    end
end