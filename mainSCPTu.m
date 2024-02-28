clear; close all; format compact; clc
%%

CPTu_data_dir = 'C:\Users\qdmofa\OneDrive - TUNI.fi\Fincone II-adjunct\Asli\FINCONE II - Data\Test sites\Database';
out_dir = 'C:\Users\qdmofa\OneDrive - TUNI.fi\Fincone II-adjunct\Asli\FINCONE II - Analytical tools\Codes\SCPTu\Code\out';

load(fullfile(CPTu_data_dir,'CPTu.mat'));
data = CPTu.Ulvila.CV020.SCPTu;
%% 
depths = [2; 3; 4; 5; 6; 7; 8; 9; 10];

fig_name = "SCPTu results";
f = figure ('Name',fig_name,'Position',[100 100 900 400]);
set(f,'defaulttextinterpreter','latex');
tiledlayout(1,2,"TileSpacing","tight","Padding","tight")
nexttile
for i = 2 : 10
    depths(i) = i;
    matrix_name = strcat('S', num2str(i)); % Create the matrix name
    matrix = data.(matrix_name); % Access the matrix from the struct
    t = matrix(:,1);
    L = 0.45 * (matrix(:,2)/max(abs(matrix(:,2)))) + depths(i); % normalizing to +-0.45
    R = 0.45 * (matrix(:,3)/max(abs(matrix(:,3)))) + depths(i); % normalizing to +-0.45
    P = 0.45 * (matrix(:,4)/max(abs(matrix(:,4)))) + depths(i); % normalizing to +-0.45
    
    plot(t,L,'r');
    hold on
    plot(t,R,'b');
end
hold off
xLab = "$Time$ [sec]";
yLab = "Depth [m]";
xlabel(xLab,'FontSize',10,'Color','k','Interpreter','latex')
ylabel(yLab,'FontSize',10,'Color','k','Interpreter','latex')
xlim([0 400])
legend('Left', 'Right', 'FontSize',8, 'Location','northeast', 'Interpreter','latex', "color","white");
grid on
grid minor
set (gca, 'YDir','reverse')
ax = gca;
set(ax,'TickLabelInterpreter','latex')

for i = 3 : 10
    mat1 = strcat('S', num2str(i-1)); % Create the matrix name
    mat1 = data.(mat1); % Access the matrix from the struct
    s1L = 0.45 * (mat1(:,2)/max(abs(mat1(:,2)))); % normalizing to +-0.45
    s1R = 0.45 * (mat1(:,3)/max(abs(mat1(:,3)))); % normalizing to +-0.45
    s1P = 0.45 * (mat1(:,4)/max(abs(mat1(:,4)))); % normalizing to +-0.45

    mat2 = strcat('S', num2str(i)); % Create the matrix name
    mat2 = data.(mat2); % Access the matrix from the struct
    s2L = 0.45 * (mat2(:,2)/max(abs(mat2(:,2)))); % normalizing to +-0.45
    s2R = 0.45 * (mat2(:,3)/max(abs(mat2(:,3)))); % normalizing to +-0.45
    s2P = 0.45 * (mat2(:,4)/max(abs(mat2(:,4)))); % normalizing to +-0.45

    f = 1 / ((t(i) - t(i-1)) / 1000); % sampling rate (frequency)

    % CWT
    %     cwt1 = cwt(s1L, 'amor');
    %     cwt2 = cwt(s2L, 'amor');
    %     % Convert the CWT output to a vector
    %     cwt1_vector = sum(cwt1, 1);
    %     cwt2_vector = sum(cwt2, 1);
    %     % Calculate the cross-correlation of the CWTs
    %     [correlation, lag] = xcorr(cwt1_vector, cwt2_vector);
    %     % Find the index of the maximum correlation
    %     [~,I] = max(abs(correlation));
    %     % Time difference corresponds to the lag at the maximum correlation
    %     time_diff_L_cwt(i) = lag(I) * (t(2) - t(1))  % Assuming uniform time steps
    %     delay_seconds_L_cwt(i) = time_diff_L_cwt(i) / 1000;
    %     vsL_cwt(i,1) = (depths(i) - depths(i-1)) / delay_seconds_L_cwt(i);

    %     % MLE
    %     likelihood = zeros(size(t));
    %     for i = 1:length(t)
    % %         % Shift signal1 by the current delay
    % %         shifted_signal1 = shift(signal1, t(i));
    %         % Calculate the likelihood of signal2 given the shifted signal1
    %         likelihood(i) = calculate_likelihood(s1L, s2L);
    %     end
    %     % Find the delay that maximizes the likelihood
    %     [~,I] = max(likelihood);
    %     time_diff = t(I);

    % Compute the matched filter output: Right
    matched_filter_output = xcorr(s1R, s2R);
    % Find the delay
    [~, idx] = max(matched_filter_output);
    delay_samples = length(s1R) - idx;
    f = 1 / ((t(i) - t(i-1)) / 1000);
    delay_seconds_R = delay_samples / f;
    vsR(i,1) = (depths(i) - depths(i-1)) / delay_seconds_R;

    fprintf('The delay between the two signals is %.2f seconds.\n', delay_seconds_R);
    fprintf('The shear wave velocity between the two signals is %.2f m/s.\n', vsR(i,1));

    % Compute the matched filter output: Left
    matched_filter_output = xcorr(s1L, s2L);
    % Find the delay
    [~, idx] = max(matched_filter_output);
    delay_samples = length(s1L) - idx;
    
    delay_seconds_L = delay_samples / f;
    vsL(i,1) = (depths(i) - depths(i-1)) / delay_seconds_L;

    fprintf('The delay between the two signals is %.2f seconds.\n', delay_seconds_L);
    fprintf('The shear wave velocity between the two signals is %.2f m/s.\n', vsL(i,1));
end

% Visually selected points:
pL = [78; 88.4; 91.8; 101.4; 107.6; 118.4; 123.6; 132; 139]; % Left peak
pR = [82; 88.2; 93.2; 100.6; 106.8; 118.4; 125; 131.6; 137]; % Right peak

for i = 1 : length(pL) - 1
    % Left
    delay_vis_L(i+2,1) = (pL(i+1,1) - pL(i,1)) / 1000; % ms to sec
    vs_vis_L(i+2,1) = (depths(i+2,1) - depths(i+1,1)) / delay_vis_L(i+2,1);
    % Right
    delay_vis_R(i+2,1) = (pR(i+1,1) - pR(i,1)) / 1000; % ms to sec
    vs_vis_R(i+2,1) = (depths(i+2,1) - depths(i+1,1)) / delay_vis_R(i+2,1);
end

% Averaging the vs values, excluding the outliers
for i = 1 : length(vsL)
    a = [vsL(i), vsR(i), vs_vis_L(i), vs_vis_R(i)];

    % Identify the outliers
    Q1 = quantile(a,0.25); % only for plot
    Q2 = quantile(a,0.50); % I have checked 0.25. 0.5 worked better!
    Q3 = quantile(a,0.75);
    IQR = Q3 - Q2;

    outlierIndices = find(a < (Q2 - 1.5*IQR) | a > (Q3 + 1.5*IQR));
    
    % Remove the outliers
    a(outlierIndices) = [];

    % Compute the average
    vs_ave(i,1) = mean(a);
    stdvs(i,1) = std(a, 0, 2);

    QQ1(i) = Q1;
    QQ2(i) = Q2;
    QQ3(i) = Q3;

end


depth = [2.5; 3.5; 4.5; 5.5; 6.5; 7.5; 8.5; 9.5];

nexttile
errorbar(vs_ave(3:end), depth, stdvs(3:end), "horizontal",'LineWidth',1, 'Color',[0 0 0]);
hold on;
% Add the quantiles to the plot
plot(QQ1(3:end), depth, ':', 'color', [0 0 1]);
plot(QQ2(3:end), depth, '-.', 'color', [0 0 1]);
plot(QQ3(3:end), depth, '--', 'color', [0 0 1]);
xlabel('$V_s$ [m/s]','FontSize',10,'Interpreter','latex');
ylabel('','FontSize',10,'Interpreter','latex');
ylim([1 11])
legend('Mean with std. dev.', '$1^{st}$ quartile', '$2^{nd}$ quartile','$3^{rd}$ quartile','FontSize',8,'Location','northeast','Interpreter','latex',"color","white");
set (gca, 'YDir','reverse')
ax = gca;
set(ax,'TickLabelInterpreter','latex')
grid on
% Turn off the x minor grid
ax.XMinorGrid = 'on';
% Turn on the y minor grid
ax.YMinorGrid = 'off';
% annotations:
a = vs_ave;
a(a==0)=[];
for i = 1 : length(depth)
    text(10,depth(i)-0.0,sprintf('$V_s$ = %-5.2f m/s',a(i)),'FontSize',10,'Interpreter','latex')
end

% Save figure
res = 300;
FncSaveFormats(out_dir,fig_name,res)


Vs = table(depths,vsL,vsR,vs_vis_L,vs_vis_R, vs_ave, VariableNames={'Succeeding depth','Left_CC','Right_CC','Left_vis','Right_vis','mean_vs'})

%%
% No, cross-correlation is not the only method to find the delay between two signals. There are several other methods, including the Wavelet Transform method you mentioned. Here’s a brief overview of some of these methods:
% Fourier Transform: The Fourier Transform can be used to convert the signals from the time domain to the frequency domain, where the phase difference can be used to calculate the time delay.
% Wavelet Transform: The Wavelet Transform provides a time-frequency representation of the signal, which can be used to detect and localize the time delay between two signals. This method is particularly useful when dealing with non-stationary signals.
% Maximum Likelihood Estimation: This method involves estimating the delay that maximizes the likelihood of observing the received signal given the transmitted signal.
% Matched Filter: The output of a matched filter (the convolution of a signal with a time-reversed version of itself) gives a peak at the point of maximum signal-to-noise ratio, which can be used to estimate the delay.
% Each of these methods has its own advantages and disadvantages, and the best method to use depends on the specific characteristics of your signals and the nature of the delay you are trying to measure. For example, if your signals are non-stationary, the Wavelet Transform method might be more appropriate.
% 
% Here’s a simple example of how you might use the Continuous Wavelet Transform (CWT) to estimate the delay between two signals in MATLAB:
% Assuming signal1 and signal2 are your two signals and t is the time vector
cwt1 = cwt(signal1, 'amor');
cwt2 = cwt(signal2, 'amor');

% Calculate the cross-correlation of the CWTs
[correlation, lag] = xcorr(cwt1, cwt2);

% Find the index of the maximum correlation
[~,I] = max(abs(correlation));

% Time difference corresponds to the lag at the maximum correlation
time_diff = lag(I) * (t(2) - t(1));  % Assuming uniform time steps
% Please replace signal1, signal2, and t with your actual signal and time variable names.
%% Maximum Likelihood Estimation
% Sure, I can provide a conceptual example of how Maximum Likelihood Estimation (MLE) might be used to estimate the delay between two signals. However, please note that implementing MLE for delay estimation in practice can be quite complex and may require a deep understanding of statistical signal processing.
% 
% Conceptually, the idea behind MLE is to find the delay that maximizes the likelihood of observing the received signal given the transmitted signal. In the context of delay estimation, you could define a likelihood function that measures the similarity between the received signal and a version of the transmitted signal that has been shifted by a certain delay. The delay that maximizes this likelihood function is then chosen as the estimated delay.
% 
% Here’s a simple MATLAB-like pseudocode that illustrates this concept:
% Assuming signal1 and signal2 are your two signals and t is the time vector
likelihood = zeros(size(t));

for i = 1:length(t)
    % Shift signal1 by the current delay
    shifted_signal1 = shift(signal1, t(i));
    
    % Calculate the likelihood of signal2 given the shifted signal1
    likelihood(i) = calculate_likelihood(signal2, shifted_signal1);
end

% Find the delay that maximizes the likelihood
[~,I] = max(likelihood);
time_diff = t(I);
% In this pseudocode, shift is a function that shifts signal1 by a certain delay, and calculate_likelihood is a function that calculates the likelihood of signal2 given shifted_signal1. The exact implementation of these functions would depend on the specific characteristics of your signals and the nature of the delay you are trying to measure.

% Please note that this is a simplified example and actual implementation may require more complex signal processing techniques. Also, this method assumes that the delay is constant over the entire duration of the signals. If the delay varies over time, more advanced methods may be needed.
%% Matched Filter
% Define two signals
fs = 1000; % Sampling frequency
t = 0:1/fs:1-1/fs; % Time vector
f = 5; % Frequency of the signal
signal1 = sin(2*pi*f*t); % Signal 1

delay = 0.1; % Delay in seconds
signal2 = [zeros(1, delay*fs), signal1(1:end-delay*fs)]; % Signal 2

% Compute the matched filter output
matched_filter_output = xcorr(signal2, signal1);

% Find the delay
[~, idx] = max(matched_filter_output);
delay_samples = length(signal1) - idx;
delay_seconds = delay_samples / fs;

fprintf('The delay between the two signals is %.2f seconds.\n', delay_seconds);

% In this example, signal1 and signal2 are two signals where signal2 is a delayed version of signal1. The xcorr function computes the cross-correlation of the two signals, which is used as the output of the matched filter. The delay is then computed as the index of the maximum value of the matched filter output, adjusted by the length of the signal and converted to seconds. Please note that this is a simple example and might need to be adjusted based on your specific application. Also, this code assumes that signal2 is a delayed version of signal1, if it’s the other way around, the delay might be computed as a negative value. You might want to take the absolute value of the delay in such a case.




