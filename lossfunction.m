clc;
clear all;
close all;

% Read CSV file
% dataNew = csvread('C:\Users\ziqi_\Downloads\run-cartridgecase20240117T0028-tag-loss.csv', 1, 0);
data = csvread('C:\Users\ziqi_\Downloads\run-cartridgecase20240113T1340-tag-loss.csv', 1, 0);

% Extract column data
% x_new = dataNew(:, 2); % x data
% y_new = dataNew(:, 3); % y data

x_old = data(:, 2); % x data
y_old = data(:, 3); % y data

% Plot the curve
figure;
% plot(x_new, y_new, '-', 'LineWidth', 1); % First line Internet Images, bold
hold on;
plot(x_old, y_old, '-', 'LineWidth', 1); % Second line NBTRD Images, bold

xlabel('Epoch');
ylabel('Total\_Loss');
title('Loss Curves for Internet Images and NBTRD Images');

% legend('Internet Images', 'NBTRD Images');
legend('NBTRD Images');
grid on;

% Add labels at the end of each curve
% text(x_new(end), y_new(end), num2str(y_new(end)), 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
text(x_old(end), y_old(end), num2str(y_old(end)), 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');

hold off;
