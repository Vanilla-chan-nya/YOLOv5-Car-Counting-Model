data=table2array(readtable("yolo_track_3color\cross.txt"));
data=data(:,2);
data=[0;diff(data)];
P=[];
T=10;

for i=1:T*25:7644
    P=[P,sum(data(i:min(7644,i+T*25-1)))];
end

% 绘制条形图
figure;
hold on;  % 保持当前图形，允许多次绘图

% 定义色彩映射
SIGEWINNE = [81 132 178;
             170 212 248;
             242 245 250;
             241 167 181;
             213 82 118] / 256;
num_colors = 100;
interp_colors = interp1(linspace(0, 1, size(SIGEWINNE, 1)), SIGEWINNE, linspace(0, 1, num_colors));
colormap(interp_colors);

% 使用patch绘制每个条，并设置颜色
for idx = 1:length(P)
    x = [idx-0.4, idx+0.4, idx+0.4, idx-0.4];
    y = [0, 0, P(idx), P(idx)];
    disp(ceil(P(idx)/max(P)*num_colors));
    patch(x, y, interp_colors(max(1,ceil(P(idx)/max(P)*num_colors)), :), 'EdgeColor', 'none');
end

plot(P, 'Color', [0, 0.8, 1], 'LineStyle', '-.', 'Marker', '.', 'LineWidth', 1.5, 'MarkerSize', 10);


% 设置颜色条和坐标轴标签
colorbar;
caxis([min(P), max(P)]);  % 设置色彩轴的范围为数据的最小值和最大值
xlabel('时间');
ylabel('车辆数');
xticks(0:10:length(P));
xticklabels(arrayfun(@num2str, (0:10:length(P))*10, 'UniformOutput', false));  % 设置自定义的刻度标签
hold off;  % 释放图形保持状态

