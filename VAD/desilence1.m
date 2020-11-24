function [data_r,fs,id]  = desilence1(audio)

% read sound 
[data, fs] = audioread(audio);
% normalize data
data = data / abs(max(data));

% do framing
f_d = 0.04;
frames = framing(data, fs, f_d);

% Short term autocorrelation and taking 1 coeff. value i.e. at lag 1 
[r,c] = size(frames);
for k = 1:r
t = frames(k,:);
sum1 = 0;ac = 0;
for i = 0:length(t)-1
    sum1 = 0;
    for j = 1:length(t)-i
        s = t(j)*t(j+i);
        sum1 = sum1 + s;
    end
    ac(i+1) = sum1;
end
temp(k) = ac(1);
stac(k) = ac(2); % taking second coeff. built in function gives normalized
% autocorrelation (it is autocorr at lag 1)
end

% short term auto corr. (stac)
stac = stac./max(stac); %normalize the data

f_size = round(f_d * fs);
stac_wave = 0;
for j = 1 : length(stac)
    l = length(stac_wave);
    stac_wave(l : l + f_size) = stac(j);
end

% plot the stac with Signal
t = [0 : 1/fs : length(data)/fs]; % time in sec
t = t(1:end - 1);
t1 = [0 : 1/fs : length(stac_wave)/fs];
t1 = t1(1:end - 1);

plot(t,data'); hold on;
plot(t1,stac_wave,'r','LineWidth',2);
legend('Speech Signal','Short Term Auto Corr.');

% Silence Removal
id = find(stac >= 0.015);
fr_ws = frames(id,:); % frames without silence

% reconstruct signal
data_r = reshape(fr_ws',1,[]);
figure;
plot(data);hold on;
plot(data_r,'g'); title('speech without silence');
end

