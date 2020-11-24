%Store in every sub-folder before running

function AF = feature_extraction(a,win,over,sp)
 
frames = mirframe(a,win,over);  
frames_data = mirgetdata(frames);  

%Features--------------------------------------
%peaks_length_data = mirgetdata(mirpeaks(frames))';
rms_data = mirgetdata(mirrms(frames))';
tonalcentroid_data = mirgetdata(mirtonalcentroid(frames))';  %6 features
zerocross_data = mirgetdata(mirzerocross(frames))';
roughness_data = mirgetdata(mirroughness(frames))';

rolloff_data1 = mirgetdata(mirrolloff(frames,'Threshold',0.2))';
rolloff_data2 = mirgetdata(mirrolloff(frames,'Threshold',0.25))';
rolloff_data3 = mirgetdata(mirrolloff(frames,'Threshold',0.3))';
rolloff_data4 = mirgetdata(mirrolloff(frames,'Threshold',0.35))';
rolloff_data5 = mirgetdata(mirrolloff(frames,'Threshold',0.75))';
rolloff_data6 = mirgetdata(mirrolloff(frames,'Threshold',0.8))';
rolloff_data7 = mirgetdata(mirrolloff(frames,'Threshold',0.85))';
rolloff_data8 = mirgetdata(mirrolloff(frames,'Threshold',0.9))';
rolloff_data9 = mirgetdata(mirrolloff(frames,'Threshold',0.95))';

brightness_data1 = mirgetdata(mirbrightness(frames,'CutOff',500))';
brightness_data2 = mirgetdata(mirbrightness(frames,'CutOff',1000))';
brightness_data3 = mirgetdata(mirbrightness(frames,'CutOff',1500))';
brightness_data4 = mirgetdata(mirbrightness(frames,'CutOff',2000))';
brightness_data5 = mirgetdata(mirbrightness(frames,'CutOff',2500))';
brightness_data6 = mirgetdata(mirbrightness(frames,'CutOff',3000))';
brightness_data7 = mirgetdata(mirbrightness(frames,'CutOff',3500))';
brightness_data8 = mirgetdata(mirbrightness(frames,'CutOff',4000))';
brightness_data9 = mirgetdata(mirbrightness(frames,'CutOff',4500))';
brightness_data10 = mirgetdata(mirbrightness(frames,'CutOff',5000))'; 
brightness_data11 = mirgetdata(mirbrightness(frames,'CutOff',5500))';     
brightness_data12 = mirgetdata(mirbrightness(frames,'CutOff',6000))';     
brightness_data13 = mirgetdata(mirbrightness(frames,'CutOff',6500))';     
brightness_data14 = mirgetdata(mirbrightness(frames,'CutOff',7000))';     
brightness_data15 = mirgetdata(mirbrightness(frames,'CutOff',7500))';         
%33 os edo
regularity_data = mirgetdata(mirregularity(frames))';

centroid_data = mirgetdata(mircentroid(frames))';
spread_data = mirgetdata(mirspread(frames))';
skewness_data = mirgetdata(mirskewness(frames))';
kurtosis_data = mirgetdata(mirkurtosis(frames))';
flatness_data = mirgetdata(mirflatness(frames))';
entropy_data = mirgetdata(mirentropy(frames))';

%39 mfccs
mfcc_data1 = mirgetdata(mirmfcc(frames))';
%Before the extend
mfcc_data2 = mirgetdata(mirmfcc(frames,'Delta',1));
mfcc_data3 = mirgetdata(mirmfcc(frames,'Delta',2));
%After the extend
mfcc_data2ext = wextend('ac','symw',mfcc_data2,2)';
mfcc_data3ext = wextend('ac','symw',mfcc_data3,4)';

%79 os edo

%12 features
chromagram_data1 = mirgetdata(mirchromagram(frames,'db', 17))'; 
chromagram_data2 = mirgetdata(mirchromagram(frames,'db', 20))'; 
chromagram_data3 = mirgetdata(mirchromagram(frames,'db', 23))'; 
chromagram_data4 = mirgetdata(mirchromagram(frames,'max', 8000))';  

mode_data = mirgetdata(mirmode(frames))';
%inharmonicity_data = mirgetdata(mirinharmonicity(frames))';

%24 features
keystrength_data =  mirgetdata(mirkeystrength(frames));  
keystrength_data1 =  keystrength_data(:,:,1,1)';
keystrength_data2 =  keystrength_data(:,:,1,2)';

%Statistics for keystrength maj and min
keystrength_data11 = mean(keystrength_data1')';
keystrength_data12 = std(keystrength_data1')';
keystrength_data13 = var(keystrength_data1')';
keystrength_data14 = skewness(keystrength_data1')';
keystrength_data15 = kurtosis(keystrength_data1')';

keystrength_data21 = mean(keystrength_data2')';
keystrength_data22 = std(keystrength_data2')';
keystrength_data23 = var(keystrength_data2')';
keystrength_data24 = skewness(keystrength_data2')';
keystrength_data25 = kurtosis(keystrength_data2')';

hcdf = mirgetdata(mirhcdf(frames));
hcdf_data = [mean(hcdf) hcdf]';
%End of Features----------------------------------------
last_column = sp*ones(size(frames_data,2),1);
AF = [rms_data    tonalcentroid_data    zerocross_data  roughness_data    rolloff_data1       rolloff_data2  rolloff_data3       rolloff_data4       rolloff_data5    rolloff_data6   rolloff_data7   rolloff_data8   rolloff_data9       brightness_data1    brightness_data2    brightness_data3   brightness_data4    brightness_data5    brightness_data6      brightness_data7     brightness_data8     brightness_data9    brightness_data10   brightness_data11   brightness_data12   brightness_data13   brightness_data14   brightness_data15    regularity_data     centroid_data  spread_data         skewness_data       kurtosis_data         flatness_data       entropy_data        mfcc_data1     mfcc_data2ext       mfcc_data3ext       chromagram_data1      chromagram_data2    chromagram_data3  chromagram_data4      mode_data  keystrength_data1  keystrength_data11  keystrength_data12    keystrength_data13  keystrength_data14  keystrength_data15   keystrength_data2   keystrength_data21  keystrength_data22  keystrength_data23  keystrength_data24     keystrength_data25  hcdf_data];
AF = [AF last_column]; 
end
