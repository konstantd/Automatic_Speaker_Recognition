function [data_r,fs,id] = vad(audio,filename)
[data_r1,fs1,id1,frames]=desilence(audio);
[data_r2,fs2,id2]=desilence1(audio);
id=union(id1,id2);

fs=fs1;
fr_ws= frames(id,:);
data_r = reshape(fr_ws',1,[]);
music = filename + ".wav";
music=convertStringsToChars(music);
audiowrite(music,data_r,fs);

end

