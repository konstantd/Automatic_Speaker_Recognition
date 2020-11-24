function generate_dataset2(win,over)
 
tic

%% Generate TrainingSet
%Speakers Subfolders
sp1  = 'id10011';
sp2  = 'id10022';
sp3  = 'id10032';
sp4  = 'id10062';
sp5  = 'id10066';
sp6  = 'id10071';
sp7  = 'id10072';
sp8  = 'id10130';
sp9  = 'id10190';
sp10 = 'id10204';

% All files
allf = [sp1; sp2; sp3; sp4; sp5; sp6; sp7; sp8; sp9; sp10];

% Stores the path
oldpath = cd;    

% Loop for each speaker/file
for sp = 1 : size(allf,1) 
    
    files = dir(strcat(allf(sp,:),'/*.wav'));   %access the name of .wav files of the speaker        
    cd(allf(sp,:))              %access the subfolder of the speaker
    
    for i = 1:length(files)
        
        [audio{i}, fs] = audioread(files(i).name);
        a = miraudio(audio{i},fs);      
        %    win = 0.04;      %Window-size in sec
        %    over = 0.5;      %Overlapping
        AF_new = feature_extraction(a,win,over,sp);   
        if (i == 1)
            AF = AF_new;          
        else
            AF = [AF; AF_new];
        end      
    end
    
    if (sp == 1)
        AF_sum= AF;
    else
        AF_sum = [AF_sum; AF];
    end
    
    cd(oldpath)
    
end


str = sprintf('training_set.mat');
save(str, 'AF_sum')


%% Generate TestSet

%All files
allf = [sp1; sp2; sp3; sp4; sp5; sp6; sp7; sp8; sp9; sp10];

%Stores the path
oldpath = cd;    

%Loop for each speaker
for sp = 1 : size(allf,1)    
  
    cd(allf(sp,:))              %access the subfolder of the speaker
    files = dir(strcat('test','/*.wav'));   %access the .wav files of the speaker  
    cd('test')
    for i = 1:length(files)
       
        [audio{i}, fs] = audioread(files(i).name);
        a = miraudio(audio{i},fs);      
        AF_new_test = feature_extraction(a,win,over,sp); 
        
        if (i == 1)
            AF_test = AF_new_test;
        else
            AF_test = [AF_test; AF_new_test];
        end 
        
    end
    
    cd(oldpath)
    str = sprintf('Speaker %d',sp);
    save(str, 'AF_test')
    
end

toc

end

% Elapsed time is 123.101229 seconds.