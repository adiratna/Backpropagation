% Contoh 2. Masalah Exor/EX-OR (Exclusive-OR) bipolar dengan toolbox matlab.

clc;

% Pola masukan
s = [1 -1 1 -1;
    -1 1 1 -1];

% Keluaran target
t = [1 1 -1 -1];

% Pembentukan jaringan backpropagation
jar = newff([-1 1; -1 1], [2 1], {'tansig' 'tansig'}, 'traingd', 'learngdm', 'sse'); 

% Cacah epoch maksimum = 1000
% Nanti coba-coba sendiri bila diberi epoch maksimum = 10000
jar.trainParam.epochs = 1000;

jar.trainParam.goal = 0.01; % Error (SSE) maksimum = 0.01
jar.trainParam.lr = 0.05;   % Learning rate = 0.05
jar.trainParam.min_grad = 0.0;  % Gradien error minimum = 0
jar = train(jar,s,t) % Pelatihan jaringan
y = sim(jar,s)  % Uji jaringan dengan pola pelatihan
    

