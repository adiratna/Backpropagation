% Contoh 1. Jaringan Backpropagation Untuk Masalah XOR Bipolar
% Deskripsi soal :
% Implementasikan jaringan backpropagation dengan satu hidden layer.
% Gunakan dua unit input, dua hidden unit, dan satu unit output. Gunakan
% bias pada setia hidden unit dan setiap unit output. Gunakan fungsi
% aktivasi sigmoid bipolar. Data pelatihan (training) diberikan oleh tabel
% berikut :

% s(1) = [1 -1]     t(1) = 1
% s(1) = [-1 1]     t(2) = 1
% s(1) = [1 1]      t(3) = -1
% s(1) = [-1 1]     t(4) = -1

% Gunakan bobot awal terdistribusi acak antara -0,5 dan 0,5. Sedangkan
% learning rate yang digunakan adalah 0,05 dan 0,5. Untuk setiap learning
% rate rate dilakukan 1000 dan 10000 epoch pelatihan (training) dengan
% bobot awal yang sama untuk setiap kasus. Untuk setiap kasus tunjukkan
% bobot akhir dan tanggapan terhadap pola masukan.

% Solusi
% Jika ditinjau untuk kasus pertama yaitu learning rate 0,05 dengan cacah
% epoch maksimum sebesar 1000.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;

s = [1 1 -1;    % 1 s1 s2 -> Matriks Pola Masukan
    1 -1 1;     % 1 s1 s2
    1 1 1;      % 1 s1 s2
    1 -1 -1];   % 1 s1 s2

t = [1 1 -1 -1];    % t1 t2 t3 t4 -> Vektor Keluaran

v = [0.4 0.3;       % v01 v02 -> Matriks Bobot Awal
    -0.4 0.3;       % v11 v12 -> Ke Hidden Unit
    0.2 -0.5];      % v21 v22

w = [0.5 -0.3 0.2]; % w01 w11 w21 -> Bobot awal ke unit keluaran (output).
a = 0.5;            % Learning Rate
epochmax = 1000;    % Cacah laju maksimum.
vb = v; 
wb = w;             % Baca bobot awal
epoch = 0;
while epoch<epochmax
    for i = 1:4      % Untuk 4 pola pelatihan (training)
        x =s(i,:);   % Baca untuk pola masukan
        zin = x*vb;  % Hitung masukan neto ke hidden unit
        z1 = (2/(1+exp(-zin(1))))-1; % Menghitung akivasi
        z2 = (2/(1+exp(-zin(2))))-1; % Dua hidden unit
        z = [1 z1 z2]; % Masukan bagi hidden unit
        
        yin = z*wb';    % Masukan neto hidden unit
        y = (2/(1+exp(-yin)))-1; % Aktivasi unit keluaran
        dk = (t(i)-y)*0.5*(1+y)*(1-y); % Suku informasi error
        
        dw11 = a*dk*z1; % perubahan bobot yang dari hidden unit 1
        dw21 = a*dk*z2; % perubahan bobot yang dari hidden unit 2
        
        dw01 = a*dk;    % Perubahan bias
        din1 = dk*wb(2);
        din2 = dk*wb(3);
        
        dj1 = din1*0.5*(1+z1)*(1-z1); % Suku informasi error yang dirambatkan
        dj2 = din2*0.5*(1+z2)*(1-z2); % Ke hidden layer
        dv11 = a*dj1*x(2);
        dv12 = a*dj2*x(2);  % Perubahan bobot dari unit input 1
        dv21 = a*dj1*x(3);  
        dv22 = a*dj2*x(3);  % Perubahan bobot dari unit input 2
       
        % Perubahan bias
        dv01 = a*dj1;
        dv02 = a*dj2; 
        
        % Bobot baru ke unit output
        w11 = wb(2)+dw11; 
        w21 = wb(3)+dw21; 
        
        w01 = wb(1)+dw01; % Bias baru ke unit output
        
        % Bobot baru ke hidden unit
        v11 = vb(2,1)+dv11; 
        v12 = vb(2,2)+dv12; 
        v21 = vb(3,1)+dv21;
        v22 = vb(3,2)+dv22;
        
        % Bias baru
        v01 = vb(1,1)+dv01;
        v02 = vb(1,2)+dv02;
        
        % Matriks bobot v baru
        vb =[v01 v02; 
            v11 v12; 
            v21 v22];
        
        % Vektor bobot w baru
        wb = [w01 w11 w21];
    end     % Akhir dari satu epoch
    
    
    epoch = epoch+1;    % Epoch selanjutnya
    error(epoch) = (t(4)-y)^2;  % Hitung error kuadrat
end;

vb = vb % Matriks bobot akhir
wb = wb % Vektor bobot w akhir

iter = 1:epochmax;
plot(iter,error); % Gambarkan kurva error
title('Kurva Error');
ylabel('Error');
xlabel('Epoch');

% Pengujian dengan pola pelatihan (training)
for i = 1:4     % Untuk 4 pola uji dari pola yang dilatihkan 
    x = s(1,:);
    zin = x*vb;     % Lakukan langkah maju (feedforward)
    z1 = (2/(1+exp(-zin(1))))-1;
    z2 = (2/(1+exp(-zin(2))))-1;
    z = [1 z1 z2];
    yin = z*wb';
    
    % Hitung output
    y = (2/(1+exp(-yin)))-1
end;
      
        
        
