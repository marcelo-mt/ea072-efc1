% 05/05/2012
% gendata.m
% Gera��o com ou sem ru�do de dados de treinamento a partir de uma rede neural
% MLP com dimens�o 2x5x1
% Tipo de ru�do: uniformemente distribu�do no intervalo [-0.2,+0.2]
% Sa�da: arquivo train.mat
%
clear all;format long;format compact;
% Definindo a regi�o de amostragem
Min_ent = -4;
Max_ent = 4;
tam_amost = 25; % O no. de amostras vai ser 25*25 = 625
passo = (Max_ent-Min_ent)/(tam_amost-1);
% Gerando o vetor de amostras para cada dimens�o da entrada
v_aux = [Min_ent:passo:Max_ent];
% Definindo arbitrariamente os pesos da rede neural
V = [-0.7 0.1 0.8;-0.4 -0.2 -0.5;-0.2 0.7 -0.3;0.7 0.3 0.9;0.1 0.3 -0.5];
W = [1 0.9 -0.4 0.8 0.7 -0.8];
% Gerando os dados de treinamento (igualmente espa�ados)
disp('(1) Dados com ru�do');
disp('(2) Dados sem ru�do');
resp = input('Escolha uma das op��es acima: ');
if resp ~= 1 && resp ~= 2,
    error('Opc�o inv�lida!');
end
ind = 1;
for i=1:tam_amost,
	for j=1:tam_amost,
			x = [1;v_aux(i);v_aux(j)];
			z(j,i) = W*[1;tanh(V*x)];
			x1(ind,1) = v_aux(i);
			x2(ind,1) = v_aux(j);
            if resp == 1,
                z(j,i) = z(j,i)+(-0.2+0.4*rand(1,1));
            end
			y(ind,1) = z(j,i);
			ind = ind+1;
		end
	end
X = [x1 x2];S = y;
save train X S;
figure(1);mesh(v_aux,v_aux,z);
title('Mapeamento a ser aproximado');
